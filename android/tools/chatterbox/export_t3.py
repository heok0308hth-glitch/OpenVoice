"""Export Chatterbox Multilingual T3 (t3_mtl23ls_v2) for on-device inference.

Outputs (in OUT):
  t3_lm.onnx      inputs_embeds[B,T,1024] + past_k/v_{0..29}[B,16,P,64] ->
                  logits[B,8194] (last position), align_attn[3,T,P+T] (batch 0, heads used by the
                  alignment analyzer), present_k/v_{0..29}
  t3_cond.onnx    speaker_emb[1,256], prompt_tokens[1,150], exaggeration[1] -> cond_emb[1,34,1024]
  t3_embed.bin    fp16 tables: text_emb, text_pos_emb, speech_emb, speech_pos_emb (+ header json)
A minimal Llama forward is used (instead of HF classes) so the KV cache and attention probabilities are
explicit graph tensors; it is verified against the original model in verify_t3.py.
"""
import os, sys, json, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(4)
import perth
perth.PerthImplicitWatermarker = perth.DummyWatermarker
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
ALIGNED = [(12, 15), (13, 11), (9, 2)]


class MiniLlama(nn.Module):
    def __init__(self, tfmr, head):
        super().__init__()
        self.tfmr = tfmr
        self.head = head
        cfg = tfmr.config
        self.nl, self.nh, self.hd = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.head_dim
        self.eps = cfg.rms_norm_eps
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        inv_freq, scaling = ROPE_INIT_FUNCTIONS["llama3"](cfg, "cpu")
        self.register_buffer("inv_freq", inv_freq.float())
        self.att_scaling = float(scaling)

    def rms(self, x, w):
        v = x.pow(2).mean(-1, keepdim=True)
        return w * (x * torch.rsqrt(v + self.eps))

    @staticmethod
    def rot(x):
        h = x.shape[-1] // 2
        return torch.cat((-x[..., h:], x[..., :h]), -1)

    def forward(self, inputs_embeds, *past):
        B, T, _ = inputs_embeds.shape
        P = past[0].shape[2]
        pos = torch.arange(T, dtype=torch.float32) + P
        freqs = pos[:, None] * self.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), -1)
        cos = (emb.cos() * self.att_scaling)[None, None]
        sin = (emb.sin() * self.att_scaling)[None, None]
        kpos = torch.arange(T + P)
        qpos = torch.arange(T) + P
        mask = torch.where(kpos[None, :] > qpos[:, None], torch.tensor(-1e9), torch.tensor(0.0))[None, None]
        h = inputs_embeds
        presents, attns = [], {}
        for li, layer in enumerate(self.tfmr.layers):
            a = layer.self_attn
            x = self.rms(h, layer.input_layernorm.weight)
            q = a.q_proj(x).view(B, T, self.nh, self.hd).transpose(1, 2)
            k = a.k_proj(x).view(B, T, self.nh, self.hd).transpose(1, 2)
            v = a.v_proj(x).view(B, T, self.nh, self.hd).transpose(1, 2)
            q = q * cos + self.rot(q) * sin
            k = k * cos + self.rot(k) * sin
            k = torch.cat((past[2 * li], k), 2)
            v = torch.cat((past[2 * li + 1], v), 2)
            presents += [k, v]
            s = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.hd) + mask
            p = torch.softmax(s.float(), -1)
            for j, (L, H) in enumerate(ALIGNED):
                if L == li:
                    attns[j] = p[0, H]
            o = torch.matmul(p, v).transpose(1, 2).reshape(B, T, self.nh * self.hd)
            h = h + a.o_proj(o)
            x = self.rms(h, layer.post_attention_layernorm.weight)
            m = layer.mlp
            h = h + m.down_proj(F.silu(m.gate_proj(x)) * m.up_proj(x))
        h = self.rms(h, self.tfmr.norm.weight)
        logits = self.head(h[:, -1, :])
        align = torch.stack([attns[0], attns[1], attns[2]])
        return (logits, align, *presents)


class CondWrap(nn.Module):
    def __init__(self, t3):
        super().__init__()
        self.t3 = t3

    def forward(self, speaker_emb, prompt_tokens, exaggeration):
        t3 = self.t3
        ce = t3.cond_enc
        spk = ce.spkr_enc(speaker_emb)[:, None]
        pe = t3.speech_emb(prompt_tokens) + t3.speech_pos_emb(prompt_tokens)
        pe = ce.perceiver(pe)
        emo = ce.emotion_adv_fc(exaggeration.view(-1, 1, 1))
        return torch.cat((spk, pe, emo), 1)


if __name__ == "__main__":
    # load only T3 (keeps memory low enough for the legacy exporter)
    from pathlib import Path
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from chatterbox.models.t3 import T3
    from chatterbox.models.t3.modules.t3_config import T3Config
    ckpt = Path(snapshot_download("ResembleAI/chatterbox", allow_patterns=["t3_mtl23ls_v2.safetensors"]))
    t3 = T3(T3Config.multilingual())
    sd = load_file(ckpt / "t3_mtl23ls_v2.safetensors")
    if "model" in sd.keys(): sd = sd["model"][0]
    t3.load_state_dict(sd); t3 = t3.eval()
    with torch.no_grad():
        # embeddings
        tabs = {"text_emb": t3.text_emb.weight, "text_pos_emb": t3.text_pos_emb.emb.weight,
                "speech_emb": t3.speech_emb.weight, "speech_pos_emb": t3.speech_pos_emb.emb.weight}
        header, off = {}, 0
        with open(f"{OUT}/t3_embed.bin", "wb") as f:
            for k, w in tabs.items():
                a = w.detach().float().numpy().astype("<f2")
                f.write(a.tobytes()); header[k] = {"shape": list(a.shape), "offset": off}; off += a.nbytes
        hp = t3.hp
        header["hp"] = dict(start_text=hp.start_text_token, stop_text=hp.stop_text_token, start_speech=hp.start_speech_token,
                            stop_speech=hp.stop_speech_token, prompt_len=hp.speech_cond_prompt_len)
        json.dump(header, open(f"{OUT}/t3_embed.json", "w"), indent=1)
        print("tables", header)
        # cond
        cw = CondWrap(t3).eval()
        args = (torch.randn(1, 256), torch.randint(0, 6561, (1, 150)), torch.tensor([0.5]))
        print("cond out", cw(*args).shape)
        torch.onnx.export(cw, args, f"{OUT}/t3_cond.onnx", input_names=["speaker_emb", "prompt_tokens", "exaggeration"],
                          output_names=["cond_emb"], dynamic_axes={"prompt_tokens": {1: "L"}}, opset_version=17, dynamo=False)
        # LM
        lm = MiniLlama(t3.tfmr, t3.speech_head).eval()
        B, T, P = 2, 5, 3
        past = []
        for _ in range(lm.nl):
            past += [torch.randn(B, 16, P, 64) * 0.1, torch.randn(B, 16, P, 64) * 0.1]
        names_in = ["inputs_embeds"] + [f"past_{kv}_{i}" for i in range(lm.nl) for kv in ("k", "v")]
        names_out = ["logits", "align_attn"] + [f"present_{kv}_{i}" for i in range(lm.nl) for kv in ("k", "v")]
        dyn = {"inputs_embeds": {0: "B", 1: "T"}, "align_attn": {1: "T", 2: "S"}, "logits": {0: "B"}}
        for n in names_in[1:]:
            dyn[n] = {0: "B", 2: "P"}
        for n in names_out[2:]:
            dyn[n] = {0: "B", 2: "S"}
        torch.onnx.export(lm, (torch.randn(B, T, 1024), *past), f"{OUT}/t3_lm_fp32.onnx", input_names=names_in,
                          output_names=names_out, dynamic_axes=dyn, opset_version=17, dynamo=False)
        print("exported")
