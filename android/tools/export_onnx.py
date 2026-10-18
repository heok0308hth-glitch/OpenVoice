"""Export the OpenVoice V2 Korean pipeline to ONNX (fp32).
Graphs:
  bert_kr.onnx        input_ids[1,T] -> hidden[T,768]           (kykim/bert-kor-base, first 10 layers == hidden_states[-3])
  melo_kr.onnx        ids,tones,langs[1,N], ja_bert[1,768,N], scales -> audio44k[1,S]
  tone_extract.onnx   wav22k[1,S] -> se[1,256]                  (STFT as conv + ReferenceEncoder)
  tone_convert.onnx   wav22k[1,S], src_se, tgt_se, tau -> wav22k[1,S']
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import torch.nn as nn, torch.nn.functional as F

OUT = os.path.join(W, "onnx"); os.makedirs(OUT, exist_ok=True)

def strip_weight_norm(model):
    for m in model.modules():
        if hasattr(m, "weight_g"):
            torch.nn.utils.remove_weight_norm(m)

class STFTMag(nn.Module):
    """|STFT| implemented as a strided conv1d (exactly OpenVoice spectrogram_torch, center=False, reflect pad)."""
    def __init__(self, n_fft=1024, hop=256, win=1024):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        window = torch.hann_window(win)
        n = torch.arange(n_fft).float(); k = torch.arange(n_fft // 2 + 1).float()
        ang = 2 * math.pi * k[:, None] * n[None, :] / n_fft
        basis = torch.cat([torch.cos(ang), -torch.sin(ang)], 0) * window[None, :]
        self.register_buffer("basis", basis[:, None, :])
    def forward(self, y):  # y [1,S]
        p = (self.n_fft - self.hop) // 2
        y = F.pad(y[:, None, :], (p, p), mode="reflect")
        out = F.conv1d(y, self.basis, stride=self.hop)
        re, im = out[:, : self.n_fft // 2 + 1], out[:, self.n_fft // 2 + 1 :]
        return torch.sqrt(re * re + im * im + 1e-6)  # [1,513,T]

class MeloWrap(nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m
        with torch.no_grad():
            self.register_buffer("g", m.emb_g(torch.LongTensor([0])).unsqueeze(-1))
    def forward(self, ids, tones, langs, ja_bert, noise_scale, length_scale, noise_scale_w, sdp_ratio):
        m = self.m
        x_lengths = torch.full((1,), ids.shape[1], dtype=torch.long)
        bert = torch.zeros(1, 1024, ids.shape[1])
        g = self.g
        x, m_p, logs_p, x_mask = m.enc_p(ids, x_lengths, tones, langs, bert, ja_bert, g=g)
        logw = m.sdp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w) * sdp_ratio + m.dp(x, x_mask, g=g) * (1 - sdp_ratio)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        from melo import commons as mc
        y_mask = torch.unsqueeze(mc.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = mc.generate_path(w_ceil, attn_mask)
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = m.flow(z_p, y_mask, g=g, reverse=True)
        o = m.dec(z * y_mask, g=g)
        return o[:, 0, :], w_ceil[:, 0, :]

class ExtractWrap(nn.Module):
    def __init__(self, conv):
        super().__init__(); self.stft = STFTMag(); self.ref_enc = conv.model.ref_enc
    def forward(self, wav):
        spec = self.stft(wav)
        return self.ref_enc(spec.transpose(1, 2))  # [1,256]

class ConvertWrap(nn.Module):
    def __init__(self, conv):
        super().__init__(); self.stft = STFTMag(); self.m = conv.model
    def forward(self, wav, src_se, tgt_se, tau):
        m = self.m
        spec = self.stft(wav)
        lengths = torch.full((1,), spec.shape[2], dtype=torch.long)
        g_src, g_tgt = src_se, tgt_se
        z, m_q, logs_q, y_mask = m.enc_q(spec, lengths, g=torch.zeros_like(g_src), tau=tau)
        z_p = m.flow(z, y_mask, g=g_src)
        z_hat = m.flow(z_p, y_mask, g=g_tgt, reverse=True)
        o = m.dec(z_hat * y_mask, g=torch.zeros_like(g_tgt))
        return o[:, 0, :]

class BertWrap(nn.Module):
    def __init__(self, bert):
        super().__init__(); self.bert = bert
    def forward(self, input_ids):
        out = self.bert(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), token_type_ids=torch.zeros_like(input_ids))
        return out.last_hidden_state[0]

def export(model, args, path, names_in, names_out, dyn):
    torch.onnx.export(model, args, path, input_names=names_in, output_names=names_out, dynamic_axes=dyn,
                      opset_version=17, do_constant_folding=True, dynamo=False)
    print("exported", path, os.path.getsize(path) / 1e6, "MB")

if __name__ == "__main__":
    which = sys.argv[1:] or ["bert", "melo", "extract", "convert"]
    with torch.no_grad():
        if "bert" in which:
            from transformers import BertModel
            b = BertModel.from_pretrained(f"{M}/bert", add_pooling_layer=False).eval()
            b.encoder.layer = b.encoder.layer[:10]
            ids = torch.LongTensor([[2, 1000, 2000, 3000, 3]])
            export(BertWrap(b), (ids,), f"{OUT}/bert_kr.onnx", ["input_ids"], ["hidden"], {"input_ids": {1: "T"}, "hidden": {0: "T"}})
        if "melo" in which:
            model, cfg = load_melo(); strip_weight_norm(model)
            _, _, ids, tones, langs, w2p = frontend("안녕하세요. 반갑습니다.", cfg)
            N = len(ids)
            a = (torch.LongTensor(ids)[None], torch.LongTensor(tones)[None], torch.LongTensor(langs)[None], torch.randn(1, 768, N),
                 torch.tensor(0.6), torch.tensor(1.0), torch.tensor(0.8), torch.tensor(0.2))
            export(MeloWrap(model), a, f"{OUT}/melo_kr.onnx",
                   ["ids", "tones", "langs", "ja_bert", "noise_scale", "length_scale", "noise_scale_w", "sdp_ratio"], ["audio", "durations"],
                   {"ids": {1: "N"}, "tones": {1: "N"}, "langs": {1: "N"}, "ja_bert": {2: "N"}, "audio": {1: "S"}, "durations": {1: "N"}})
        if "extract" in which or "convert" in which:
            conv = load_converter(); strip_weight_norm(conv.model)
            wav = torch.randn(1, 22050 * 2) * 0.1
            if "extract" in which:
                export(ExtractWrap(conv), (wav,), f"{OUT}/tone_extract.onnx", ["wav"], ["se"], {"wav": {1: "S"}})
            if "convert" in which:
                se = torch.randn(1, 256, 1)
                export(ConvertWrap(conv), (wav, se, se, torch.tensor(0.3)), f"{OUT}/tone_convert.onnx",
                       ["wav", "src_se", "tgt_se", "tau"], ["out"], {"wav": {1: "S"}, "out": {1: "S2"}})
