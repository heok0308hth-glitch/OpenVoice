"""Pure-ONNX Chatterbox Multilingual pipeline (reference for the Kotlin engine).
Mirrors ChatterboxMultilingualTTS.generate(): CFG, repetition penalty, temperature, min_p, top_p, and the
AlignmentStreamAnalyzer heuristics (from the 3 alignment heads exported by t3_lm.onnx)."""
import json, unicodedata, numpy as np, onnxruntime as ort, librosa
from tokenizers import Tokenizer

S3GEN_SR, S3_SR = 24000, 16000


class CBOnnx:
    def __init__(self, d, enc_dir, lm="t3_lm_q4.onnx", threads=4, decoder="conditional_decoder.onnx"):
        so = ort.SessionOptions(); so.intra_op_num_threads = threads
        mk = lambda p: ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])
        self.lm = mk(f"{d}/{lm}"); self.cond = mk(f"{d}/t3_cond.onnx")
        self.campplus = mk(f"{d}/campplus.onnx"); self.s3mel = mk(f"{d}/s3_mel.onnx")
        self.vemel = mk(f"{d}/ve_mel.onnx"); self.velstm = mk(f"{d}/ve_lstm.onnx")
        self.tokzr = mk(f"{enc_dir}/speech_encoder.onnx"); self.dec = mk(f"{enc_dir}/{decoder}")
        hdr = json.load(open(f"{d}/t3_embed.json")); raw = np.fromfile(f"{d}/t3_embed.bin", dtype="<f2")
        def tab(k):
            h = hdr[k]; n = int(np.prod(h["shape"])); o = h["offset"] // 2
            return raw[o:o + n].reshape(h["shape"]).astype(np.float32)
        self.TE, self.TP, self.SE, self.SP = tab("text_emb"), tab("text_pos_emb"), tab("speech_emb"), tab("speech_pos_emb")
        self.tok = Tokenizer.from_file(f"{d}/tokenizer.json")
        self.outnames = [o.name for o in self.lm.get_outputs()]

    # ------------------------------------------------------------- voice registration
    def speech_tokens(self, wav24):
        return self.tokzr.run(None, {"audio_values": wav24[None].astype(np.float32)})[1]

    def ve_embed(self, wav16):
        y, _ = librosa.effects.trim(wav16, top_db=20)
        mel = self.vemel.run(None, {"wav": y[None].astype(np.float32)})[0][0]
        step, win = 80, 160
        n = len(mel)
        nw = (n - win + step) // step if n >= win else 0
        rem = (n - win + step) % step if n >= win else n
        if nw == 0 or (rem + (win - step)) / win >= 0.8: nw += 1
        target = win + step * (nw - 1)
        if n < target: mel = np.concatenate([mel, np.zeros((target - n, 40), np.float32)])
        parts = np.stack([mel[i * step:i * step + win] for i in range(nw)])
        e = self.velstm.run(None, {"partials": parts.astype(np.float32)})[0].mean(0)
        return e / np.linalg.norm(e)

    def make_voice(self, wav24_full):
        wav24 = wav24_full[: 10 * S3GEN_SR].astype(np.float32)
        wav16_full = librosa.resample(wav24_full, orig_sr=S3GEN_SR, target_sr=S3_SR).astype(np.float32)
        wav16 = librosa.resample(wav24, orig_sr=S3GEN_SR, target_sr=S3_SR).astype(np.float32)
        gen_tokens = self.speech_tokens(wav24)
        mel = self.s3mel.run(None, {"wav": wav24[None]})[0]
        if mel.shape[1] != 2 * gen_tokens.shape[1]:
            gen_tokens = gen_tokens[:, : mel.shape[1] // 2]
        t3_tokens = self.speech_tokens(wav24_full[: 6 * S3GEN_SR].astype(np.float32))[:, :150]
        return dict(ve=self.ve_embed(wav16_full)[None].astype(np.float32),
                    xvec=self.campplus.run(None, {"wav": wav16[None]})[0].astype(np.float32),
                    gen_tokens=gen_tokens.astype(np.int64), gen_mel=mel.astype(np.float32),
                    t3_tokens=t3_tokens.astype(np.int64))

    # ------------------------------------------------------------- text
    def text_tokens(self, text):
        t = unicodedata.normalize("NFKD", text.lower())
        t = "[ko]" + t
        t = t.replace(" ", "[SPACE]")
        return np.array([255] + self.tok.encode(t).ids + [0], np.int64)

    # ------------------------------------------------------------- generation
    def generate(self, text, voice, exaggeration=0.5, cfg_weight=0.5, temperature=0.8, repetition_penalty=2.0,
                 min_p=0.05, top_p=1.0, max_new_tokens=1000, rng=None, use_analyzer=True):
        rng = rng or np.random.default_rng(0)
        tt = self.text_tokens(text)
        cond = self.cond.run(None, {"speaker_emb": voice["ve"], "prompt_tokens": voice["t3_tokens"],
                                    "exaggeration": np.array([exaggeration], np.float32)})[0][0]
        te = self.TE[tt] + self.TP[np.arange(len(tt))]; tu = self.TP[np.arange(len(tt))]
        bos = (self.SE[6561] + self.SP[0])[None]
        emb = np.stack([np.concatenate([cond, te, bos]), np.concatenate([cond, tu, bos])]).astype(np.float32)
        i0, j0 = len(cond), len(cond) + len(tt)
        past = {f"past_{kv}_{i}": np.zeros((2, 16, 0, 64), np.float32) for i in range(30) for kv in "kv"}
        out = self.lm.run(None, {"inputs_embeds": emb, **past})
        gen = [6561]; S = j0 - i0
        A = np.zeros((0, S), np.float32); pos = 0; text_pos = 0; started = False; complete = False; completed_at = None
        for step in range(max_new_tokens):
            lg = out[0]; att = out[1].mean(0)  # [Tq, Ttot]
            if use_analyzer:
                chunk = att[j0:, i0:j0].copy() if step == 0 else att[:, i0:j0].copy()
                chunk[:, pos + 1:] = 0
                A = np.concatenate([A, chunk]); T = len(A)
                cur = int(chunk[-1].argmax())
                if -4 < cur - text_pos < 7: text_pos = cur
                false_start = (not started) and (A[-2:, -2:].max() > 0.1 or A[:, :4].max() < 0.5)
                started = not false_start
                complete = complete or text_pos >= S - 3
                if complete and completed_at is None: completed_at = T
                long_tail = complete and A[completed_at:, -3:].sum(0).max() >= 5
                rep = complete and A[completed_at:, :-5].max(1).sum() > 5
            logits = lg[0] + cfg_weight * (lg[0] - lg[1])
            if use_analyzer:
                tok_rep = len(gen) >= 3 and gen[-1] == gen[-2] and step > 0
                if cur < S - 3 and S > 5: logits[6562] = -2 ** 15
                if long_tail or rep or tok_rep:
                    logits[:] = -2 ** 15; logits[6562] = 2 ** 15
                pos += 1
            # repetition penalty over generated ids (incl. BOS), HF semantics
            ids = np.unique(gen); s = logits[ids]
            logits[ids] = np.where(s < 0, s * repetition_penalty, s / repetition_penalty)
            logits = logits / temperature
            p = np.exp(logits - logits.max()); p /= p.sum()
            # min_p
            keep = p >= min_p * p.max()
            logits = np.where(keep, logits, -np.inf)
            if top_p < 1.0:
                o = np.argsort(-logits); cp = np.cumsum(np.exp(logits[o] - logits[o][0]) / np.exp(logits - logits.max()).sum())
                rm = o[cp > top_p][1:]; logits[rm] = -np.inf
            p = np.exp(logits - logits.max()); p /= p.sum()
            tok = int(rng.choice(len(p), p=p)); gen.append(tok)
            if tok == 6562: break
            e = (self.SE[tok] + self.SP[step + 1])[None, None].repeat(2, 0).astype(np.float32)
            pres = {n.replace("present", "past"): v for n, v in zip(self.outnames[2:], out[2:])}
            out = self.lm.run(None, {"inputs_embeds": e, **pres})
        sp = np.array([t for t in gen[1:] if t < 6561], np.int64)
        return sp

    def decode(self, speech_tokens, voice):
        toks = np.concatenate([voice["gen_tokens"], speech_tokens[None]], 1)
        return self.dec.run(None, {"speech_tokens": toks, "speaker_embeddings": voice["xvec"], "speaker_features": voice["gen_mel"]})[0][0]

    def tts(self, text, voice, **kw):
        return self.decode(self.generate(text, voice, **kw), voice)
