"""Python mirror of the on-device pipeline (ONNX models only). Used to cross-check the Kotlin engine."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import onnxruntime as ort
from scipy.signal import resample_poly

class OnnxPipeline:
    def __init__(self, d, threads=4):
        o = ort.SessionOptions(); o.intra_op_num_threads = threads
        mk = lambda n: ort.InferenceSession(f"{d}/{n}", o, providers=["CPUExecutionProvider"])
        self.bert, self.melo, self.extract, self.convert = mk("bert_kr.onnx"), mk("melo_kr.onnx"), mk("tone_extract.onnx"), mk("tone_convert.onnx")
        self.cfg = json.load(open(f"{M}/melo_kr/config.json"))
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(f"{M}/bert")
        self.src_se = src_se().numpy().astype(np.float32)
    def tts44(self, text, noise_scale=0.6, noise_scale_w=0.8, sdp_ratio=0.2, speed=1.0, frontend_out=None):
        norm, phones, ids, tones, langs, w2p = frontend_out or frontend(text, self.cfg)
        ii = self.tok(norm, return_tensors="np")["input_ids"].astype(np.int64)
        h = self.bert.run(None, {"input_ids": ii})[0]
        feats = np.repeat(h, w2p, axis=0).T[None].astype(np.float32)
        a, _ = self.melo.run(None, {"ids": np.array([ids], np.int64), "tones": np.array([tones], np.int64), "langs": np.array([langs], np.int64),
            "ja_bert": feats, "noise_scale": np.array(noise_scale, np.float32), "length_scale": np.array(1.0 / speed, np.float32),
            "noise_scale_w": np.array(noise_scale_w, np.float32), "sdp_ratio": np.array(sdp_ratio, np.float32)})
        return a[0]
    def se(self, wav22):
        return self.extract.run(None, {"wav": wav22[None].astype(np.float32)})[0][:, :, None]
    def convert22(self, wav22, tgt_se, tau=0.3):
        return self.convert.run(None, {"wav": wav22[None].astype(np.float32), "src_se": self.src_se, "tgt_se": tgt_se.astype(np.float32), "tau": np.array(tau, np.float32)})[0][0]
    def synth(self, text, tgt_se=None, **kw):
        a = self.tts44(text, **kw)
        w = resample_poly(a, 1, 2).astype(np.float32)
        return w if tgt_se is None else self.convert22(w, tgt_se)
