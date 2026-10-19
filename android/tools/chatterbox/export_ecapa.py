"""Export SpeechBrain ECAPA-TDNN (spkrec-ecapa-voxceleb, Apache-2.0) as an independent on-device speaker judge:
wav16k[1,S] -> embedding[1,192] (Fbank + sentence mean normalisation + ECAPA), checked against encode_batch.
usage: export_ecapa.py OUT_DIR SAVEDIR wav"""
import sys, numpy as np, torch, librosa
from speechbrain.inference.speaker import EncoderClassifier
OUT, SAVE, WAV = sys.argv[1], sys.argv[2], sys.argv[3]
enc = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=SAVE, run_opts={"device": "cpu"})

from speechbrain.processing.features import spectral_magnitude

class Wrap(torch.nn.Module):
    """torch.stft does not export, so the STFT (n_fft 400, hop 160, periodic Hamming, centred, zero padding)
    is a strided convolution with a DFT basis; the mel filterbank/log/top-dB part is SpeechBrain's own."""
    def __init__(self, m):
        super().__init__(); self.fb = m.mods.compute_features.compute_fbanks; self.e = m.mods.embedding_model
        n = torch.arange(400).float(); k = torch.arange(201).float()
        win = torch.hamming_window(400)
        ang = 2 * np.pi * k[:, None] * n[None, :] / 400
        self.register_buffer("basis", (torch.cat([torch.cos(ang), -torch.sin(ang)], 0) * win[None, :])[:, None, :])
    def forward(self, wav):
        y = torch.nn.functional.pad(wav[:, None, :], (200, 200))
        o = torch.nn.functional.conv1d(y, self.basis, stride=160)          # [B, 402, T]
        stft = torch.stack([o[:, :201], o[:, 201:]], -1).transpose(1, 2)   # [B, T, F, 2]
        feats = self.fb(spectral_magnitude(stft))
        feats = feats - feats.mean(dim=1, keepdim=True)        # InputNormalization(sentence, std_norm=False)
        return self.e(feats).squeeze(1)

w = Wrap(enc).eval()
x, _ = librosa.load(WAV, sr=16000); x = torch.tensor(x[:16000 * 6])[None]
with torch.no_grad():
    ref = enc.encode_batch(x).squeeze(1); got = w(x)
    print("torch wrapper cos", float(torch.nn.functional.cosine_similarity(ref, got)))
    torch.onnx.export(w, (x,), f"{OUT}/ecapa.onnx", input_names=["wav"], output_names=["embedding"],
                      dynamic_axes={"wav": {1: "S"}}, opset_version=17, dynamo=False)
import onnxruntime as ort
s = ort.InferenceSession(f"{OUT}/ecapa.onnx", providers=["CPUExecutionProvider"])
for n in (16000 * 3, 16000 * 9):
    y, _ = librosa.load(WAV, sr=16000); y = y[:n].astype(np.float32)
    o = s.run(None, {"wav": y[None]})[0]
    with torch.no_grad(): r = enc.encode_batch(torch.tensor(y)[None]).squeeze(1).numpy()
    print(n, "onnx vs speechbrain cos", float((o @ r.T) / np.linalg.norm(o) / np.linalg.norm(r)))
