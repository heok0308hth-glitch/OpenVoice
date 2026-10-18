import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from compress import compress
import onnxruntime as ort
def sess(p):
    o = ort.SessionOptions(); o.intra_op_num_threads = 4
    return ort.InferenceSession(p, o, providers=["CPUExecutionProvider"])
def logmel(x, sr=22050):
    S = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
    return np.log10(np.maximum(S, 1e-5))
def mel_dist(a, b):
    n = min(len(a), len(b)); A, B = logmel(a[:n]), logmel(b[:n]); return float(np.mean(np.abs(A - B)))
def snr(a, b):
    n = min(len(a), len(b)); a, b = a[:n], b[:n]; return 10 * np.log10(np.sum(a**2) / (np.sum((a - b) ** 2) + 1e-20))
wav, _ = librosa.load(f"{W}/out/ref_melo.wav", sr=22050); wav = wav.astype(np.float32)
refw, _ = librosa.load("/home/user/OpenVoice/resources/demo_speaker0.mp3", sr=22050)
tse = sess(f"{W}/onnx/tone_extract.onnx").run(None, {"wav": refw[None].astype(np.float32)})[0][:, :, None]
sse = src_se().numpy().astype(np.float32)
feed = {"wav": wav[None], "src_se": sse, "tgt_se": tse, "tau": np.array(0.0, np.float32)}
ref = sess(f"{W}/onnx/tone_convert.onnx").run(None, feed)[0][0]
# noise floor: fp32 with tau=0.3 (the normal stochastic setting) vs tau=0
f2 = dict(feed); f2["tau"] = np.array(0.3, np.float32)
o = sess(f"{W}/onnx/tone_convert.onnx").run(None, f2)[0][0]
print(f"fp32 tau0.3 vs tau0: SNR {snr(ref,o):.1f} mel {mel_dist(ref,o):.4f}")
os.makedirs(f"{W}/tmpq", exist_ok=True)
for name, prefixes in [("all_fp16", []), ("dec", ["m.dec."]), ("enc_q", ["m.enc_q."]), ("flow", ["m.flow."]), ("dec+enc_q", ["m.dec.", "m.enc_q."]), ("all", None)]:
    p = f"{W}/tmpq/c_{name}.onnx"
    compress(f"{W}/onnx/tone_convert.onnx", p, "int8", skip=("stft.basis",), int8_prefixes=prefixes)
    o = sess(p).run(None, feed)[0][0]
    print(f"{name}: size {os.path.getsize(p)/1e6:.1f}MB SNR {snr(ref,o):.1f} mel {mel_dist(ref,o):.4f}", flush=True)
