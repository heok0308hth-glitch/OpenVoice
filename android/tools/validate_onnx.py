import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import onnxruntime as ort, time
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(W, "onnx")
def sess(n):
    o = ort.SessionOptions(); o.intra_op_num_threads = 4
    return ort.InferenceSession(f"{D}/{n}", o, providers=["CPUExecutionProvider"])
def snr(a, b):
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    return 10 * np.log10(np.sum(a**2) / (np.sum((a - b) ** 2) + 1e-20))
text = "어제 친구와 함께 한강 공원에서 자전거를 탔습니다. 정말 즐거웠어요!"
model, cfg = load_melo()
norm, phones, ids, tones, langs, w2p = frontend(text, cfg)
# BERT
tok, bm = bert_model()
ref_bert = bert_feature(norm, w2p)
ii = tok(norm, return_tensors="np")["input_ids"]
t = time.time(); h = sess("bert_kr.onnx").run(None, {"input_ids": ii.astype(np.int64)})[0]; tb = time.time() - t
feats = np.concatenate([np.repeat(h[i:i+1], w2p[i], 0) for i in range(len(w2p))], 0).T
print("bert max abs diff", np.abs(feats - ref_bert.numpy()).max(), "cos", (feats * ref_bert.numpy()).sum() / np.linalg.norm(feats) / np.linalg.norm(ref_bert.numpy()), f"{tb:.3f}s")
# Melo deterministic (noise=0)
with torch.no_grad():
    ref = model.infer(torch.LongTensor(ids)[None], torch.LongTensor([len(ids)]), torch.LongTensor([0]), torch.LongTensor(tones)[None],
                      torch.LongTensor(langs)[None], torch.zeros(1, 1024, len(ids)), ref_bert[None], sdp_ratio=0.2, noise_scale=0.0,
                      noise_scale_w=0.0, length_scale=1.0)[0][0, 0].numpy()
s = sess("melo_kr.onnx")
feed = {"ids": np.array([ids], np.int64), "tones": np.array([tones], np.int64), "langs": np.array([langs], np.int64),
        "ja_bert": feats[None].astype(np.float32), "noise_scale": np.array(0.0, np.float32), "length_scale": np.array(1.0, np.float32),
        "noise_scale_w": np.array(0.0, np.float32), "sdp_ratio": np.array(0.2, np.float32)}
t = time.time(); a, dur = s.run(None, feed); tm = time.time() - t
print("melo len", len(ref), a.shape[1], "SNR dB", snr(ref, a[0]), f"{tm:.2f}s for {a.shape[1]/44100:.2f}s audio (RTF {tm/(a.shape[1]/44100):.3f})")
# Converter
conv = load_converter()
wav22 = librosa.resample(ref, orig_sr=44100, target_sr=22050).astype(np.float32)
refwav, _ = librosa.load("/home/user/OpenVoice/resources/demo_speaker0.mp3", sr=22050)
tse_ref = extract_se_wave(conv, refwav).numpy()
t = time.time(); tse = sess("tone_extract.onnx").run(None, {"wav": refwav[None].astype(np.float32)})[0]; te = time.time() - t
print("se max diff", np.abs(tse - tse_ref[:, :, 0]).max(), f"{te:.2f}s for {len(refwav)/22050:.1f}s ref")
sse = src_se().numpy()
oref = convert_wave(conv, wav22, torch.tensor(sse), torch.tensor(tse_ref), tau=0.0)
s = sess("tone_convert.onnx")
t = time.time(); o = s.run(None, {"wav": wav22[None], "src_se": sse.astype(np.float32), "tgt_se": tse[:, :, None].astype(np.float32), "tau": np.array(0.0, np.float32)})[0]; tc = time.time() - t
print("conv SNR dB", snr(oref, o[0]), f"{tc:.2f}s (RTF {tc/(len(wav22)/22050):.3f})")
