import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from compress import compress
from conv_quant_eval import sess, mel_dist, snr
model, cfg = load_melo()
text = "어제 친구와 함께 한강 공원에서 자전거를 탔습니다. 정말 즐거웠어요!"
norm, phones, ids, tones, langs, w2p = frontend(text, cfg)
feats = bert_feature(norm, w2p).numpy()
feed = {"ids": np.array([ids], np.int64), "tones": np.array([tones], np.int64), "langs": np.array([langs], np.int64),
        "ja_bert": feats[None].astype(np.float32), "noise_scale": np.array(0.0, np.float32), "length_scale": np.array(1.0, np.float32),
        "noise_scale_w": np.array(0.0, np.float32), "sdp_ratio": np.array(0.2, np.float32)}
ref = sess(f"{W}/onnx/melo_kr.onnx").run(None, feed)[0][0]
f2 = dict(feed); f2["noise_scale"] = np.array(0.6, np.float32)
o = sess(f"{W}/onnx/melo_kr.onnx").run(None, f2)[0][0]
print(f"fp32 noise0.6 vs 0: len {len(o)} vs {len(ref)} mel {mel_dist(ref[::2],o[::2]):.4f}")
for name, prefixes in [("all_fp16", []), ("dec", ["m.dec."]), ("flow", ["m.flow."]), ("dec+flow", ["m.dec.", "m.flow."]), ("all", None)]:
    p = f"{W}/tmpq/m_{name}.onnx"
    compress(f"{W}/onnx/melo_kr.onnx", p, "int8", int8_prefixes=prefixes)
    o = sess(p).run(None, feed)[0][0]
    print(f"{name}: size {os.path.getsize(p)/1e6:.1f}MB len {len(o)} SNR {snr(ref,o):.1f} mel {mel_dist(ref[::2],o[::2]):.4f}", flush=True)
