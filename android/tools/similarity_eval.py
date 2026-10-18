"""Speaker-similarity experiments: timbre emphasis (alpha) and per-utterance source embedding."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from onnx_pipeline import *
import soundfile as sf, asr
p = OnnxPipeline(f"{W}/modelpack")
c = lambda a, b: float(np.dot(a.ravel(), b.ravel()) / np.linalg.norm(a) / np.linalg.norm(b))
lines = [l.strip() for l in open(os.path.join(os.path.dirname(__file__), "corpus.txt")) if l.strip()][:8]
refs = {"spk0": "/home/user/OpenVoice/resources/demo_speaker0.mp3", "example": "/home/user/OpenVoice/resources/example_reference.mp3",
        "spk2": "/home/user/OpenVoice/resources/demo_speaker2.mp3"}
bases = []
for i, t in enumerate(lines):
    np.random.seed(i); bases.append(resample_poly(p.tts44(t), 1, 2).astype(np.float32))
os.makedirs(f"{W}/out/sim", exist_ok=True)
for rn, rp in refs.items():
    ref, _ = librosa.load(rp, sr=22050)
    tse = p.se(ref)
    print(f"== {rn}: base(kr) vs target cos {c(p.src_se, tse):.3f}; unconverted base output vs target {np.mean([c(p.se(b), tse) for b in bases]):.3f}")
    for src_mode in ("kr", "utt"):
        for alpha in (1.0, 1.25, 1.5):
            sims, cers = [], []
            for i, (t, b) in enumerate(zip(lines, bases)):
                s = p.src_se if src_mode == "kr" else p.se(b)
                tgt = s + alpha * (tse - s)
                np.random.seed(100 + i)
                o = p.convert.run(None, {"wav": b[None], "src_se": s.astype(np.float32), "tgt_se": tgt.astype(np.float32), "tau": np.array(0.3, np.float32)})[0][0]
                sims.append(c(p.se(o), tse))
                f = f"{W}/out/sim/{rn}_{src_mode}_{alpha}_{i}.wav"; sf.write(f, o, 22050)
                cers.append(asr.cer(t, asr.transcribe(f)))
            print(f"  src={src_mode} alpha={alpha}: speaker cos {np.mean(sims):.4f}  CER {np.mean(cers):.3f}", flush=True)
