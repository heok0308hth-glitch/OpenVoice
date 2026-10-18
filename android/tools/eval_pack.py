import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from onnx_pipeline import *
import soundfile as sf, asr
lines = [l.strip() for l in open(os.path.join(os.path.dirname(__file__), "corpus.txt")) if l.strip()]
refw, _ = librosa.load("/home/user/OpenVoice/resources/demo_speaker0.mp3", sr=22050)
os.makedirs(f"{W}/out/pack", exist_ok=True)
for d in sys.argv[1:]:
    p = OnnxPipeline(f"{W}/{d}"); se = p.se(refw); cers = []
    for i, t in enumerate(lines):
        np.random.seed(i)
        o = p.synth(t, se); f = f"{W}/out/pack/{d}_{i:02d}.wav"; sf.write(f, o, 22050)
        h = asr.transcribe(f); c = asr.cer(t, h); cers.append(c)
        if c > 0.06: print(d, i, f"{c:.3f}", h, flush=True)
    print(d, "mean CER", np.mean(cers), flush=True)
