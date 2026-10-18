import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import soundfile as sf, asr, numpy as np
model, cfg = load_melo()
lines = [l.strip() for l in open(os.path.join(os.path.dirname(__file__), "corpus.txt")) if l.strip()]
os.makedirs(f"{W}/out/eval", exist_ok=True)
res = {"bert": [], "zero": []}
for i, t in enumerate(lines):
    for mode in res:
        for seed in (0, 1):
            a = melo_tts(model, cfg, t, zero_bert=(mode == "zero"), seed=seed)
            p = f"{W}/out/eval/{i:02d}_{mode}_{seed}.wav"; sf.write(p, a, 44100)
            h = asr.transcribe(p); c = asr.cer(t, h); res[mode].append(c)
            if c > 0: print(i, mode, seed, f"{c:.3f}", h, flush=True)
for m, v in res.items(): print(m, "mean CER", np.mean(v))
