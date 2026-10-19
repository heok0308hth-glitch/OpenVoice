"""Independent scoring of v2 outputs: Whisper CER, ECAPA similarity to held-out real clips, F0 stats.
usage: score_v2.py W outdir systems(A,B,C) sentences.json"""
import sys, os, glob, re, json, numpy as np, librosa, torch
from speechbrain.inference.speaker import EncoderClassifier
from faster_whisper import WhisperModel
import jiwer
W, D = sys.argv[1], sys.argv[2]; systems = sys.argv[3].split(","); lines = json.load(open(sys.argv[4]))
spks = sys.argv[5].split(",") if len(sys.argv) > 5 else ["spk0", "spk1", "spk2", "spk3"]
ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=f"{W}/models/ecapa", run_opts={"device": "cpu"})
asr = WhisperModel("mobiuslabsgmbh/faster-whisper-large-v3-turbo", device="cpu", compute_type="int8", cpu_threads=4)
def emb(path):
    x, _ = librosa.load(path, sr=16000)
    with torch.no_grad(): e = ecapa.encode_batch(torch.tensor(x)[None]).squeeze().numpy()
    return e / np.linalg.norm(e)
def f0stats(path):
    x, sr = librosa.load(path, sr=16000)
    f0, vf, _ = librosa.pyin(x, fmin=60, fmax=500, sr=sr, frame_length=1024)
    f = f0[vf & ~np.isnan(f0)]
    if len(f) < 10: return None
    st = 12 * np.log2(f / 55.0); return np.median(st), np.std(st)
def norm(s): return re.sub(r"[^가-힣a-z0-9]", "", s.lower())
held = {}
for s in spks:
    hs = sorted(glob.glob(f"{W}/kref/{s}_held*.wav"))
    H = np.stack([emb(h) for h in hs]); hc = H.mean(0); hc /= np.linalg.norm(hc)
    hf = [x for x in (f0stats(h) for h in hs) if x]
    held[s] = (hc, np.median([x[0] for x in hf]), np.mean([x[1] for x in hf]))
for sysname in systems:
    per = {}
    for s in spks:
        hc, hmed, hstd = held[s]; cers, sims, dm, sr_ = [], [], [], []
        for i, t in enumerate(lines):
            p = f"{D}/{s}_{sysname}_{i}.wav"
            if not os.path.exists(p): continue
            segs, _ = asr.transcribe(p, language="ko", beam_size=5, condition_on_previous_text=False)
            h = "".join(x.text for x in segs); cers.append(jiwer.cer(norm(t), norm(h)))
            sims.append(float(emb(p) @ hc)); f = f0stats(p)
            if f: dm.append(abs(f[0] - hmed)); sr_.append(f[1] / hstd)
            if cers[-1] > 0.15: print("  hi-CER", s, sysname, i, repr(h))
        per[s] = (np.mean(cers), np.mean(sims), np.mean(dm), np.mean(sr_))
    allv = np.array(list(per.values()))
    print(sysname, "CER=%.3f sim=%.3f f0err=%.2fst spread=%.2f" % tuple(allv.mean(0)), {k: "%.3f" % v[1] for k, v in per.items()}, flush=True)
