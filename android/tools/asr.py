"""ASR oracle: transcribe Korean audio with Whisper and compute CER against the reference text."""
import sys, re, os
from faster_whisper import WhisperModel
import jiwer
_m = None
def model():
    global _m
    if _m is None:
        _m = WhisperModel("mobiuslabsgmbh/faster-whisper-large-v3-turbo", device="cpu", compute_type="int8", cpu_threads=4)
    return _m
def norm(s):
    return re.sub(r"[^가-힣a-z0-9]", "", s.lower())
def transcribe(path):
    segs, _ = model().transcribe(path, language="ko", beam_size=5, condition_on_previous_text=False)
    return "".join(s.text for s in segs).strip()
def cer(ref, hyp):
    return jiwer.cer(norm(ref), norm(hyp))
if __name__ == "__main__":
    ref = sys.argv[1]
    for p in sys.argv[2:]:
        h = transcribe(p); print(f"{os.path.basename(p)}\tCER={cer(ref,h):.3f}\t{h}")
