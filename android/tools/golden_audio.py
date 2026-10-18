"""Deterministic (noise=0) reference outputs of the ONNX model pack for the Kotlin parity test."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from onnx_pipeline import *
import soundfile as sf
p = OnnxPipeline(f"{W}/modelpack")
texts = ["안녕하세요. 오늘은 날씨가 정말 좋네요.", "어제 친구와 함께 한강 공원에서 자전거를 탔습니다."]
out = f"{W}/golden_audio"
for i, t in enumerate(texts):
    a = p.tts44(t, noise_scale=0.0, noise_scale_w=0.0, sdp_ratio=0.2)
    sf.write(f"{out}/base44_{i}.wav", a, 44100, subtype="FLOAT")
ref, _ = librosa.load(f"{out}/ref_speaker0.wav", sr=22050)
se = p.se(ref)[0, :, 0]
np.save(f"{out}/se_speaker0_whole.npy", se)
json.dump(texts, open(f"{out}/texts.json", "w"), ensure_ascii=False)
print("ok", se[:4])
