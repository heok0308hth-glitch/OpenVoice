"""Build the on-device model pack from the fp32 ONNX exports."""
import sys, os, json, hashlib, onnx, logging
sys.path.insert(0, os.path.dirname(__file__))
from compress import compress
from gather_int8 import rewrite
logging.disable(logging.INFO)
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S, D, T = f"{W}/onnx", f"{W}/modelpack", f"{W}/tmpq"
os.makedirs(D, exist_ok=True); os.makedirs(T, exist_ok=True)
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer, DefaultWeightOnlyQuantConfig
# BERT: 8-bit block-wise MatMulNBits + int8 embedding gather
m = onnx.load(f"{S}/bert_kr.onnx")
q = MatMulNBitsQuantizer(m, algo_config=DefaultWeightOnlyQuantConfig(block_size=32, is_symmetric=True, accuracy_level=4, bits=8)); q.process()
onnx.save(q.model.model, f"{T}/bert_a.onnx")
rewrite(f"{T}/bert_a.onnx", f"{T}/bert_b.onnx")
compress(f"{T}/bert_b.onnx", f"{D}/bert_kr.onnx", "fp16")
# MeloTTS-KR: int8 for decoder + flow, fp16 for text encoder & duration predictors (keeps durations stable)
compress(f"{S}/melo_kr.onnx", f"{D}/melo_kr.onnx", "int8", int8_prefixes=["m.dec.", "m.flow."])
# Tone color converter: int8 decoder + flow, fp16 posterior encoder, fp32 STFT basis
compress(f"{S}/tone_convert.onnx", f"{D}/tone_convert.onnx", "int8", skip=("stft.basis",), int8_prefixes=["m.dec.", "m.flow."])
compress(f"{S}/tone_extract.onnx", f"{D}/tone_extract.onnx", "fp16", skip=("stft.basis",))
manifest = {}
for f in sorted(os.listdir(D)):
    if f.endswith(".onnx"):
        p = f"{D}/{f}"; manifest[f] = {"size": os.path.getsize(p), "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()}
        print(f, round(manifest[f]["size"] / 1e6, 1), "MB")
json.dump(manifest, open(f"{D}/manifest.json", "w"), indent=1)
