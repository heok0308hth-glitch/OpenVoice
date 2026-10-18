import sys, os, onnx, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from compress import compress
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer, DefaultWeightOnlyQuantConfig
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = f"{W}/onnx/bert_kr.onnx"; os.makedirs(f"{W}/tmpq", exist_ok=True)
# variant A: int4 block MatMulNBits for encoder + int8 DQ for embeddings
m = onnx.load(src)
q = MatMulNBitsQuantizer(m, block_size=32, is_symmetric=True, accuracy_level=4)
q.process(); onnx.save(q.model.model, f"{W}/tmpq/bert_int4_tmp.onnx")
compress(f"{W}/tmpq/bert_int4_tmp.onnx", f"{W}/tmpq/bert_int4.onnx", "int8")
# variant B: ORT dynamic int8 (MatMulInteger) + int8 DQ embeddings
quantize_dynamic(src, f"{W}/tmpq/bert_dyn_tmp.onnx", weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"], per_channel=False)
compress(f"{W}/tmpq/bert_dyn_tmp.onnx", f"{W}/tmpq/bert_dyn.onnx", "int8")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(f"{W}/models/bert")
texts = [l.strip() for l in open(f"{W}/ref/corpus.txt") if l.strip()]
def s(p):
    o = ort.SessionOptions(); o.intra_op_num_threads = 4; return ort.InferenceSession(p, o)
base = s(src)
import time
for name in ["onnx_int8/bert_kr.onnx", "tmpq/bert_int4.onnx", "tmpq/bert_dyn.onnx"]:
    p = f"{W}/{name}"; ss = s(p); cs = []; tt = 0
    for t in texts:
        ii = tok(t, return_tensors="np")["input_ids"].astype(np.int64)
        a = base.run(None, {"input_ids": ii})[0]
        t0 = time.time(); b = ss.run(None, {"input_ids": ii})[0]; tt += time.time() - t0
        cs.append(np.mean(np.sum(a * b, -1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)))
    print(f"{name}: {os.path.getsize(p)/1e6:.1f}MB mean token cos {np.mean(cs):.5f} min {np.min(cs):.5f} time {tt:.2f}s")
