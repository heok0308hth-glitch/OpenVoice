"""Compress Chatterbox ONNX graphs for mobile: MatMul -> MatMulNBits (8-bit, block 32), remaining large
float initializers (Conv etc.) -> int8 DequantizeLinear or fp16 Cast (weight-only; compute stays fp32)."""
import sys, os, logging, onnx
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from compress import compress
logging.disable(logging.INFO)
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer, DefaultWeightOnlyQuantConfig

def nbits(src, dst, bits=8):
    m = onnx.load(src)
    q = MatMulNBitsQuantizer(m, algo_config=DefaultWeightOnlyQuantConfig(block_size=32, is_symmetric=True, accuracy_level=4, bits=bits))
    q.process()
    q.model.save_model_to_file(dst, use_external_data_format=False)

if __name__ == "__main__":
    src, dst, mode = sys.argv[1], sys.argv[2], sys.argv[3]
    tmp = dst + ".tmp.onnx"
    if mode.startswith("nb"):
        bits = int(mode[2:3]); rest = mode[4:] if len(mode) > 3 else "none"   # e.g. nb8, nb8_int8, nb4_fp16
        nbits(src, tmp, bits)
        if rest in ("int8", "fp16"): compress(tmp, dst, rest); os.remove(tmp)
        else: os.replace(tmp, dst)
    else:
        compress(src, dst, mode)
    print(dst, round(os.path.getsize(dst) / 1e6, 1), "MB")
