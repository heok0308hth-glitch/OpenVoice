"""Make the S3Gen ONNX decoders deterministic like the original PyTorch code.

CosyVoice/Chatterbox's flow-matching decoder starts from a *fixed* noise buffer (rand_noise), but the published
ONNX graphs use RandomNormalLike, so every decode of the same tokens sounds slightly different. This replaces the
mel-shaped [1,80,T] RandomNormalLike nodes with slices of fixed Gaussian constants and the HiFT harmonic start phase
([1,9,1] RandomUniformLike) with a constant. Batch size 1 only (as used by the app).
usage: fix_decoder_noise.py in.onnx out.onnx"""
import sys, numpy as np, onnx
from onnx import helper, numpy_helper, TensorProto
src, dst = sys.argv[1], sys.argv[2]
MAXT = 4000
m = onnx.load(src)
g = m.graph
rng = np.random.default_rng(0)
new_nodes = []
k = 0
for n in list(g.node):
    if n.op_type == "RandomNormalLike" and not n.name.startswith("/m_source"):
        c = f"fixed_noise_{k}"; k += 1
        g.initializer.append(numpy_helper.from_array(rng.standard_normal((1, 80, MAXT)).astype(np.float32), c))
        sh, t, z, two, ax = (f"{c}_{s}" for s in ("shape", "T", "zero", "two", "axes"))
        g.initializer.extend([numpy_helper.from_array(np.array([0], np.int64), z), numpy_helper.from_array(np.array([2], np.int64), two),
                              numpy_helper.from_array(np.array([2], np.int64), ax)])
        repl = [helper.make_node("Shape", [n.input[0]], [sh]),
                helper.make_node("Gather", [sh, two], [t], axis=0),
                helper.make_node("Slice", [c, z, t, ax], [n.output[0]])]
        idx = list(g.node).index(n); g.node.remove(n)
        for j, r in enumerate(repl): g.node.insert(idx + j, r)
    elif n.op_type == "RandomUniformLike" and n.name.startswith("/m_source"):
        c = "fixed_phase"
        g.initializer.append(numpy_helper.from_array(rng.uniform(0, 1, (1, 9, 1)).astype(np.float32), c))
        idx = list(g.node).index(n); g.node.remove(n)
        g.node.insert(idx, helper.make_node("Identity", [c], [n.output[0]]))
onnx.checker.check_model(m)
onnx.save(m, dst)
print("replaced", k, "mel noise nodes ->", dst)
