"""Weight-only compression of ONNX graphs: large float initializers are stored as int8 (per-channel symmetric)
or fp16 and expanded by DequantizeLinear / Cast nodes. Compute stays fp32, so numerics are close to the original."""
import sys, onnx, numpy as np
from onnx import numpy_helper, helper, TensorProto

AXIS = {("Conv", 1): 0, ("ConvTranspose", 1): 1, ("MatMul", 1): 1, ("Gemm", 1): 0, ("Gather", 0): 0}

def compress(src, dst, mode="int8", min_elems=4096, skip=(), int8_prefixes=None):
    m = onnx.load(src)
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    uses = {}
    for n in g.node:
        for k, inp in enumerate(n.input):
            if inp in inits:
                uses.setdefault(inp, []).append((n.op_type, k))
    new_inits, new_nodes, removed = [], [], set()
    for name, init in inits.items():
        if init.data_type != TensorProto.FLOAT or name in skip:
            continue
        arr = numpy_helper.to_array(init)
        if arr.size < min_elems:
            continue
        u = uses.get(name, [])
        kinds = {AXIS.get(x) for x in u}
        use_int8 = mode == "int8" and len(u) > 0 and None not in kinds and len(kinds) == 1
        if use_int8 and int8_prefixes is not None:
            use_int8 = any(name.startswith(p) for p in int8_prefixes)
        if mode == "int8" and not use_int8 and int8_prefixes is None:
            continue
        if use_int8:
            axis = kinds.pop()
            red = tuple(i for i in range(arr.ndim) if i != axis)
            amax = np.abs(arr).max(axis=red, keepdims=True)
            scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
            q = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
            new_inits += [numpy_helper.from_array(q, name + "_q"), numpy_helper.from_array(scale.reshape(-1), name + "_s"),
                          numpy_helper.from_array(np.zeros(scale.size, np.int8), name + "_z")]
            new_nodes.append(helper.make_node("DequantizeLinear", [name + "_q", name + "_s", name + "_z"], [name], axis=axis, name=name + "_dq"))
        else:  # fp16
            new_inits.append(numpy_helper.from_array(arr.astype(np.float16), name + "_h"))
            new_nodes.append(helper.make_node("Cast", [name + "_h"], [name], to=TensorProto.FLOAT, name=name + "_cast"))
        removed.add(name)
    keep = [i for i in g.initializer if i.name not in removed]
    del g.initializer[:]
    g.initializer.extend(keep + new_inits)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(new_nodes + nodes)
    onnx.save(m, dst)
    print(dst, "compressed", len(removed), "tensors")

if __name__ == "__main__":
    compress(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "int8", skip=("stft.basis",))
