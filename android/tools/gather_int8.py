"""Rewrite Gather(float_table, idx) -> Cast(Gather(int8_table, idx)) * Unsqueeze(Gather(scale, idx)).
The table stays int8 in memory and only the looked-up rows are dequantized (no full-table transient)."""
import onnx, numpy as np, sys
from onnx import numpy_helper, helper, TensorProto
def rewrite(src, dst, min_elems=100000):
    m = onnx.load(src); g = m.graph
    inits = {i.name: i for i in g.initializer}
    new_nodes, removed, new_inits = [], set(), []
    out_nodes = []
    for n in g.node:
        if n.op_type == "Gather" and n.input[0] in inits and inits[n.input[0]].data_type == TensorProto.FLOAT:
            arr = numpy_helper.to_array(inits[n.input[0]])
            ax = next((a.i for a in n.attribute if a.name == "axis"), 0)
            if arr.ndim == 2 and ax == 0 and arr.size >= min_elems:
                name = n.input[0]
                amax = np.abs(arr).max(axis=1); scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
                q = np.clip(np.round(arr / scale[:, None]), -127, 127).astype(np.int8)
                if name not in removed:
                    new_inits += [numpy_helper.from_array(q, name + "_gq"), numpy_helper.from_array(scale, name + "_gs")]
                    removed.add(name)
                o = n.output[0]
                out_nodes += [
                    helper.make_node("Gather", [name + "_gq", n.input[1]], [o + "_qi8"], axis=0),
                    helper.make_node("Cast", [o + "_qi8"], [o + "_qf"], to=TensorProto.FLOAT),
                    helper.make_node("Gather", [name + "_gs", n.input[1]], [o + "_sc"], axis=0),
                    helper.make_node("Unsqueeze", [o + "_sc", "gq_axes_m1"], [o + "_sc2"]),
                    helper.make_node("Mul", [o + "_qf", o + "_sc2"], [o]),
                ]
                continue
        out_nodes.append(n)
    if removed:
        new_inits.append(numpy_helper.from_array(np.array([-1], np.int64), "gq_axes_m1"))
    keep = [i for i in g.initializer if i.name not in removed]
    del g.initializer[:]; g.initializer.extend(keep + new_inits)
    del g.node[:]; g.node.extend(out_nodes)
    onnx.save(m, dst); print("gather-int8:", sorted(removed))
if __name__ == "__main__":
    rewrite(sys.argv[1], sys.argv[2])
