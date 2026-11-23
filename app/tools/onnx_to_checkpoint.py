"""Rebuild Model A's state_dict from the ONNX export (weights are identical; MatMul weights are W^T)."""
import sys, json
from dataclasses import asdict
import numpy as np, onnx, torch
from onnx import numpy_helper
sys.path.insert(0, ".")
from app.models.model_a_arch import ModelAConfig, ScratchTransformerClassifier

m = onnx.load(sys.argv[1]); g = m.graph
inits = {t.name: numpy_helper.to_array(t) for t in g.initializer}
consumers = {}
for n in g.node:
    for i in n.input: consumers.setdefault(i, []).append(n)
state = {}
for name, arr in inits.items():
    if name.startswith("model."):
        state[name[len("model."):]] = torch.from_numpy(arr.copy())
for n in g.node:
    if n.op_type == "MatMul" and n.input[1] in inits and not n.input[1].startswith("model."):
        adds = [c for c in consumers.get(n.output[0], []) if c.op_type == "Add"]
        bias = [i for i in adds[0].input if i.startswith("model.") and i.endswith(".bias")]
        assert len(bias) == 1, (n.name, bias)
        wname = bias[0][len("model."):-len(".bias")] + ".weight"
        state[wname] = torch.from_numpy(inits[n.input[1]].T.copy())
    if n.op_type == "Gemm":
        print("Gemm", n.input, [ (a.name, a.i) for a in n.attribute])
cfg = ModelAConfig()
model = ScratchTransformerClassifier(cfg)
ref = model.state_dict()
missing = sorted(set(ref) - set(state)); extra = sorted(set(state) - set(ref))
print("missing", missing, "extra", extra)
for k in ref: assert ref[k].shape == state[k].shape, (k, ref[k].shape, state[k].shape)
model.load_state_dict(state, strict=True)
torch.save({"model_state_dict": model.state_dict(), "model_cfg": asdict(cfg), "global_step": None,
            "best_val_macro_f1": None, "reconstructed_from": "onnx/model_a.onnx"}, sys.argv[2])
print("saved", sys.argv[2], sum(p.numel() for p in model.parameters()))
