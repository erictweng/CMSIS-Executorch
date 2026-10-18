"""BlazePalm: onnx2torch module -> self-contained model/blazepalm_192.pt2 (torch.export).

onnx2torch's Reshape, Resize and PReLU converters branch on tensor values, which
torch.export cannot guard; they are replaced by fixed-shape equivalents recorded on one
trace run (batch 1). The result is checked against the TFLite reference outputs.

    .venv/bin/python -I ml/blazepalm_export.py <palm_detection_lite.onnx> <blazepalm_ref.npz> [out.pt2]
"""
import sys
import numpy as np, onnx, torch, torch.nn as nn, torch.nn.functional as F
from onnx2torch import convert


class FixedReshape(nn.Module):
    def __init__(self, shape): super().__init__(); self.shape = tuple(int(s) for s in shape)
    def forward(self, x, *rest): return x.reshape(self.shape)


class Up2Bilinear(nn.Module):  # ONNX Resize linear / half_pixel, scales (1, 1, 2, 2) on NCHW
    def forward(self, x, *rest): return F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)


class PReLUNative(nn.Module):
    def forward(self, x, slope): return F.prelu(x, slope.reshape(-1))


class FixedPad(nn.Module):  # ONNX Pad with its pads given as a constant tensor input
    def __init__(self, pads, mode, value): super().__init__(); self.pads, self.mode, self.value = pads, mode, value
    def forward(self, x, *rest): return F.pad(x, self.pads, mode=self.mode, value=self.value)


def exportable(net: torch.fx.GraphModule, example: torch.Tensor) -> tuple[torch.fx.GraphModule, dict]:
    shapes = {}
    hooks = []
    for node in net.graph.nodes:
        if node.op == "call_module" and type(net.get_submodule(node.target)).__name__.lower().startswith("onnxreshape"):
            hooks.append(net.get_submodule(node.target).register_forward_hook(
                lambda mod, inp, out, name=node.target: shapes.__setitem__(name, tuple(out.shape))))
    with torch.no_grad():
        net(example)
    for h in hooks: h.remove()
    from onnx2torch.node_converters.pad import _onnx_padding_to_torch, _torch_padding_to_mode_format

    buffers = dict(net.named_buffers())
    counts = {}
    for node in list(net.graph.nodes):
        if node.op != "call_module": continue
        sub = net.get_submodule(node.target)
        cls = type(sub).__name__.lower()
        if cls.startswith("onnxreshape"): repl = FixedReshape(shapes[node.target])
        elif cls.startswith("onnxresize"): repl = Up2Bilinear()
        elif cls.startswith("onnxprelu"): repl = PReLUNative()
        elif cls.startswith("onnxpaddynamic"):
            pads = buffers[node.args[1].target].tolist()
            value = float(node.args[2]) if len(node.args) > 2 and not isinstance(node.args[2], torch.fx.Node) else 0.0
            repl = FixedPad(_torch_padding_to_mode_format(_onnx_padding_to_torch(pads), sub.mode), sub.mode, value)
        else: continue
        name = node.target.replace(".", "_").replace("/", "_") + "_x"
        setattr(net, name, repl); node.target = name
        counts[type(repl).__name__] = counts.get(type(repl).__name__, 0) + 1
    # torch.export.save's weight packager overflows on onnx2torch's initializers (views
    # sharing storage): give every tensor its own contiguous storage.
    for mod in net.modules():
        for name, p in list(mod.named_parameters(recurse=False)):
            setattr(mod, name, nn.Parameter(p.detach().clone().contiguous(), requires_grad=False))
        for name, b in list(mod.named_buffers(recurse=False)):
            mod.register_buffer(name, b.detach().clone().contiguous())
    net.recompile()
    return net, counts


def main():
    onnx_path, ref_path = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else "model/blazepalm_192.pt2"
    ref = np.load(ref_path)
    net = convert(onnx.load(onnx_path)).eval()
    x = torch.from_numpy(ref["x"][0:1])
    net, counts = exportable(net, x)
    print("replaced:", counts)
    worst_s = worst_r = 0.0
    for i, xi in enumerate(ref["x"]):
        with torch.no_grad(): outs = net(torch.from_numpy(xi[None]))
        by = {o.shape[-1]: o.numpy()[0] for o in outs}
        worst_s = max(worst_s, float(np.abs(by[1] - ref["scores"][i]).max()))
        worst_r = max(worst_r, float(np.abs(by[18] - ref["regressors"][i]).max() / max(1.0, np.abs(ref["regressors"][i]).max())))
    print(f"vs TFLite on {len(ref['x'])} inputs: worst score-logit diff {worst_s:.1e}, worst relative regressor diff {worst_r:.1e}")
    assert worst_s < 1e-3 and worst_r < 1e-4
    ep = torch.export.export(net, (x,))
    # onnx2torch's module names hold '/', ';' and ',', which the nn_module_stack
    # metadata format cannot carry through save/load; it is provenance only.
    for node in ep.graph_module.graph.nodes:
        node.meta.pop("nn_module_stack", None)
        node.meta.pop("source_fn_stack", None)
    torch.export.save(ep, out)
    reloaded = torch.export.load(out).module()
    with torch.no_grad():
        outs = reloaded(x)
    by = {o.shape[-1]: o.numpy()[0] for o in outs}
    assert float(np.abs(by[1] - ref["scores"][0]).max()) < 1e-3
    print("saved and reloaded", out)


if __name__ == "__main__":
    main()
