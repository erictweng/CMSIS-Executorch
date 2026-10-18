"""MediaPipe hand_landmark_lite (224x224): onnx2torch module -> model/hand_landmark_224.pt2.

Same recipe as ml/blazepalm_export.py (whose `exportable()` does the module fix-ups): the
graph is checked against the TFLite reference outputs, saved as a self-contained exported
program, reloaded and checked again. Inputs are float in [0, 1] (MediaPipe's graph uses
output_tensor_float_range 0..1 for this model too). Outputs, in the TFLite order:
landmarks (63: 21 x (x, y, z) in input pixels, z in the same units), presence logit-free
score (1, already a sigmoid), handedness (1, sigmoid, 1 = right), world landmarks (63, metres).

    .venv/bin/python -I ml/handlandmark_export.py <hand_landmark_lite.onnx> <lm_ref.npz> [out.pt2]
"""
import sys
from pathlib import Path

import numpy as np, onnx, torch
from onnx2torch import convert

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blazepalm_export import exportable  # noqa: E402

REF_KEYS = ("Identity", "Identity_1", "Identity_2", "Identity_3")


def check(outs, ref, i):
    """Max abs diff per output against the reference, outputs matched by position."""
    return [float(np.abs(o.numpy()[0] - ref[k][i]).max()) for o, k in zip(outs, REF_KEYS)]


def main():
    onnx_path, ref_path = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else "model/hand_landmark_224.pt2"
    ref = np.load(ref_path)
    xs = torch.from_numpy(ref["x_u8"]).float().div(255)
    net = convert(onnx.load(onnx_path)).eval()
    net, counts = exportable(net, xs[0:1])
    print("replaced:", counts)
    worst = np.zeros(4)
    for i in range(len(xs)):
        with torch.no_grad(): outs = net(xs[i:i + 1])
        assert len(outs) == 4 and [tuple(o.shape) for o in outs] == [(1, 63), (1, 1), (1, 1), (1, 63)], [o.shape for o in outs]
        worst = np.maximum(worst, check(outs, ref, i))
    print("vs TFLite on %d inputs, worst abs diff: landmarks %.2e px, presence %.2e, handedness %.2e, world %.2e"
          % (len(xs), *worst))
    assert worst[0] < 5e-2 and worst[1] < 1e-3 and worst[2] < 1e-3 and worst[3] < 1e-3, worst
    ep = torch.export.export(net, (xs[0:1],))
    for node in ep.graph_module.graph.nodes:  # onnx2torch names break (de)serialisation
        node.meta.pop("nn_module_stack", None)
        node.meta.pop("source_fn_stack", None)
    torch.export.save(ep, out)
    reloaded = torch.export.load(out).module()
    with torch.no_grad(): outs = reloaded(xs[0:1])
    assert max(check(outs, ref, 0)) < 5e-2
    print("saved and reloaded", out)


if __name__ == "__main__":
    main()
