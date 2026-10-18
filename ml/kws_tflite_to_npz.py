#!/usr/bin/env python3
"""Pull the trained DS-CNN "small" weights out of Arm ML-Zoo's fp32 TFLite file into an npz
that ml/kws_export.py (project venv, PyTorch, no TensorFlow) turns into model/ds_cnn_kws.pt2.

    tools-venv/bin/python -I ml/kws_tflite_to_npz.py ds_cnn_s.tflite kws_ref.npz out.npz

Needs TensorFlow (the tools venv, not the project venv). Walks the ops in execution order so the
layer mapping does not depend on tensor names: CONV_2D (stem, weights OHWI [64,10,4,1]), then
four pairs of DEPTHWISE_CONV_2D ([1,3,3,64]) + CONV_2D (pointwise [64,1,1,64]), then
FULLY_CONNECTED ([12,64]); batch norm is already folded into the weights and biases. Also runs the
TFLite interpreter on the reference MFCCs of kws_ref.npz and stores its softmax outputs, so the
PyTorch port can be checked without TensorFlow.

Source: https://github.com/ARM-software/ML-zoo/tree/master/models/keyword_spotting/ds_cnn_small
(Apache-2.0; fp32 93.89 %, int8 93.11 % on the Speech Commands v2 test set, per its README).
"""
import sys

import numpy as np
import tensorflow as tf

tflite_path, ref_path, out_path = sys.argv[1:4]
it = tf.lite.Interpreter(model_path=tflite_path)
it.allocate_tensors()
ops = it._get_ops_details()  # execution order; private API, fine for a one-off tool
tensor = it.get_tensor
out = {}
blocks = 0
for op in ops:
    name = op["op_name"]
    if name == "CONV_2D":
        w, b = tensor(op["inputs"][1]), tensor(op["inputs"][2])
        if w.shape[1:] == (10, 4, 1):
            out["stem_w"], out["stem_b"] = w, b  # OHWI
        else:
            out[f"pw{blocks}_w"], out[f"pw{blocks}_b"] = w, b  # OHWI [64,1,1,64]
            blocks += 1
    elif name == "DEPTHWISE_CONV_2D":
        out[f"dw{blocks}_w"], out[f"dw{blocks}_b"] = tensor(op["inputs"][1]), tensor(op["inputs"][2])  # [1,3,3,C]
    elif name == "FULLY_CONNECTED":
        out["fc_w"], out["fc_b"] = tensor(op["inputs"][1]), tensor(op["inputs"][2])
assert blocks == 4 and "stem_w" in out and "fc_w" in out, sorted(out)
print("ops:", [o["op_name"] for o in ops])
print({k: v.shape for k, v in out.items()})

ref = np.load(ref_path)
inp, outp = it.get_input_details()[0], it.get_output_details()[0]
probs = []
for m in ref["mfcc"]:
    it.set_tensor(inp["index"], m.reshape(1, 490).astype(np.float32))
    it.invoke()
    probs.append(it.get_tensor(outp["index"])[0].copy())
out["ref_probs"] = np.stack(probs)
out["ref_words"] = ref["words"]
labels = ["_silence_", "_unknown_", "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
top = [labels[int(p.argmax())] for p in probs]
print("interpreter top-1 on the reference clips:", sum(t == w for t, w in zip(top, ref["words"])), "/", len(top))
for t, w in zip(top, ref["words"]):
    if t != w:
        print("  miss:", w, "->", t)
np.savez(out_path, **out)
