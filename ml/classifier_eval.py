"""Float baseline of the HF EfficientNet-B0 ASL classifier on a held-out slice of the
dataset, at 224 (its training size) and 128 (the board's size).
    .venv/bin/python -I ml/classifier_eval.py <hf-classifier dir> <dataset parquet> [n]
"""
import io, json, sys, time
import numpy as np, torch
from PIL import Image
from transformers import EfficientNetForImageClassification
import pyarrow.parquet as pq

model_dir, parquet, n = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1000
cfg = json.load(open(f"{model_dir}/config.json")); pre = json.load(open(f"{model_dir}/preprocessor_config.json"))
id2label = {int(k): v for k, v in cfg["id2label"].items()}
mean, std = np.array(pre["image_mean"], np.float32), np.array(pre["image_std"], np.float32)
model = EfficientNetForImageClassification.from_pretrained(model_dir).eval()
dev = "mps" if torch.backends.mps.is_available() else "cpu"; model.to(dev)
t = pq.read_table(parquet); labels_names = t.schema.metadata  # labels 0..25 = A..Z
rng = np.random.default_rng(0); idx = rng.permutation(t.num_rows); held = idx[: n]
imgs, labels = t.column("image").to_pylist(), t.column("label").to_pylist()
def run(size):
    correct = 0; t0 = time.time()
    for b in range(0, n, 32):
        batch = []
        for i in held[b:b+32]:
            im = Image.open(io.BytesIO(imgs[i]["bytes"])).convert("RGB").resize((size, size), Image.BOX)
            x = (np.asarray(im, np.float32) / 255 - mean) / std
            batch.append(torch.from_numpy(x).permute(2, 0, 1))
        with torch.no_grad():
            logits = model(pixel_values=torch.stack(batch).to(dev)).logits.cpu()
        pred = [id2label[int(p)] for p in logits.argmax(1)]
        correct += sum(p == "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[labels[i]] for p, i in zip(pred, held[b:b+32]))
    return correct / n, time.time() - t0
for size in (224, 128):
    acc, secs = run(size); print(f"size {size}: accuracy {acc:.3f} on {n} held-out images ({secs:.0f} s, {dev})")
sizes = [Image.open(io.BytesIO(imgs[i]["bytes"])).size for i in held[:20]]
print("sample image sizes:", sizes[:8])
