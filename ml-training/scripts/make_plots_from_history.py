# save as: make_plots_from_history.py
import json, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras

# --- PATHS ---
HISTORY_JSON = "checkpoints/tiny_mobilenetv2/history.json"
MODEL_PATH   = "checkpoints/tiny_mobilenetv2/best.keras"                # or best_by_acc.keras
DATA_ROOT    = "/root/pokedex_BA/ml-training/data/tiny-imagenet-200"    # Tiny-ImageNet root
OUT_DIR      = Path("runs/offline_plots")                               # where PNGs will be written
# ------------------

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 1) training_curves.png from history.json ----------
with open(HISTORY_JSON) as f:
    H = json.load(f)

def get(key):
    return H.get(key, []) or []

epochs = np.arange(1, max(len(get("loss")), len(get("val_loss")), len(get("accuracy")), len(get("val_accuracy"))) + 1)

plt.figure(figsize=(9,7))

# loss
plt.subplot(3,1,1)
if get("loss"):      plt.plot(epochs[:len(get("loss"))], get("loss"), label="train")
if get("val_loss"):  plt.plot(epochs[:len(get("val_loss"))], get("val_loss"), label="val")
plt.ylabel("loss"); plt.legend(loc="best")

# accuracy
plt.subplot(3,1,2)
if get("accuracy"):     plt.plot(epochs[:len(get("accuracy"))], get("accuracy"), label="train")
if get("val_accuracy"): plt.plot(epochs[:len(get("val_accuracy"))], get("val_accuracy"), label="val")
plt.ylabel("accuracy"); plt.legend(loc="best")

# lr if present
plt.subplot(3,1,3)
if get("lr"): plt.plot(epochs[:len(get("lr"))], get("lr"), label="lr")
plt.xlabel("epoch"); plt.ylabel("lr")

plt.tight_layout()
plt.savefig(OUT_DIR / "training_curves.png", dpi=180)
print(f"✔ saved {OUT_DIR/'training_curves.png'}")

# ---------- 2) confusion_matrix.png (optional) ----------
# Only runs if MODEL_PATH and DATA_ROOT exist.
if Path(MODEL_PATH).exists() and Path(DATA_ROOT).exists():
    model = keras.models.load_model(MODEL_PATH)

    root = Path(DATA_ROOT)
    wnids = [w.strip() for w in (root / "wnids.txt").read_text().splitlines() if w.strip()]
    wnid_to_index = {wnid:i for i,wnid in enumerate(wnids)}

    # map val filename -> class index from val_annotations.txt
    ann = (root / "val" / "val_annotations.txt").read_text().splitlines()
    val_map = {}
    for ln in ann:
        if not ln.strip(): continue
        fname, wnid, *_ = ln.split("\t")
        val_map[fname] = wnid_to_index[wnid]

    val_img_dir = root / "val" / "images"

    #files = sorted([str(val_img_dir / f) for f in os.listdir(val_img_dir) if f.endswith(".JPEG")])
    files = sorted([str(val_img_dir / f) for f in os.listdir(val_img_dir) if f.endswith(".JPEG")])
    labels = [val_map[os.path.basename(f)] for f in files]  # compute labels in Python

    def load_img(fp):
        b = tf.io.read_file(fp)
        x = tf.image.decode_jpeg(b, channels=3)
        x = tf.image.resize(x, (224,224), antialias=True)
        return tf.cast(x, tf.float32) / 255.0

    # build a tf.data pipeline for val
    ds = tf.data.Dataset.from_tensor_slices((files, labels))
    ds = ds.map(
        lambda p, l: (load_img(p), tf.cast(l, tf.int32)),
        num_parallel_calls=tf.data.AUTOTUNE
    ).batch(64).prefetch(2)

    y_true, y_pred = [], []
    for xb, yb in ds:
        p = model.predict(xb, verbose=0)
        y_true.append(yb.numpy())
        y_pred.append(np.argmax(p, axis=1))
    y_true = np.concatenate(y_true, 0)
    y_pred = np.concatenate(y_pred, 0)

    cm = confusion_matrix(y_true, y_pred)
    cmn = cm / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(8,6))
    plt.imshow(cmn, interpolation="nearest")
    plt.title("Normalized Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=180)
    print(f"✔ saved {OUT_DIR/'confusion_matrix.png'}")
else:
    print("Skipped confusion_matrix: set MODEL_PATH and DATA_ROOT to valid paths.")
