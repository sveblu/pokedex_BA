import os, csv, pathlib, random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore

# ---------- config ----------
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "tiny-imagenet-200"

IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS_WARMUP = 5          # BN-only warm-up
EPOCHS_FT = 20             # fine-tune
SEED = 42

CLASSES = 5
TRAIN_PER_CLASS = 200
VAL_PER_CLASS   = 50

AUTOTUNE = tf.data.AUTOTUNE
random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.optimizer.set_jit(False)  # keep it simple
except Exception:
    pass

# ---------- data helpers ----------
def read_wnids():
    wnids = (DATA_DIR / "wnids.txt").read_text().strip().splitlines()
    return wnids[:CLASSES]

def gather_train_files(wnids):
    paths, labels = [], []
    for cid, wnid in enumerate(wnids):
        d = DATA_DIR / "train" / wnid / "images"
        imgs = sorted([*d.glob("*.JPEG"), *d.glob("*.jpg")])[:TRAIN_PER_CLASS]
        paths += [str(p) for p in imgs]
        labels += [cid] * len(imgs)
    return paths, labels

def gather_val_files(wnids):
    ann = (DATA_DIR / "val" / "val_annotations.txt").read_text().splitlines()
    wnid_to_id = {w: i for i, w in enumerate(wnids)}
    images_dir = DATA_DIR / "val" / "images"
    per_class = {i: [] for i in range(len(wnids))}
    for line in ann:
        parts = line.split("\t")
        fname, wnid = parts[0], parts[1]
        if wnid in wnid_to_id:
            cid = wnid_to_id[wnid]
            p = images_dir / fname
            if p.exists():
                per_class[cid].append(str(p))
    paths, labels = [], []
    for cid in range(len(wnids)):
        take = per_class[cid][:VAL_PER_CLASS]
        paths += take
        labels += [cid] * len(take)
    return paths, labels

def decode(path, label, train=False):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)        # [0..255]
    img = keras.applications.mobilenet_v2.preprocess_input(img)  # -> [-1,1]
    return img, label

# ---------- model ----------
def build_model(num_classes: int):
    inp = layers.Input(shape=(*IMG_SIZE, 3), dtype=tf.float32)
    base = keras.applications.MobileNetV2(
        include_top=False, input_shape=(*IMG_SIZE, 3), weights=None
    )
    x = base(inp)  # Keras toggles training/eval; we'll control BN trainable flags
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inp, out)
    return model, base

def count_trainables(model: keras.Model) -> int:
    return int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))

# ---------- train ----------
def main():
    wnids = read_wnids()
    print("WNIDs (order -> class_id):")
    for i, w in enumerate(wnids):
        print(f"  {i} -> {w}")

    tr_paths, tr_labels = gather_train_files(wnids)
    va_paths, va_labels = gather_val_files(wnids)
    print(f"\nTrain samples: {len(tr_paths)}  Val samples: {len(va_paths)}")
    print("Class counts (train):", {i: tr_labels.count(i) for i in range(len(wnids))})
    print("Class counts (val):  ", {i: va_labels.count(i) for i in range(len(wnids))})

    train_ds = (tf.data.Dataset.from_tensor_slices((tr_paths, tr_labels))
                .shuffle(len(tr_paths), seed=SEED, reshuffle_each_iteration=True)
                .map(lambda p,y: decode(p,y, train=True), num_parallel_calls=AUTOTUNE)
                .batch(BATCH)
                .prefetch(AUTOTUNE))
    val_ds = (tf.data.Dataset.from_tensor_slices((va_paths, va_labels))
              .map(lambda p,y: decode(p,y, train=False), num_parallel_calls=AUTOTUNE)
              .batch(BATCH)
              .prefetch(AUTOTUNE))

    model, base = build_model(len(wnids))

    # ---- WARM-UP: Update ONLY BatchNorm moving stats (convs frozen)
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
        else:
            layer.trainable = False

    # call with training=True so BN updates moving_mean/var
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"],
                  run_eagerly=False)
    print(f"[diag] trainables (warm-up): {count_trainables(model)} params (should be BN + head)")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_WARMUP,
              verbose=2)

    # ---- FINE-TUNE: Unfreeze convs, FREEZE all BN
    base.trainable = True
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    print(f"[diag] trainables (fine-tune): {count_trainables(model)} params (millions expected)")
    hist = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FT,
                     verbose=2)

    # --- diagnostics: predictions distribution on one batch
    def dist_info(name, preds):
        mean = preds.mean(axis=0)
        counts = tf.math.bincount(tf.argmax(preds, axis=1), minlength=preds.shape[1]).numpy()
        print(f"[diag] {name} softmax mean:", np.round(mean, 3))
        print(f"[diag] {name} argmax counts:", counts)

    for xb, yb in val_ds.take(1):
        p_inf = model(xb, training=False).numpy()
        p_trn = model(xb, training=True ).numpy()
        dist_info("VAL (training=False)", p_inf)
        dist_info("VAL (training=True) ", p_trn)
        break

    for xb, yb in train_ds.take(1):
        p_inf = model(xb, training=False).numpy()
        p_trn = model(xb, training=True ).numpy()
        dist_info("TRAIN (training=False)", p_inf)
        dist_info("TRAIN (training=True) ", p_trn)
        break

    tr = model.evaluate(train_ds, verbose=0)
    va = model.evaluate(val_ds, verbose=0)
    print(f"\nEVAL — train: acc={tr[1]:.3f}, val: acc={va[1]:.3f}")

    print("\nLast 5 epochs (acc/val_acc):")
    for a, va in zip(hist.history["accuracy"][-5:], hist.history["val_accuracy"][-5:]):
        print(f"  {a:.3f} / {va:.3f}")

if __name__ == "__main__":
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    main()
