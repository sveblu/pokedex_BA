# scripts/debug_overfit_5class.py
import os, csv, random, pathlib, numpy as np
from typing import List, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "tiny-imagenet-200"
IMG_SIZE = (224, 224)                 # Tiny-ImageNet is 64x64; we upscale for MobileNetV2
BATCH = 32
SEED = 42

# keep runs quick and deterministic
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

# ---------- helpers ----------
def read_wnids(d: pathlib.Path) -> List[str]:
    return (d / "wnids.txt").read_text().strip().splitlines()

def take_first_n(classes: List[str], n: int = 5) -> List[str]:
    return classes[:n]

def parse_val_file(val_dir: pathlib.Path, keep_wnids: List[str]) -> List[Tuple[str,int]]:
    wnid_to_idx = {w:i for i,w in enumerate(keep_wnids)}
    out = []
    with (val_dir/"val_annotations.txt").open("r", newline="") as f:
        r = csv.reader(f, delimiter="\t")
        for row in r:
            fname, wnid = row[0], row[1]
            if wnid in wnid_to_idx:
                out.append((str(val_dir/"images"/fname), wnid_to_idx[wnid]))
    return out

def decode_img(path, label, train: bool):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    if train:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.15)
        img = tf.image.random_contrast(img, 0.85, 1.15)
    return img, label

def make_datasets() -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str]]:
    # choose 5 classes in a fixed order
    all_wnids = read_wnids(DATA_DIR)
    keep = take_first_n(all_wnids, 5)
    print("\nWNIDs (order -> class_id):")
    for i, w in enumerate(keep):
        print(f"  {i} -> {w}")

    # train: directory structure train/<wnid>/images/*.JPEG
    train_files, train_labels = [], []
    for idx, wnid in enumerate(keep):
        img_dir = DATA_DIR / "train" / wnid / "images"
        files = sorted([str(p) for p in img_dir.glob("*.JPEG")])
        train_files += files
        train_labels += [idx] * len(files)

    # val: flat images + mapping file
    val_pairs = parse_val_file(DATA_DIR / "val", keep)
    val_files = [p for p, _ in val_pairs]
    val_labels = [y for _, y in val_pairs]

    print(f"\nTrain samples: {len(train_files)}  Val samples: {len(val_files)}")
    # quick histogram to make sure labels are spread
    t_counts = {i: int(np.sum(np.array(train_labels)==i)) for i in range(len(keep))}
    v_counts = {i: int(np.sum(np.array(val_labels)==i)) for i in range(len(keep))}
    print("Class counts (train):", t_counts)
    print("Class counts (val):  ", v_counts)

    train_ds = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    train_ds = (train_ds
                .shuffle(len(train_files), seed=SEED, reshuffle_each_iteration=True)
                .map(lambda p,y: decode_img(p,y,train=True), num_parallel_calls=tf.data.AUTOTUNE)
                .batch(BATCH)
                .prefetch(tf.data.AUTOTUNE))

    val_ds = tf.data.Dataset.from_tensor_slices((val_files, val_labels))
    val_ds = (val_ds
              .map(lambda p,y: decode_img(p,y,train=False), num_parallel_calls=tf.data.AUTOTUNE)
              .batch(BATCH)
              .prefetch(tf.data.AUTOTUNE))

    return train_ds, val_ds, keep

def count_trainables(model: keras.Model) -> int:
    return int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))

# ---------- model ----------
def build_model(num_classes: int) -> tuple[keras.Model, keras.Model]:
    base = keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",            # for this debug, imagenet weights help
        pooling=None
    )
    inp = layers.Input(shape=(*IMG_SIZE, 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inp)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    model = keras.Model(inp, out)
    return model, base

# ---------- train ----------
def main():
    train_ds, val_ds, keep = make_datasets()
    num_classes = len(keep)
    model, base = build_model(num_classes)

    # WARM-UP: freeze all convs, keep BN updating stats + train the head
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
        else:
            layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"[diag] trainables (warm-up): {count_trainables(model)} params")
    model.fit(train_ds, validation_data=val_ds, epochs=5, verbose=2)

    # FINE-TUNE: unfreeze all non-BN layers, freeze BN
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True   # <-- critical: actually unfreeze convs

    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"[diag] trainables (fine-tune): {count_trainables(model)} params (should be in the millions)")

    hist = model.fit(train_ds, validation_data=val_ds, epochs=20, verbose=2)

    # quick end-of-run diagnostics on a tiny batch
    def softmax_diag(ds, training_flag):
        for xb, _ in ds.take(1):
            probs = model(xb, training=training_flag).numpy()
            m = probs.mean(axis=0)
            counts = np.bincount(probs.argmax(axis=1), minlength=num_classes)
            return m, counts
        return None, None

    m_val_f, c_val_f = softmax_diag(val_ds, training_flag=False)
    m_val_t, c_val_t = softmax_diag(val_ds, training_flag=True)
    m_tr_f,  c_tr_f  = softmax_diag(train_ds, training_flag=False)
    m_tr_t,  c_tr_t  = softmax_diag(train_ds, training_flag=True)

    print("\n[diag] VAL (training=False) softmax mean:", np.round(m_val_f, 3))
    print("[diag] VAL (training=False) argmax counts:", c_val_f)
    print("[diag] VAL (training=True)  softmax mean:", np.round(m_val_t, 3))
    print("[diag] VAL (training=True)  argmax counts:", c_val_t)
    print("[diag] TRAIN (training=False) softmax mean:", np.round(m_tr_f, 3))
    print("[diag] TRAIN (training=False) argmax counts:", c_tr_f)
    print("[diag] TRAIN (training=True)  softmax mean:", np.round(m_tr_t, 3))
    print("[diag] TRAIN (training=True)  argmax counts:", c_tr_t)

    # compact summary
    def last(xs, k=5): return xs[-k:] if xs else []
    print("\nEVAL — train: acc=%.3f, val: acc=%.3f" %
          (hist.history["accuracy"][-1], hist.history["val_accuracy"][-1]))
    print("\nLast 5 epochs (acc/val_acc):")
    for a, va in zip(last(hist.history["accuracy"]), last(hist.history["val_accuracy"])):
        print(f"  {a:.3f} / {va:.3f}")

if __name__ == "__main__":
    main()
