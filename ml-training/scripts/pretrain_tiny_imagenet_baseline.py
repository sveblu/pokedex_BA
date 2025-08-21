import os
import csv
import json
import pathlib
from typing import Tuple, List

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore

# ----------------------------
# Config
# ----------------------------
ROOT = pathlib.Path(__file__).resolve().parents[1]  # -> ml-training/
DATA_DIR = ROOT / "data" / "tiny-imagenet-200"
CKPT_DIR = ROOT / "checkpoints" / "tiny_baseline"
IMG_SIZE = (64, 64)          # Tiny ImageNet images are 64x64
BATCH = 64
EPOCHS = 3                   # set to 1 for smoke test
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

# ----------------------------
# Data
# ----------------------------
def _load_val_annotations(val_dir: pathlib.Path) -> Tuple[List[str], List[int], List[str]]:
    """Read val_annotations.txt and produce (filepaths, int_labels, wnids)."""
    wnids = (DATA_DIR / "wnids.txt").read_text().strip().splitlines()
    wnid_to_id = {w: i for i, w in enumerate(wnids)}

    files, labels, label_wnids = [], [], []
    with (val_dir / "val_annotations.txt").open("r", newline="") as f:
        r = csv.reader(f, delimiter="\t")
        for row in r:
            fname, wnid = row[0], row[1]
            p = val_dir / "images" / fname
            if p.exists() and wnid in wnid_to_id:
                files.append(str(p))
                labels.append(wnid_to_id[wnid])
                label_wnids.append(wnid)
    return files, labels, wnids

def _decode(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

def make_datasets():
    # Train: directory-of-directories
    train_dir = DATA_DIR / "train"
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=BATCH,
        shuffle=True,
        seed=SEED,
    )

    # Val: flat folder + annotations -> build from file list
    val_dir = DATA_DIR / "val"
    val_files, val_labels, wnids = _load_val_annotations(val_dir)
    val_ds = tf.data.Dataset.from_tensor_slices((val_files, val_labels))
    val_ds = (
        val_ds
        .map(_decode, num_parallel_calls=AUTOTUNE)
        .batch(BATCH)
        .prefetch(AUTOTUNE)
    )

    # Basic augmentation only on train
    def aug(x, y):
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_brightness(x, 0.2)
        x = tf.image.random_contrast(x, 0.8, 1.2)
        return x, y

    train_ds = (
        train_ds
        .map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y), num_parallel_calls=AUTOTUNE)
        .map(aug, num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )

    num_classes = len(wnids)
    return train_ds, val_ds, num_classes

# ----------------------------
# Model
# ----------------------------
def build_simple_cnn(num_classes: int) -> keras.Model:
    return keras.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax"),
    ])

# Swap to MobileNetV2 later if you want a stronger backbone:
def build_mobilenet_head(num_classes: int) -> keras.Model:
    base = keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights=None,          # pretrain yourself here (Tiny ImageNet)
    )
    inp = layers.Input(shape=(*IMG_SIZE, 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inp)
    x = base(x, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inp, out)

# ----------------------------
# Train
# ----------------------------
def main():
    tf.random.set_seed(SEED)
    os.makedirs(CKPT_DIR, exist_ok=True)

    train_ds, val_ds, num_classes = make_datasets()
    print(f"Classes: {num_classes}")

    # Choose one:
    model = build_simple_cnn(num_classes)
    # model = build_mobilenet_head(num_classes)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    ckpt_cb = keras.callbacks.ModelCheckpoint(
        filepath=str(CKPT_DIR / "best.keras"),
        save_best_only=True,
        monitor="val_accuracy",
        mode="max",
    )
    es_cb = keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=3, restore_best_weights=True
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[ckpt_cb, es_cb],
    )

    model.save(CKPT_DIR / "final.keras")

    with open(CKPT_DIR / "history.json", "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)

    print(f"\nSaved: {CKPT_DIR}\\best.keras and final.keras")

if __name__ == "__main__":
    main()
