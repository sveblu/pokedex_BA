import os, csv, json, pathlib
from typing import Tuple, List

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# =========================
# Paths (WSL/local only)
# =========================
ROOT = pathlib.Path(__file__).resolve().parents[1]  # -> ml-training/
DATA_DIR = ROOT / "data" / "tiny-imagenet-200"
CKPT_DIR = ROOT / "checkpoints" / "tiny_mobilenetv2"
BACKBONE_DIR = ROOT / "checkpoints" / "backbone_tiny_mnv2"
LOG_DIR = ROOT / "checkpoints" / "tblogs"

CKPT_DIR.mkdir(parents=True, exist_ok=True)
BACKBONE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Config
# =========================
IMG_SIZE    = (224, 224)
BATCH       = 64
EPOCHS_HEAD = 3      # warm-up with backbone frozen
EPOCHS_FT   = 20     # fine-tune
SEED        = 42
AUTOTUNE    = tf.data.AUTOTUNE

# Mixed precision is good on RTX cards. Comment out if you see instability.
try:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
except Exception:
    pass

# =========================
# Data
# =========================
def _load_val_annotations(val_dir: pathlib.Path) -> Tuple[List[str], List[int], List[str]]:
    """Parse Tiny-ImageNet validation mapping and return (files, labels, wnids)."""
    wnids = (DATA_DIR / "wnids.txt").read_text().strip().splitlines()
    wnid_to_id = {w: i for i, w in enumerate(wnids)}
    files, labels = [], []
    with (val_dir / "val_annotations.txt").open("r") as f:
        for fname, wnid, *_ in csv.reader(f, delimiter="\t"):
            p = val_dir / "images" / fname
            if p.exists() and wnid in wnid_to_id:
                files.append(str(p))
                labels.append(wnid_to_id[wnid])
    return files, labels, wnids

def _decode(path, label, train: bool = False):
    """Decode file path -> image (float32 in [0..255]), resize, light aug on train."""
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)  # keep [0..255]; MobileNetV2 preprocess will scale
    if train:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_contrast(img, 0.8, 1.2)
    return img, label

def _augment_from_tensor(x, y):
    # x from image_dataset_from_directory is float32 [0..255]; just augment
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_brightness(x, 0.2)
    x = tf.image.random_contrast(x, 0.8, 1.2)
    return x, y

def make_datasets():
    # Ensure both train and val share the SAME class index order (from wnids.txt)
    _, _, wnids = _load_val_annotations(DATA_DIR / "val")

    # TRAIN
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "train",
        labels="inferred",
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=BATCH,
        shuffle=True,
        seed=SEED,
        class_names=wnids,                # <-- crucial: align to wnids order
    )
    train_ds = train_ds.map(_augment_from_tensor, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.prefetch(AUTOTUNE)

    # VAL (uses val_annotations.txt which we map to wnids)
    val_files, val_labels, _ = _load_val_annotations(DATA_DIR / "val")
    val_ds = (tf.data.Dataset.from_tensor_slices((val_files, val_labels))
              .map(lambda p, y: _decode(p, y, train=False), num_parallel_calls=AUTOTUNE)
              .batch(BATCH)
              .prefetch(AUTOTUNE))

    return train_ds, val_ds, len(wnids)

# =========================
# Model
# =========================
def build_mobilenet_v2(num_classes: int) -> keras.Model:
    base = keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights=None,      # pretrain from scratch on Tiny-ImageNet
        alpha=1.0,
    )
    inp = layers.Input(shape=(*IMG_SIZE, 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inp)  # scales to [-1,1]
    x = base(x, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)  # a bit more regularization helps
    out = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    model = keras.Model(inp, out)
    return model, base

# =========================
# Train
# =========================
def main():
    tf.random.set_seed(SEED)

    train_ds, val_ds, num_classes = make_datasets()
    print(f"Classes: {num_classes}")

    model, base = build_mobilenet_v2(num_classes)

    # Warm-up head (freeze backbone)
    base.trainable = False
    model.compile(
        optimizer=keras.optimizers.AdamW(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD, verbose=2)

    # Fine-tune: unfreeze last 50% of layers
    for i, layer in enumerate(base.layers):
        layer.trainable = (i >= int(len(base.layers) * 0.5))

    # Callbacks
    ckpt_cb = keras.callbacks.ModelCheckpoint(
        filepath=str(CKPT_DIR / "best.keras"),
        save_best_only=True, monitor="val_accuracy", mode="max"
    )
    # make LR react to bad val_loss quickly
    lr_cb = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=1, min_lr=1e-5, verbose=1
    )
    es_cb = keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=10, restore_best_weights=True
    )
    tb_cb = keras.callbacks.TensorBoard(log_dir=str(LOG_DIR))

    # Stable FT setup
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FT,
        callbacks=[ckpt_cb, lr_cb, es_cb, tb_cb],
        verbose=2,
    )

    # Save final artifacts
    model.save(CKPT_DIR / "final.keras")
    (BACKBONE_DIR / "backbone.keras").parent.mkdir(parents=True, exist_ok=True)
    base.save(BACKBONE_DIR / "backbone.keras")

    with open(CKPT_DIR / "history.json", "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in hist.history.items()}, f)

    print("✅ Training finished. Saved:")
    print("  -", CKPT_DIR / "best.keras")
    print("  -", CKPT_DIR / "final.keras")
    print("  -", BACKBONE_DIR / "backbone.keras")

if __name__ == "__main__":
    # Optional: let TF grow GPU memory instead of pre-allocating
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    main()
