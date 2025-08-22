import os, csv, json, pathlib
from typing import Tuple, List

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers  # type: ignore

# ----------------------------
# Config
# ----------------------------
ROOT = pathlib.Path(__file__).resolve().parents[1]      # -> ml-training/
DATA_DIR = ROOT / "data" / "tiny-imagenet-200"
CKPT_DIR = ROOT / "checkpoints" / "tiny_mobilenetv2"
BACKBONE_DIR = ROOT / "checkpoints" / "backbone_tiny_mnv2"
LOG_DIR = ROOT / "checkpoints" / "tblogs"

IMG_SIZE = (224, 224)    # upscale Tiny-ImageNet (64->224) for MobileNetV2
BATCH = 64               # 
EPOCHS_HEAD = 3          # warm-up head
EPOCHS_FT = 20           # fine-tune
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

# Optional mixed precision (helps on GPU)
try:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
except Exception:
    pass

# ----------------------------
# Data
# ----------------------------
def _load_val_annotations(val_dir: pathlib.Path) -> Tuple[List[str], List[int], List[str]]:
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

def _decode(path, label, train=False):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    if train:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_contrast(img, 0.8, 1.2)
    return img, label

def make_datasets():
    # Train (directory-of-directories) — already batched by Keras util
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "train",
        labels="inferred",
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=BATCH,       # <-- batches here
        shuffle=True,
        seed=SEED,
    )
    # normalize + augment; DO NOT .batch() again
    train_ds = (train_ds
        .map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y), num_parallel_calls=AUTOTUNE)
        .map(lambda x, y: _decode_from_tensor(x, y, train=True), num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )

    # Val (flat folder + annotations) — we build from file list, so we batch once here
    val_files, val_labels, wnids = _load_val_annotations(DATA_DIR / "val")
    val_ds = (tf.data.Dataset.from_tensor_slices((val_files, val_labels))
        .map(lambda p, y: _decode(p, y, train=False), num_parallel_calls=AUTOTUNE)
        .batch(BATCH)           # <-- batch only here
        .prefetch(AUTOTUNE)
    )

    return train_ds, val_ds, len(wnids)

def _decode_from_tensor(x, y, train=False):
    # x already sized/normalized by image_dataset_from_directory, just augment lightly
    if train:
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_brightness(x, 0.2)
        x = tf.image.random_contrast(x, 0.8, 1.2)
    return x, y

# ----------------------------
# Model (MobileNetV2 backbone)
# ----------------------------
def build_mobilenet_v2(num_classes: int) -> keras.Model:
    base = keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights=None  # <- since pretraining myself (no ImageNet weights)
    )
    inp = layers.Input(shape=(*IMG_SIZE, 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inp)
    x = base(x, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)  # force float32 on output
    model = keras.Model(inp, out)
    return model, base

# ----------------------------
# Train
# ----------------------------
def main():
    tf.random.set_seed(SEED)
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(BACKBONE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    train_ds, val_ds, num_classes = make_datasets()
    print(f"Classes: {num_classes}")

    model, base = build_mobilenet_v2(num_classes)

    # 1) Head warm-up (freeze backbone) Helps to stabilize training
    base.trainable = False
    model.compile(
        optimizer=keras.optimizers.AdamW(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD)

    # 2) Fine-tune last ~30% of layers
    for i, layer in enumerate(base.layers):
        layer.trainable = (i >= int(len(base.layers) * 0.7))

    # Optional LR schedule for longer runs
    lr_cb = keras.callbacks.ReduceLROnPlateau(
        monitor="val_accuracy", factor=0.5, patience=2, min_lr=1e-5, verbose=1
    )
    ckpt_cb = keras.callbacks.ModelCheckpoint(
        filepath=str(CKPT_DIR / "best.keras"),
        save_best_only=True, monitor="val_accuracy", mode="max"
    )
    es_cb = keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5, restore_best_weights=True
    )
    tb_cb = keras.callbacks.TensorBoard(log_dir=str(LOG_DIR))

    model.compile(
        optimizer=keras.optimizers.AdamW(3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FT,
        callbacks=[lr_cb, ckpt_cb, es_cb, tb_cb],
    )

    # Save full model and backbone
    model.save(CKPT_DIR / "final.keras")
    base.save(BACKBONE_DIR)

    with open(CKPT_DIR / "history.json", "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in hist.history.items()}, f)

    print(f"\nSaved:\n  full -> {CKPT_DIR/'best.keras'} & {CKPT_DIR/'final.keras'}\n  backbone -> {BACKBONE_DIR}")

    # Export a float32 TFLite now (quantize later after Pokémon FT)
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite = converter.convert()
        with open(CKPT_DIR / "final_float32.tflite", "wb") as f:
            f.write(tflite)
        print(f"TFLite exported -> {CKPT_DIR/'final_float32.tflite'}")
    except Exception as e:
        print(f"Skipped TFLite export: {e}")

if __name__ == "__main__":
    main()
