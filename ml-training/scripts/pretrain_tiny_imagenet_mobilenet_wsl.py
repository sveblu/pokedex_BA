import os, csv, json, pathlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ----------------------------
# Config
# ----------------------------
ROOT = pathlib.Path(__file__).resolve().parents[1]  # -> ml-training/
DATA_DIR = ROOT / "data" / "tiny-imagenet-200"

CKPT_DIR     = ROOT / "checkpoints" / "tiny_mobilenetv2"
BACKBONE_DIR = ROOT / "checkpoints" / "backbone_tiny_mnv2"
LOG_DIR      = ROOT / "checkpoints" / "tblogs"

IMG_SIZE = (224, 224)
BATCH = 64
EPOCHS_HEAD = 3
EPOCHS_FT = 10
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE
OVERFIT_TEST = True   # <<< set True for debugging, False for full run

# Optional mixed precision (speeds up on GPU)
try:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
except Exception:
    pass

# ----------------------------
# Data
# ----------------------------
def _load_val_annotations(val_dir: pathlib.Path, wnid_to_id):
    files, labels, wnids_used = [], [], []
    with (val_dir / "val_annotations.txt").open("r") as f:
        r = csv.reader(f, delimiter="\t")
        for row in r:
            fname, wnid = row[0], row[1]
            p = val_dir / "images" / fname
            if p.exists() and wnid in wnid_to_id:
                files.append(str(p))
                labels.append(wnid_to_id[wnid])
                wnids_used.append(wnid)
    return files, labels


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
    wnids = (DATA_DIR / "wnids.txt").read_text().splitlines()

    # --- DEBUG: shrink to 5 classes ---
    if OVERFIT_TEST:
        wnids = wnids[:5]

    wnid_to_id = {w: i for i, w in enumerate(wnids)}

    # Train (force class order!)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "train",
        labels="inferred",
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=BATCH,
        shuffle=True,
        seed=SEED,
        classes=wnids,   # <-- enforce class order
    )
    train_ds = (train_ds
        .map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y), num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )

    # Val (manual annotations)
    val_files, val_labels = _load_val_annotations(DATA_DIR / "val", wnid_to_id)
    val_ds = (tf.data.Dataset.from_tensor_slices((val_files, val_labels))
        .map(lambda p, y: _decode(p, y, train=False), num_parallel_calls=AUTOTUNE)
        .batch(BATCH)
        .prefetch(AUTOTUNE)
    )

    # --- Sanity print ---
    print("\n=== Sanity: class order (first 10) ===")
    print(wnids[:10])
    print("=====================================\n")

    print("=== Sanity: val mappings (8 samples) ===")
    for f, l in zip(val_files[:8], val_labels[:8]):
        print(os.path.basename(f), "->", "idx=", l)
    print("=======================================\n")

    return train_ds, val_ds, len(wnids)


# ----------------------------
# Model
# ----------------------------
def build_mobilenet_v2(num_classes: int):
    base = keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights=None
    )
    inp = layers.Input(shape=(*IMG_SIZE, 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inp)
    x = base(x, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    model = keras.Model(inp, out)
    return model, base


# ----------------------------
# Train
# ----------------------------
def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(BACKBONE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    train_ds, val_ds, num_classes = make_datasets()
    print(f"Classes: {num_classes}")

    model, base = build_mobilenet_v2(num_classes)

    # 1) Warmup
    base.trainable = False
    model.compile(
        optimizer=keras.optimizers.AdamW(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD)

    # 2) Fine-tune
    for i, layer in enumerate(base.layers):
        layer.trainable = (i >= int(len(base.layers) * 0.7))

    lr_cb = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1
    )
    ckpt_cb = keras.callbacks.ModelCheckpoint(
        filepath=str(CKPT_DIR / "best.keras"),
        save_best_only=True, monitor="val_loss", mode="min"
    )
    es_cb = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
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

    # Save
    model.save(CKPT_DIR / "final.keras")
    base.save(BACKBONE_DIR / "backbone.keras")

    with open(CKPT_DIR / "history.json", "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in hist.history.items()}, f)

    print(f"\n✅ Training finished, models saved.")


if __name__ == "__main__":
    main()
