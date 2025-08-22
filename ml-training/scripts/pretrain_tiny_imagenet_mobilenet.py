import os, csv, json, pathlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ----------------------------
# Paths
# ----------------------------
ROOT = pathlib.Path(__file__).resolve().parents[1]  # -> ml-training/
DATA_DIR = ROOT / "data" / "tiny-imagenet-200"
CKPT_DIR = ROOT / "checkpoints" / "tiny_mobilenetv2"
BACKBONE_DIR = ROOT / "checkpoints" / "backbone_tiny_mnv2"
LOG_DIR = ROOT / "checkpoints" / "tblogs"

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(BACKBONE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ----------------------------
# Config
# ----------------------------
IMG_SIZE = (224, 224)
BATCH = 64
EPOCHS_HEAD = 3
EPOCHS_FT = 20
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

# Mixed precision for GPU
try:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
except Exception:
    pass

# ----------------------------
# Data
# ----------------------------
def _load_val_annotations(val_dir: pathlib.Path):
    wnids = (DATA_DIR / "wnids.txt").read_text().strip().splitlines()
    wnid_to_id = {w: i for i, w in enumerate(wnids)}
    files, labels = [], []
    with (val_dir / "val_annotations.txt").open("r") as f:
        for row in csv.reader(f, delimiter="\t"):
            fname, wnid = row[0], row[1]
            p = val_dir / "images" / fname
            if p.exists() and wnid in wnid_to_id:
                files.append(str(p))
                labels.append(wnid_to_id[wnid])
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
    # Train
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "train",
        labels="inferred",
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=BATCH,
        shuffle=True,
        seed=SEED,
    )
    train_ds = train_ds.prefetch(AUTOTUNE)

    # Val
    val_files, val_labels, wnids = _load_val_annotations(DATA_DIR / "val")
    val_ds = (tf.data.Dataset.from_tensor_slices((val_files, val_labels))
              .map(lambda p, y: _decode(p, y, train=False), num_parallel_calls=AUTOTUNE)
              .batch(BATCH)
              .prefetch(AUTOTUNE))

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
    tf.random.set_seed(SEED)
    train_ds, val_ds, num_classes = make_datasets()
    print(f"Classes: {num_classes}")

    model, base = build_mobilenet_v2(num_classes)

    # Warm-up head
    base.trainable = False
    model.compile(optimizer=keras.optimizers.AdamW(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD)

    # Fine-tune last 30%
    for i, layer in enumerate(base.layers):
        layer.trainable = (i >= int(len(base.layers) * 0.7))

    ckpt_cb = keras.callbacks.ModelCheckpoint(
        filepath=str(CKPT_DIR / "best.keras"),
        save_best_only=True, monitor="val_accuracy", mode="max"
    )
    es_cb = keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5, restore_best_weights=True
    )
    tb_cb = keras.callbacks.TensorBoard(log_dir=str(LOG_DIR))

    model.compile(optimizer=keras.optimizers.AdamW(3e-4),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    hist = model.fit(train_ds, validation_data=val_ds,
                     epochs=EPOCHS_FT, callbacks=[ckpt_cb, es_cb, tb_cb])

    model.save(CKPT_DIR / "final.keras")
    base.save(BACKBONE_DIR / "backbone.keras")

    with open(CKPT_DIR / "history.json", "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in hist.history.items()}, f)

    print("✅ Training finished, models saved.")

if __name__ == "__main__":
    main()
