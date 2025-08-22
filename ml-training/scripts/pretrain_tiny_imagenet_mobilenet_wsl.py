import os, csv, json, pathlib, random
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
OVERFIT_TEST = True   # True = 5-class sanity run; False = full 200 classes

random.seed(SEED)
tf.random.set_seed(SEED)

# Optional mixed precision (good on RTX)
try:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
except Exception:
    pass

for p in (CKPT_DIR, BACKBONE_DIR, LOG_DIR):
    p.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Data
# ----------------------------
def _load_val_annotations(val_dir: pathlib.Path, wnid_to_id):
    files, labels = [], []
    with (val_dir / "val_annotations.txt").open("r") as f:
        r = csv.reader(f, delimiter="\t")
        for row in r:
            fname, wnid = row[0], row[1]
            p = val_dir / "images" / fname
            if p.exists() and wnid in wnid_to_id:
                files.append(str(p))
                labels.append(wnid_to_id[wnid])
    return files, labels

def _decode(path, label, train=False):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)        # [0..255]; preprocess_input will scale
    if train:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_contrast(img, 0.8, 1.2)
    return img, label

def make_datasets():
    wnids = (DATA_DIR / "wnids.txt").read_text().splitlines()
    if OVERFIT_TEST:
        wnids = wnids[:5]  # tiny subset

    wnid_to_id = {w: i for i, w in enumerate(wnids)}

    # TRAIN — enforce class order via class_names
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "train",
        labels="inferred",
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=BATCH,
        shuffle=True,
        seed=SEED,
        class_names=wnids,          # <- force exact order (matches val)
    ).prefetch(AUTOTUNE)

    # VAL — from annotations mapped to wnids order
    val_files, val_labels = _load_val_annotations(DATA_DIR / "val", wnid_to_id)
    val_ds = (tf.data.Dataset.from_tensor_slices((val_files, val_labels))
              .map(lambda p, y: _decode(p, y, train=False), num_parallel_calls=AUTOTUNE)
              .batch(BATCH)
              .prefetch(AUTOTUNE))

    # Sanity prints
    print("\n=== Sanity: class order (first 10) ===")
    print(wnids[:10])
    print("=====================================\n")

    print("=== Sanity: val mappings (8 samples) ===")
    for f, l in list(zip(val_files, val_labels))[:8]:
        print(os.path.basename(f), "-> idx=", l)
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
    x = keras.applications.mobilenet_v2.preprocess_input(inp)  # scales to [-1,1]
    # IMPORTANT: don't force training=True here; let Keras control BN behavior
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    model = keras.Model(inp, out)
    return model, base

# ----------------------------
# Train
# ----------------------------
def main():
    train_ds, val_ds, num_classes = make_datasets()
    print(f"Classes: {num_classes} (OVERFIT_TEST={OVERFIT_TEST})")

    model, base = build_mobilenet_v2(num_classes)

    # 1) Warmup (freeze backbone)
    base.trainable = False
    model.compile(
        optimizer=keras.optimizers.AdamW(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD, verbose=2)

    # 2) Fine-tune: unfreeze last 70% of layers
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
        verbose=2,
    )

    model.save(CKPT_DIR / "final.keras")
    (BACKBONE_DIR / "backbone.keras").parent.mkdir(parents=True, exist_ok=True)
    base.save(BACKBONE_DIR / "backbone.keras")

    with open(CKPT_DIR / "history.json", "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in hist.history.items()}, f)

    print("\n✅ Training finished. Saved:")
    print("  -", CKPT_DIR / "best.keras")
    print("  -", CKPT_DIR / "final.keras")
    print("  -", BACKBONE_DIR / "backbone.keras")

if __name__ == "__main__":
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    main()
