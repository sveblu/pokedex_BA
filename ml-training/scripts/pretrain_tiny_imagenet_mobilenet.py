import os, csv, json, pathlib, shutil, zipfile
from typing import Tuple, List

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers  # type: ignore

# =========================================================
# Colab-friendly hybrid paths (local <-> Drive)
# =========================================================
ROOT = pathlib.Path(__file__).resolve().parents[1]  # -> ml-training/

LOCAL_DATA_DIR = ROOT / "data" / "tiny-imagenet-200"
ENV_DATA = os.getenv("DATA_DIR", "").strip()
if ENV_DATA:
    LOCAL_DATA_DIR = pathlib.Path(ENV_DATA)

DRIVE_DATA_DIR = pathlib.Path("/content/drive/MyDrive/datasets/tiny-imagenet-200")
DRIVE_ZIP      = pathlib.Path("/content/drive/MyDrive/datasets/tiny-imagenet-200.zip")

# Local artifacts (fast during the session)
CKPT_DIR         = ROOT / "checkpoints" / "tiny_mobilenetv2"
BACKBONE_DIR     = ROOT / "checkpoints" / "backbone_tiny_mnv2"
BACKBONE_PATH    = BACKBONE_DIR / "backbone.keras"
LOG_DIR_LOCAL    = ROOT / "checkpoints" / "tblogs"

# Drive artifacts (persist across sessions)
DRIVE_CKPT_DIR      = pathlib.Path("/content/drive/MyDrive/pokedex_ckpts/tiny_mobilenetv2")
DRIVE_BACKBONE_PATH = pathlib.Path("/content/drive/MyDrive/pokedex_ckpts/backbone_tiny_mnv2/backbone.keras")
DRIVE_LOG_DIR       = pathlib.Path("/content/drive/MyDrive/pokedex_ckpts/tblogs")
DRIVE_BKP_DIR       = pathlib.Path("/content/drive/MyDrive/pokedex_ckpts/_backup/tiny_mobilenetv2")

def ensure_local_dataset():
    if LOCAL_DATA_DIR.exists():
        print(f"[data] Using local dataset at: {LOCAL_DATA_DIR}")
        return
    if DRIVE_DATA_DIR.exists():
        print(f"[data] Copying dataset from Drive...\n  {DRIVE_DATA_DIR} -> {LOCAL_DATA_DIR}")
        (ROOT / "data").mkdir(parents=True, exist_ok=True)
        shutil.copytree(DRIVE_DATA_DIR, LOCAL_DATA_DIR)
        print("[data] Copy complete.")
        return
    if DRIVE_ZIP.exists():
        print(f"[data] Unzipping dataset from Drive...\n  {DRIVE_ZIP} -> {LOCAL_DATA_DIR.parent}")
        (ROOT / "data").mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(DRIVE_ZIP, "r") as zf:
            zf.extractall(LOCAL_DATA_DIR.parent)
        if not LOCAL_DATA_DIR.exists():
            raise FileNotFoundError(f"Unzipped, but not found at {LOCAL_DATA_DIR}")
        print("[data] Unzip complete.")
        return
    raise FileNotFoundError(
        f"Tiny ImageNet not found.\n"
        f"- Local: {LOCAL_DATA_DIR}\n- Drive folder: {DRIVE_DATA_DIR}\n- Drive zip: {DRIVE_ZIP}\n"
        "Upload it to Drive or set DATA_DIR env var."
    )

ensure_local_dataset()

# =========================================================
# Training config
# =========================================================
DATA_DIR     = LOCAL_DATA_DIR
IMG_SIZE     = (224, 224)   # MobileNetV2 standard
ALPHA        = 1.0          # 0.35..1.0; lower = narrower & faster
BATCH        = 64
EPOCHS_HEAD  = 3            # warmup (backbone frozen)
EPOCHS_FT    = 20           # fine-tune (use BackupAndRestore to resume)
SEED         = 42
AUTOTUNE     = tf.data.AUTOTUNE

# Enable GPU memory growth (avoids upfront allocation hiccups)
for g in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

# Mixed precision (good on T4); comment out if unstable
try:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
except Exception:
    pass

# =========================================================
# Data pipeline
#   Important: DO NOT divide by 255 here because MobileNetV2
#   preprocess_input expects [0..255] floats and scales internally.
# =========================================================
def _load_val_annotations(val_dir: pathlib.Path):
    wnids = (DATA_DIR / "wnids.txt").read_text().strip().splitlines()
    wnid_to_id = {w: i for i, w in enumerate(wnids)}
    files, labels, label_wnids = [], [], []
    with (val_dir / "val_annotations.txt").open("r", newline="") as f:
        r = csv.reader(f, delimiter="\t")
        for row in r:
            fname, wnid = row[0], row[1]
            p = val_dir / "images" / fname
            if p.exists() and wnid in wnid_to_id:
                files.append(str(p)); labels.append(wnid_to_id[wnid]); label_wnids.append(wnid)
    return files, labels, wnids

def _decode(path, label, train=False):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)  # keep [0..255] float
    if train:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_contrast(img, 0.8, 1.2)
    return img, label

def _decode_from_tensor(x, y, train=False):
    # x from image_dataset_from_directory is float32 [0..255]
    if train:
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_brightness(x, 0.2)
        x = tf.image.random_contrast(x, 0.8, 1.2)
    return x, y

def make_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "train",
        labels="inferred",
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=BATCH,
        shuffle=True,
        seed=SEED,
    )
    train_ds = train_ds.map(lambda x, y: _decode_from_tensor(x, y, train=True),
                            num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    val_files, val_labels, wnids = _load_val_annotations(DATA_DIR / "val")
    val_ds = tf.data.Dataset.from_tensor_slices((val_files, val_labels))
    val_ds = val_ds.map(lambda p, y: _decode(p, y, train=False),
                        num_parallel_calls=AUTOTUNE).batch(BATCH).prefetch(AUTOTUNE)
    return train_ds, val_ds, len(wnids)

# =========================================================
# Model
# =========================================================
def build_mobilenet_v2(num_classes: int) -> keras.Model:
    base = keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights=None,           # pretrain yourself
        alpha=ALPHA,
    )
    inp = layers.Input(shape=(*IMG_SIZE, 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inp)  # scales [0..255] -> [-1,1]
    x = base(x, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)  # float32 output
    model = keras.Model(inp, out)
    return model, base

# =========================================================
# Helpers
# =========================================================
def _mirror(src: pathlib.Path, dst: pathlib.Path, label: str):
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        print(f"[mirror] {label}: {src} -> {dst}")
    except Exception as e:
        print(f"[mirror] Skipped mirroring {label}: {e}")

# =========================================================
# Train
# =========================================================
def main():
    tf.random.set_seed(SEED)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    BACKBONE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR_LOCAL.mkdir(parents=True, exist_ok=True)
    DRIVE_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    DRIVE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    DRIVE_BKP_DIR.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, num_classes = make_datasets()
    print(f"Classes: {num_classes}")

    model, base = build_mobilenet_v2(num_classes)

    # Warm-up few forward passes to initialize kernels (reduces timer warnings)
    for batch, _ in val_ds.take(2):
        _ = model(batch, training=False)

    # 1) Head warm-up (backbone frozen)
    base.trainable = False
    model.compile(optimizer=keras.optimizers.AdamW(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD, verbose=2)

    # 2) Fine-tune last ~30%
    for i, layer in enumerate(base.layers):
        layer.trainable = (i >= int(len(base.layers) * 0.7))

    # Persistent, resumable training
    ckpt_local = keras.callbacks.ModelCheckpoint(
        filepath=str(CKPT_DIR / "best.keras"),
        save_best_only=True, monitor="val_accuracy", mode="max"
    )
    ckpt_drive = keras.callbacks.ModelCheckpoint(
        filepath=str(DRIVE_CKPT_DIR / "best.keras"),
        save_best_only=True, monitor="val_accuracy", mode="max"
    )
    lr_cb   = keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=2, min_lr=1e-5, verbose=1)
    es_cb   = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
    tb_cb   = keras.callbacks.TensorBoard(log_dir=str(DRIVE_LOG_DIR))  # write to Drive so logs persist
    csv_cb  = keras.callbacks.CSVLogger(str(DRIVE_CKPT_DIR / "metrics.csv"), append=True)
    backup_cb = keras.callbacks.BackupAndRestore(backup_dir=str(DRIVE_BKP_DIR))  # auto-resume

    model.compile(optimizer=keras.optimizers.AdamW(3e-4),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    hist = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FT,
                     callbacks=[lr_cb, ckpt_local, ckpt_drive, es_cb, tb_cb, csv_cb, backup_cb],
                     verbose=2)

    # Save artifacts (local)
    model.save(CKPT_DIR / "final.keras")
    base.save(BACKBONE_PATH)  # Keras 3 file format

    with open(CKPT_DIR / "history.json", "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in hist.history.items()}, f)

    print(f"\nSaved:\n  best -> {CKPT_DIR/'best.keras'} (also mirrored on Drive)\n"
          f"  final -> {CKPT_DIR/'final.keras'}\n  backbone -> {BACKBONE_PATH}")

    # Export float32 TFLite (quantize later after Pokémon FT)
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite = converter.convert()
        with open(CKPT_DIR / "final_float32.tflite", "wb") as f:
            f.write(tflite)
        print(f"TFLite exported -> {CKPT_DIR/'final_float32.tflite'}")
    except Exception as e:
        print(f"Skipped TFLite export: {e}")

    # Also mirror local artifacts to Drive (convenience)
    _mirror(CKPT_DIR,               DRIVE_CKPT_DIR,      "checkpoints")
    _mirror(BACKBONE_PATH,          DRIVE_BACKBONE_PATH, "backbone")

if __name__ == "__main__":
    main()
