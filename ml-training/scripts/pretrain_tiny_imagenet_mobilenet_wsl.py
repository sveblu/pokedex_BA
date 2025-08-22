import os, csv, json, pathlib, random
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
for p in (CKPT_DIR, BACKBONE_DIR, LOG_DIR):
    p.mkdir(parents=True, exist_ok=True)

# =========================
# Config
# =========================
IMG_SIZE    = (224, 224)
BATCH       = 64
EPOCHS_HEAD = 3
EPOCHS_FT   = 20
SEED        = 42
AUTOTUNE    = tf.data.AUTOTUNE

# Debug toggles
PRINT_SANITY     = True   # print class order and sample mappings
OVERFIT_TEST     = False  # set True to overfit a tiny subset first
OVERFIT_CLASSES  = 5
OVERFIT_IMAGES_PER_CLASS = 200  # ~1000 images total -> should hit high train acc quickly

random.seed(SEED)
tf.random.set_seed(SEED)

# Mixed precision is fine on RTX; comment out if unstable
try:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
except Exception:
    pass

# =========================
# Helpers: data + sanity
# =========================
def read_wnids() -> List[str]:
    wnids = (DATA_DIR / "wnids.txt").read_text().strip().splitlines()
    if len(wnids) != 200:
        print(f"[warn] wnids.txt has {len(wnids)} entries (expected 200).")
    return wnids

def _load_val_annotations(val_dir: pathlib.Path) -> Tuple[List[str], List[int], List[str]]:
    wnids = read_wnids()
    wnid_to_id = {w: i for i, w in enumerate(wnids)}
    files, labels = [], []
    ann_path = val_dir / "val_annotations.txt"
    with ann_path.open("r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            fname, wnid = row[0], row[1]
            p = val_dir / "images" / fname
            if p.exists() and wnid in wnid_to_id:
                files.append(str(p))
                labels.append(wnid_to_id[wnid])
    return files, labels, wnids

def _decode(path, label, train: bool = False):
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
    # x is float32 [0..255]
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_brightness(x, 0.2)
    x = tf.image.random_contrast(x, 0.8, 1.2)
    return x, y

def make_train_dataset(wnids: List[str], limit_to_subset=False):
    """Create train dataset; if limit_to_subset, only a few classes/images to test overfitting."""
    if not limit_to_subset:
        ds = tf.keras.utils.image_dataset_from_directory(
            DATA_DIR / "train",
            labels="inferred",
            label_mode="int",
            image_size=IMG_SIZE,
            batch_size=BATCH,
            shuffle=True,
            seed=SEED,
            class_names=wnids,  # CRUCIAL: enforce class order == wnids.txt
        )
        return ds.map(_augment_from_tensor, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    # --- tiny overfit subset ---
    # Build file list manually so we can limit images per class
    paths, labels = [], []
    for wnid in wnids[:OVERFIT_CLASSES]:
        class_dir = DATA_DIR / "train" / wnid / "images"
        imgs = sorted([p for p in class_dir.glob("*.JPEG")])
        if not imgs:
            imgs = sorted([p for p in class_dir.glob("*.jpg")])
        imgs = imgs[:OVERFIT_IMAGES_PER_CLASS]
        paths.extend([str(p) for p in imgs])
        labels.extend([wnids.index(wnid)] * len(imgs))
    # shuffle
    perm = list(range(len(paths)))
    random.shuffle(perm)
    paths = [paths[i] for i in perm]
    labels = [labels[i] for i in perm]

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = (ds.map(lambda p, y: _decode(p, y, train=True), num_parallel_calls=AUTOTUNE)
            .batch(BATCH)
            .prefetch(AUTOTUNE))
    return ds

def make_val_dataset(wnids: List[str], limit_to_subset=False):
    val_files, val_labels, _ = _load_val_annotations(DATA_DIR / "val")
    if limit_to_subset:
        # keep only samples whose label < OVERFIT_CLASSES to match tiny train subset
        filt = [(p, y) for (p, y) in zip(val_files, val_labels) if y < OVERFIT_CLASSES]
        val_files, val_labels = zip(*filt) if filt else ([], [])
    ds = tf.data.Dataset.from_tensor_slices((list(val_files), list(val_labels)))
    ds = ds.map(lambda p, y: _decode(p, y, train=False), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH).prefetch(AUTOTUNE)
    return ds

def print_sanity(wnids: List[str], train_ds, val_files, val_labels):
    print("\n=== Sanity: class order (first 10) ===")
    print("wnids.txt[:10]:", wnids[:10])
    try:
        print("train_ds.class_names[:10]:", getattr(train_ds, "class_names")[:10])
    except Exception:
        print("train_ds.class_names not available on this Keras; assumed equal to wnids")

    # show 8 random val samples: (file -> wnid -> label_index)
    print("\n=== Sanity: val mappings (8 samples) ===")
    for i in random.sample(range(min(200, len(val_files))), k=min(8, len(val_files))):
        f = pathlib.Path(val_files[i]); y = val_labels[i]
        # infer wnid from ann mapping (reverse from y)
        wnid = wnids[y]
        print(f"{f.name:>30}  ->  wnid={wnid}  ->  idx={y}")
    print("=======================================\n")

# =========================
# Model
# =========================
def build_mobilenet_v2(num_classes: int) -> keras.Model:
    base = keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights=None,
        alpha=1.0,
    )
    inp = layers.Input(shape=(*IMG_SIZE, 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inp)  # scales [0..255] -> [-1,1]
    x = base(x, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    model = keras.Model(inp, out)
    return model, base

# =========================
# Train
# =========================
def main():
    wnids = read_wnids()

    # choose full train/val or tiny overfit subset
    limit = OVERFIT_TEST
    train_ds = make_train_dataset(wnids, limit_to_subset=limit)
    val_ds   = make_val_dataset(wnids,   limit_to_subset=limit)

    # pull val files/labels for sanity print
    val_files, val_labels, _ = _load_val_annotations(DATA_DIR / "val")

    if PRINT_SANITY:
        print_sanity(wnids, train_ds, val_files, val_labels)

    num_classes = len(wnids)
    print(f"Classes: {num_classes}  (subset={limit})")

    model, base = build_mobilenet_v2(num_classes)

    # Warm-up head (freeze backbone)
    base.trainable = False
    model.compile(
        optimizer=keras.optimizers.AdamW(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD, verbose=2)

    # Fine-tune: unfreeze last 50%
    for i, layer in enumerate(base.layers):
        layer.trainable = (i >= int(len(base.layers) * 0.5))

    ckpt_cb = keras.callbacks.ModelCheckpoint(
        filepath=str(CKPT_DIR / "best.keras"),
        save_best_only=True, monitor="val_accuracy", mode="max"
    )
    # react to rising val_loss early
    lr_cb = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=1, min_lr=1e-5, verbose=1
    )
    es_cb = keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=10, restore_best_weights=True
    )
    tb_cb = keras.callbacks.TensorBoard(log_dir=str(LOG_DIR))

    # Stable FT setup (no label smoothing with sparse integers)
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
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    main()
