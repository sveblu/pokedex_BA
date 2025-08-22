import os, csv, pathlib, random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --------- config ---------
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "tiny-imagenet-200"
IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS_FROZEN = 3
EPOCHS_FT = 20
SEED = 42
CLASSES = 5
TRAIN_PER_CLASS = 200   # from train/
VAL_PER_CLASS   = 50    # Tiny-ImageNet val has 50 per class
AUTOTUNE = tf.data.AUTOTUNE

random.seed(SEED)
tf.random.set_seed(SEED)

# Disable XLA JIT to avoid any eval-mode quirks
try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

# Keep everything in float32; no mixed precision for this debug
# (We’ll re-enable later once val behaves.)
# tf.keras.mixed_precision.set_global_policy("float32")

def read_wnids():
    wnids = (DATA_DIR / "wnids.txt").read_text().strip().splitlines()
    return wnids[:CLASSES]  # first 5 consistently

def gather_train_files(wnids):
    paths, labels = [], []
    for cid, wnid in enumerate(wnids):
        img_dir = DATA_DIR / "train" / wnid / "images"
        imgs = sorted([*img_dir.glob("*.JPEG"), *img_dir.glob("*.jpg")])
        imgs = imgs[:TRAIN_PER_CLASS]
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
        take = per_class[cid][:VAL_PER_CLASS]  # Tiny-ImageNet ~= 50/class
        paths += take
        labels += [cid] * len(take)
    return paths, labels

def decode(path, label, train=False):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    # Simple, identical normalization for train & val:
    img = tf.cast(img, tf.float32) / 255.0
    if train:
        # keep augment OFF for now to reduce variables
        pass
    return img, label

def build_model(num_classes: int):
    inp = layers.Input(shape=(*IMG_SIZE, 3), dtype=tf.float32)
    x = layers.Rescaling(1.0, offset=0.0)(inp)  # no-op (kept for clarity)
    # MobileNetV2 WITHOUT special preprocess; we already scaled to [0,1]
    base = keras.applications.MobileNetV2(
        include_top=False, input_shape=(*IMG_SIZE,3), weights=None
    )
    # Do NOT force training=True/False here; Keras will handle BN correctly
    x = base(x, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    # No dropout for overfit
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inp, out)
    return model, base

def main():
    wnids = read_wnids()
    print("WNIDs (order -> class_id):")
    for i, w in enumerate(wnids):
        print(f"  {i} -> {w}")

    tr_paths, tr_labels = gather_train_files(wnids)
    va_paths, va_labels = gather_val_files(wnids)

    print(f"\nTrain samples: {len(tr_paths)}  Val samples: {len(va_paths)}")
    print("Class counts (train):",
          {i: tr_labels.count(i) for i in range(len(wnids))})
    print("Class counts (val):  ",
          {i: va_labels.count(i) for i in range(len(wnids))})

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

    # Warm-up a bit (head only)
    base.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FROZEN, verbose=2)

    # Overfit: unfreeze EVERYTHING and train longer
    base.trainable = True
    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    hist = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FT, verbose=2)

    # Evaluate explicitly on TRAIN and VAL to compare
    train_score = model.evaluate(train_ds, verbose=0)
    val_score = model.evaluate(val_ds, verbose=0)
    print(f"\nEVAL — train: acc={train_score[1]:.3f}, val: acc={val_score[1]:.3f}")

    print("\nLast 5 epochs (acc/val_acc):")
    for a, va in zip(hist.history["accuracy"][-5:], hist.history["val_accuracy"][-5:]):
        print(f"  {a:.3f} / {va:.3f}")

if __name__ == "__main__":
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    main()
