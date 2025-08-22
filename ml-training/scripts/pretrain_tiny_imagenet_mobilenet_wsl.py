import os, csv, pathlib, random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --------- config ---------
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "tiny-imagenet-200"
IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS = 25
SEED = 42
CLASSES = 5
TRAIN_PER_CLASS = 200   # from train/
VAL_PER_CLASS   = 100   # from val/
AUTOTUNE = tf.data.AUTOTUNE

random.seed(SEED)
tf.random.set_seed(SEED)

# IMPORTANT: keep everything in float32 for this debug
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
    # Map val images via val_annotations.txt to our 5 wnids
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
        take = per_class[cid][:VAL_PER_CLASS]
        paths += take
        labels += [cid] * len(take)
    return paths, labels

def decode(path, label, train=False):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)  # [0..255]
    if train:
        # NO augmentation in this debug
        pass
    return img, label

def mobilenet_head(num_classes):
    inp = layers.Input(shape=(*IMG_SIZE, 3), dtype=tf.float32)
    x = keras.applications.mobilenet_v2.preprocess_input(inp)  # scales [-1,1]
    base = keras.applications.MobileNetV2(include_top=False, input_shape=(*IMG_SIZE,3), weights=None)
    # DO NOT force training=True here; let Keras control BN
    x = base(x)
    x = layers.GlobalAveragePooling2D()(x)
    # NO dropout for overfit
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

    model, base = mobilenet_head(len(wnids))

    # Freeze base for a couple epochs then unfreeze all for overfit
    base.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=3, verbose=2)

    base.trainable = True  # unfreeze entire backbone to force overfit
    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    hist = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)

    # Print last few metrics to see trend quickly
    print("\nLast 5 epochs (acc/val_acc):")
    for a, va in zip(hist.history["accuracy"][-5:], hist.history["val_accuracy"][-5:]):
        print(f"  {a:.3f} / {va:.3f}")

if __name__ == "__main__":
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    main()
