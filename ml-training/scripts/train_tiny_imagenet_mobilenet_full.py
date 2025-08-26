
"""
Tiny-ImageNet (200 classes) training with MobileNetV2 (ImageNet-pretrained).

Key points:
- Loads the official Tiny-ImageNet directory structure.
- Uses `mobilenet_v2.preprocess_input` INSIDE the model. No external /255 scaling.
- Warms up by training only the classification head, then fine-tunes upper backbone blocks.
- Adds light data augmentation, ReduceLROnPlateau, EarlyStopping, and checkpoints.
- Avoids training BatchNorm layers during fine-tuning.

Expected dataset layout (unzipped Tiny-ImageNet-200):
  TINY_IMAGENET_ROOT/
    train/<WNID>/images/*.JPEG
    val/images/*.JPEG
    val/val_annotations.txt      (filename -> wnid mapping)
    wnids.txt, words.txt, ...

Set `TINY_IMAGENET_ROOT` below if required.
"""

import os
import csv
from pathlib import Path
from typing import List, Tuple, Dict

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import mobilenet_v2


# -----------------------------
# Configuration
# -----------------------------

# Root of the Tiny-ImageNet dataset. Adjust if needed.
TINY_IMAGENET_ROOT = os.environ.get(
    "TINY_IMAGENET_ROOT", str(Path.home() / "data" / "tiny-imagenet-200")
)

# Training knobs
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 1337

# Warm-up (head only), then fine-tune
WARMUP_EPOCHS = 5
FINETUNE_EPOCHS = 20

# Fine-tuning: unfreeze the top N layers of the backbone (BatchNorm stays frozen)
UNFREEZE_TOP_N_LAYERS = 60

# Checkpoint directory
CKPT_DIR = Path("checkpoints") / "tiny_mobilenetv2_full"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Utilities
# -----------------------------

def list_wnids_train(train_dir: Path) -> List[str]:
    """
    Discovers WNIDs from train/ subfolders and returns a deterministic class list.
    The order is used as the single source of truth for class indexing.
    """
    wnids = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
    if len(wnids) != 200:
        print(f"[warn] Expected 200 WNIDs, found {len(wnids)}.")
    return wnids


def parse_val_annotations(val_dir: Path) -> List[Tuple[str, str]]:
    """
    Parses val/val_annotations.txt into a list of (filename, wnid).
    """
    annot_path = val_dir / "val_annotations.txt"
    pairs: List[Tuple[str, str]] = []
    with open(annot_path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            # Format per line: <filename>\t<wnid>\t<x>\t<y>\t<w>\t<h>
            filename, wnid = row[0], row[1]
            pairs.append((filename, wnid))
    return pairs


def make_train_dataset(train_dir: Path, class_names: List[str]) -> tf.data.Dataset:
    """
    Builds the training dataset from train/ subfolders using image_dataset_from_directory.
    Uses the provided class_names to fix label ordering across train/val.
    """
    ds = tf.keras.utils.image_dataset_from_directory(
        directory=str(train_dir),
        labels="inferred",
        label_mode="int",
        class_names=class_names,  # critical to align labels with val set
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=True,
        seed=SEED,
    )
    # Cache + prefetch for performance
    return ds.cache().prefetch(tf.data.AUTOTUNE)


def make_val_dataset(val_dir: Path, class_names: List[str]) -> tf.data.Dataset:
    """
    Builds the validation dataset directly from val/images plus val_annotations.txt.
    Ensures labels align with the same class_names ordering as train.
    """
    pairs = parse_val_annotations(val_dir)
    wnid_to_index: Dict[str, int] = {wnid: idx for idx, wnid in enumerate(class_names)}

    filepaths = []
    labels = []
    images_dir = val_dir / "images"
    for filename, wnid in pairs:
        fp = str(images_dir / filename)
        if wnid not in wnid_to_index:
            # Unseen wnid, skip defensively (should not happen in official set)
            continue
        filepaths.append(fp)
        labels.append(wnid_to_index[wnid])

    path_ds = tf.data.Dataset.from_tensor_slices(filepaths)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int32))

    def load_and_resize(path: tf.Tensor) -> tf.Tensor:
        img = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMAGE_SIZE, method=tf.image.ResizeMethod.BILINEAR)
        # Keep dtype uint8 → the model's preprocess_input will handle casting/scaling.
        img = tf.cast(img, tf.uint8)
        return img

    img_ds = path_ds.map(load_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
    ds = tf.data.Dataset.zip((img_ds, label_ds))

    # Batch without shuffling (validation)
    ds = ds.batch(BATCH_SIZE)
    return ds.cache().prefetch(tf.data.AUTOTUNE)


def build_model(num_classes: int) -> keras.Model:
    """
    Constructs MobileNetV2 classifier with augmentations and proper preprocessing.
    Preprocessing is applied inside the model to avoid double normalization elsewhere.
    """
    inputs = keras.Input(shape=(*IMAGE_SIZE, 3), dtype=tf.uint8, name="input_uint8")

    # Data augmentation (kept light). These layers will cast to float internally.
    aug = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="augment",
    )

    # Preprocessing required by MobileNetV2 (expects float in [0..255], maps to [-1, 1])
    x = aug(inputs)
    x = layers.Lambda(mobilenet_v2.preprocess_input, name="preprocess")(x)

    # Backbone
    backbone = mobilenet_v2.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
        pooling=None,
    )
    backbone.trainable = False  # frozen during warm-up

    x = backbone(x, training=False)  # BN stays in inference mode while frozen
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.2, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    model = keras.Model(inputs, outputs, name="tiny_imagenet_mnv2")
    return model


def count_trainable_params(model: keras.Model) -> int:
    """
    Returns the count of trainable parameters.
    """
    return int(
        sum(tf.reduce_prod(tf.shape(v)).numpy() for v in model.trainable_variables)
    )


def set_backbone_trainable(backbone: keras.Model, unfreeze_top_n: int) -> None:
    """
    Unfreezes the top N layers of the backbone while keeping BatchNorm layers frozen.
    This pattern stabilizes fine-tuning for MobileNet-like architectures.
    """
    # Unfreeze entire backbone first
    backbone.trainable = True

    # Freeze all layers by default
    for layer in backbone.layers:
        layer.trainable = False

    # Unfreeze the last `unfreeze_top_n` layers except BatchNorm
    layers_to_unfreeze = backbone.layers[-unfreeze_top_n:] if unfreeze_top_n > 0 else []
    for layer in layers_to_unfreeze:
        # Skip BatchNorm layers
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True


def main():
    tf.keras.utils.set_random_seed(SEED)

    root = Path(TINY_IMAGENET_ROOT)
    train_dir = root / "train"
    val_dir = root / "val"

    assert train_dir.exists(), f"Missing: {train_dir}"
    assert (val_dir / "images").exists(), f"Missing: {val_dir/'images'}"
    assert (val_dir / "val_annotations.txt").exists(), f"Missing: {val_dir/'val_annotations.txt'}"

    # Discover class list from train/ and reuse identically for val/
    class_names = list_wnids_train(train_dir)
    num_classes = len(class_names)

    print("\nWNIDs (order -> class_id):")
    for i, wnid in enumerate(class_names[:10]):
        print(f"  {i:>3} -> {wnid}")
    if num_classes > 10:
        print("  ...")
    print(f"\nClasses detected: {num_classes}\n")

    # Build datasets
    train_ds = make_train_dataset(train_dir, class_names)
    val_ds = make_val_dataset(val_dir, class_names)

    # Compute simple dataset cardinalities (for logging only)
    train_count = sum(int(b.shape[0]) for b, _ in train_ds.unbatch().batch(1).take(10))  # sample
    # Full counts (exact) if needed:
    # train_count = sum(1 for _ in train_ds.unbatch())
    # val_count = sum(1 for _ in val_ds.unbatch())

    # Build model
    model = build_model(num_classes)

    # -----------------------------
    # Warm-up: train head only
    # -----------------------------
    head_opt = keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(
        optimizer=head_opt,
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    print(f"[warm-up] trainable params: {count_trainable_params(model):,}")
    warmup_callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(CKPT_DIR / "best_warmup.keras"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
        ),
        keras.callbacks.CSVLogger(str(CKPT_DIR / "warmup_log.csv"), append=False),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=WARMUP_EPOCHS,
        callbacks=warmup_callbacks,
        verbose=2,
    )

    # -----------------------------
    # Fine-tune: unfreeze top blocks
    # -----------------------------
    # Locate backbone submodel and unfreeze top layers (keep BN frozen)
    backbone = None
    for layer in model.layers:
        if isinstance(layer, keras.Model) and layer.name.startswith("mobilenetv2"):
            backbone = layer
            break
    assert backbone is not None, "Backbone not found in model graph."

    set_backbone_trainable(backbone, UNFREEZE_TOP_N_LAYERS)

    # Recompile with a lower LR and training=True for the backbone
    ft_opt = keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=ft_opt,
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    print(f"[fine-tune] trainable params: {count_trainable_params(model):,}")

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(CKPT_DIR / "best.keras"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(CKPT_DIR / "final.keras"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=False,
            save_weights_only=False,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=5e-6, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.CSVLogger(str(CKPT_DIR / "finetune_log.csv"), append=False),
        keras.callbacks.TerminateOnNaN(),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINETUNE_EPOCHS,
        callbacks=callbacks,
        verbose=2,
    )

    # -----------------------------
    # Save artifacts
    # -----------------------------
    model.save(str(CKPT_DIR / "final.keras"))
    print("\n✅ Training complete. Saved:")
    print(f"  - {CKPT_DIR / 'best.keras'}")
    print(f"  - {CKPT_DIR / 'final.keras'}")
    print(f"  - Logs: {CKPT_DIR / 'warmup_log.csv'} , {CKPT_DIR / 'finetune_log.csv'}")


if __name__ == "__main__":
    # Environment tweak: ensure TF does not pre-allocate all GPU memory.
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    main()
