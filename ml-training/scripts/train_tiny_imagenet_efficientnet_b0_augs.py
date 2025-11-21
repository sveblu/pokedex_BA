#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
import argparse
from pathlib import Path
from typing import Tuple, List

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.efficientnet import preprocess_input  # type: ignore

# -------------------------------------------------------------------
# Runtime configuration
# -------------------------------------------------------------------

tf.config.optimizer.set_jit(False)

for g in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

AUTOTUNE = tf.data.AUTOTUNE


# -------------------------------------------------------------------
# Small helpers
# -------------------------------------------------------------------

def count_trainable_params(model: keras.Model) -> int:
    total = 0
    for v in model.trainable_variables:
        n = 1
        for d in v.shape:
            n *= int(d) if d is not None else 1
        total += int(n)
    return int(total)


# -------------------------------------------------------------------
# Label lookup tables
# -------------------------------------------------------------------

def build_lookup_tables(
    root: Path,
) -> Tuple[tf.lookup.StaticHashTable, tf.lookup.StaticHashTable, List[str]]:
    wnids_path = root / "wnids.txt"
    assert wnids_path.exists(), f"Missing: {wnids_path}"
    wnids = [w.strip() for w in wnids_path.read_text().splitlines() if w.strip()]
    assert wnids, "wnids.txt is empty"

    wnid_keys = tf.constant(wnids, tf.string)
    wnid_vals = tf.range(len(wnids), dtype=tf.int32)
    wnid_init = tf.lookup.KeyValueTensorInitializer(wnid_keys, wnid_vals)
    wnid_to_index = tf.lookup.StaticHashTable(wnid_init, default_value=-1)

    ann_path = root / "val" / "val_annotations.txt"
    assert ann_path.exists(), f"Missing: {ann_path}"
    lines = [ln.strip() for ln in ann_path.read_text().splitlines() if ln.strip()]

    val_files: List[str] = []
    val_inds: List[int] = []
    for ln in lines:
        fname, wnid, *_ = ln.split("\t")
        idx = wnids.index(wnid)
        val_files.append(fname)
        val_inds.append(idx)

    vf_keys = tf.constant(val_files, tf.string)
    vf_vals = tf.constant(val_inds, tf.int32)
    vf_init = tf.lookup.KeyValueTensorInitializer(vf_keys, vf_vals)
    valfile_to_index = tf.lookup.StaticHashTable(vf_init, default_value=-1)

    return wnid_to_index, valfile_to_index, wnids


# -------------------------------------------------------------------
# Decode / resize with EfficientNet normalization
# -------------------------------------------------------------------

def _decode_and_resize(img_bytes: tf.Tensor, image_size: tuple[int, int]) -> tf.Tensor:
    img = tf.image.decode_jpeg(img_bytes, channels=3)     # uint8 0–255
    img = tf.image.resize(img, image_size, antialias=True)
    img = tf.cast(img, tf.float32)                        # float32 0–255
    img = preprocess_input(img)                           # EfficientNet preprocessing
    return img


# -------------------------------------------------------------------
# Dataset construction with Keras augmentations
# -------------------------------------------------------------------

def make_datasets(
    root_dir: str,
    batch_size: int = 32,
    image_size: tuple[int, int] = (224, 224),
    cache_to_disk: bool = False,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, int, List[str]]:
    root = Path(root_dir)
    train_dir = root / "train"
    val_img_dir = root / "val" / "images"

    assert train_dir.exists(), f"Missing: {train_dir}"
    assert val_img_dir.exists(), f"Missing: {val_img_dir}"

    wnid_to_index, valfile_to_index, wnids = build_lookup_tables(root)
    num_classes = len(wnids)

    # Same augmentations as MobileNetV2 version
    aug = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.125, fill_mode="reflect"),  # ≈ ±45°
            keras.layers.RandomZoom(
                height_factor=(-0.3, 0.0),
                width_factor=(-0.3, 0.0),
                fill_mode="reflect",
            ),
        ],
        name="aug",
    )

    def _load_train_raw(path: tf.Tensor):
        img_bytes = tf.io.read_file(path)
        img = _decode_and_resize(img_bytes, image_size)
        parts = tf.strings.split(path, os.sep)
        wnid = parts[-3]
        label = wnid_to_index.lookup(wnid)
        return img, label

    def _load_val(path: tf.Tensor):
        img_bytes = tf.io.read_file(path)
        img = _decode_and_resize(img_bytes, image_size)
        fname = tf.strings.split(path, os.sep)[-1]
        label = valfile_to_index.lookup(fname)
        return img, label

    train_files = tf.data.Dataset.list_files(
        str(train_dir / "*" / "images" / "*.JPEG"),
        shuffle=True,
    )
    val_files = tf.data.Dataset.list_files(
        str(val_img_dir / "*.JPEG"),
        shuffle=False,
    )

    train_raw = train_files.map(_load_train_raw, num_parallel_calls=AUTOTUNE)
    val_raw = val_files.map(_load_val, num_parallel_calls=AUTOTUNE)

    if cache_to_disk:
        cache_root = Path.home() / ".tfdata_cache" / "tiny-imagenet-200-efficientnet-aug"
        cache_root.mkdir(parents=True, exist_ok=True)
        train_raw = train_raw.cache(str(cache_root / "train.cache"))
        val_raw = val_raw.cache(str(cache_root / "val.cache"))

    train_ds = (
        train_raw
        .map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
        .shuffle(2000)
        .batch(batch_size, drop_remainder=True)
        .prefetch(2)
    )

    val_ds = (
        val_raw
        .batch(batch_size, drop_remainder=False)
        .prefetch(2)
    )

    return train_ds, val_ds, num_classes, wnids


# -------------------------------------------------------------------
# Model
# -------------------------------------------------------------------

def build_model(num_classes: int) -> tuple[keras.Model, keras.Model]:
    base = keras.applications.EfficientNetB0(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    inp = keras.Input(shape=(224, 224, 3))
    x = base(inp, training=False)       # BN in inference mode during warm-up
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    out = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inp, out, name="efficientnetb0_tiny_imagenet_aug")
    return model, base


def unfreeze_tail(base: keras.Model, num_unfrozen: int) -> None:
    for layer in base.layers:
        layer.trainable = False
    for layer in base.layers[-num_unfrozen:]:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True


def print_mapping_summary(wnids: List[str], limit: int = 10) -> None:
    print("\nWNIDs (order -> class_id):")
    for i, w in enumerate(wnids[:limit]):
        print(f"  {i:<2d} -> {w}")
    if len(wnids) > limit:
        print("  ...")
    print(f"\nClasses detected: {len(wnids)}\n")


# -------------------------------------------------------------------
# Training
# -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="EfficientNetB0 on Tiny-ImageNet-200 with Keras augmentations",
    )
    ap.add_argument("--data_root", required=True, help="tiny-imagenet-200 root")
    ap.add_argument("--run_name", default="tinyimagenet_efficientnet_b0_aug")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--image_size", type=int, nargs=2, default=(224, 224))

    ap.add_argument("--epochs_warmup", type=int, default=5)
    ap.add_argument("--epochs_finetune", type=int, default=30)
    ap.add_argument("--unfreeze_last", type=int, default=120)
    ap.add_argument("--lr_warmup", type=float, default=3e-4)
    ap.add_argument("--lr_finetune", type=float, default=1e-4)

    ap.add_argument("--no_cache", action="store_true",
                    help="Disable on-disk caching")

    ap.add_argument(
        "--monitor",
        type=str,
        default="val_loss",
        choices=["val_loss", "val_accuracy"],
    )
    ap.add_argument("--reduce_lr_patience", type=int, default=3)
    ap.add_argument("--reduce_lr_factor", type=float, default=0.5)
    ap.add_argument("--min_lr", type=float, default=1e-6)

    args = ap.parse_args()
    image_size = tuple(args.image_size)
    cache_to_disk = not args.no_cache

    run_dir = Path("runs") / args.run_name
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, num_classes, wnids = make_datasets(
        root_dir=args.data_root,
        batch_size=args.batch_size,
        image_size=image_size,
        cache_to_disk=cache_to_disk,
    )

    print_mapping_summary(wnids)

    model, base = build_model(num_classes)

    # warm-up
    base.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(args.lr_warmup),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    train_steps = tf.data.experimental.cardinality(train_ds).numpy()
    val_steps = tf.data.experimental.cardinality(val_ds).numpy()

    print(f"[warm-up] trainable params: {count_trainable_params(model):,}")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_warmup,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        verbose=2,
        callbacks=[
            keras.callbacks.CSVLogger(str(run_dir / "warmup_history.csv")),
        ],
    )

    # fine-tune
    unfreeze_tail(base, args.unfreeze_last)
    model.compile(
        optimizer=keras.optimizers.Adam(args.lr_finetune),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    monitor_mode = "max" if args.monitor == "val_accuracy" else "min"

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_dir / "best.keras"),
            monitor=args.monitor,
            mode=monitor_mode,
            save_best_only=True,
            save_weights_only=False,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=args.monitor,
            factor=args.reduce_lr_factor,
            patience=args.reduce_lr_patience,
            min_lr=args.min_lr,
            verbose=1,
            mode=monitor_mode,
        ),
        keras.callbacks.CSVLogger(str(run_dir / "finetune_history.csv")),
    ]

    print(f"[fine-tune] unfreezing last {args.unfreeze_last} backbone layers (BN frozen)")
    print(f"[diag] trainables (fine-tune): {count_trainable_params(model):,}")

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_finetune,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=2,
    )

    final_path = ckpt_dir / "final.keras"
    model.save(final_path)

    eval_train = model.evaluate(train_ds, verbose=0)
    eval_val = model.evaluate(val_ds, verbose=0)
    print(f"\nEVAL — train: acc={eval_train[1]:.3f}, val: acc={eval_val[1]:.3f}\n")
    print(
        "Saved to:\n"
        f"  - {ckpt_dir / 'best.keras'}\n"
        f"  - {final_path}\n"
        f"  - {run_dir / 'warmup_history.csv'}\n"
        f"  - {run_dir / 'finetune_history.csv'}\n"
    )


if __name__ == "__main__":
    main()
