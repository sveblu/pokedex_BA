#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MobileNetV2 pretraining on Tiny-ImageNet-200 with *deterministic* augmentations:
- original image
- horizontal flip
- +45° rotation
- -45° rotation
- zoom + blur

Each training file produces 5 samples.
"""

from __future__ import annotations
import os
import math
import argparse
from pathlib import Path
from typing import Tuple, List

import tensorflow as tf
from tensorflow import keras

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
# Augmentation primitives (pure TF, no addons)
# -------------------------------------------------------------------

RAD_45 = math.pi / 4.0


def _decode_and_resize(img_bytes: tf.Tensor, image_size: tuple[int, int]) -> tf.Tensor:
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, image_size, antialias=True)
    img = tf.cast(img, tf.float32) / 255.0
    return img


def rotate_image(img, radians):
    img_shape = tf.shape(img)
    h = tf.cast(img_shape[0], tf.float32)
    w = tf.cast(img_shape[1], tf.float32)

    cx = w / 2.0
    cy = h / 2.0

    cos_a = tf.math.cos(radians)
    sin_a = tf.math.sin(radians)

    # Build transform: maps output -> input coordinates
    # [a0, a1, a2, a3, a4, a5, a6, a7]
    # x_in = a0 * x_out + a1 * y_out + a2
    # y_in = a3 * x_out + a4 * y_out + a5
    tx = cx - cos_a * cx + sin_a * cy
    ty = cy - sin_a * cx - cos_a * cy

    transform = tf.stack([cos_a, -sin_a, tx,
                          sin_a,  cos_a, ty,
                          0.0,    0.0])
    transform = tf.reshape(transform, (1, 8))

    img_b = tf.expand_dims(img, 0)

    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=img_b,
        transforms=transform,
        output_shape=tf.cast(tf.shape(img)[:2], tf.int32),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,          # 👈 this is the missing argument
    )

    return tf.squeeze(out, 0)



def blur_image(img: tf.Tensor) -> tf.Tensor:
    """Simple 3x3 Gaussian-ish blur."""
    kernel = tf.constant(
        [[1, 2, 1],
         [2, 4, 2],
         [1, 2, 1]],
        dtype=tf.float32,
    )
    kernel = kernel / tf.reduce_sum(kernel)
    kernel = tf.reshape(kernel, [3, 3, 1, 1])

    img_b = tf.expand_dims(img, 0)
    # apply same kernel to all channels
    img_blur = tf.nn.depthwise_conv2d(
        img_b, tf.tile(kernel, [1, 1, 3, 1]), strides=[1, 1, 1, 1], padding="SAME"
    )
    return img_blur[0]


def zoom_image(img: tf.Tensor, central_fraction: float = 0.8) -> tf.Tensor:
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    cropped = tf.image.central_crop(img, central_fraction=central_fraction)
    cropped = tf.image.resize(cropped, (h, w), antialias=True)
    return cropped


# -------------------------------------------------------------------
# Dataset construction (with deterministic augmentations)
# -------------------------------------------------------------------

def make_datasets(
    root_dir: str,
    batch_size: int = 32,
    image_size: tuple[int, int] = (224, 224),
) -> Tuple[tf.data.Dataset, tf.data.Dataset, int, List[str]]:
    root = Path(root_dir)
    train_dir = root / "train"
    val_img_dir = root / "val" / "images"

    assert train_dir.exists(), f"Missing: {train_dir}"
    assert val_img_dir.exists(), f"Missing: {val_img_dir}"

    wnid_to_index, valfile_to_index, wnids = build_lookup_tables(root)
    num_classes = len(wnids)

    def _load_train(path: tf.Tensor):
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

    # one path -> Dataset of 5 augmented images with same label
    def _augment_from_path(path: tf.Tensor) -> tf.data.Dataset:
        img, label = _load_train(path)

        flip = tf.image.flip_left_right(img)
        rot_p = rotate_image(img, RAD_45)
        rot_m = rotate_image(img, -RAD_45)
        zoom_blur = blur_image(zoom_image(img))

        imgs = tf.stack([img, flip, rot_p, rot_m, zoom_blur], axis=0)
        labels = tf.fill([5], label)
        return tf.data.Dataset.from_tensor_slices((imgs, labels))

    # list_files for train and val
    train_files = tf.data.Dataset.list_files(
        str(train_dir / "*" / "images" / "*.JPEG"), shuffle=True
    )
    val_files = tf.data.Dataset.list_files(
        str(val_img_dir / "*.JPEG"), shuffle=False
    )

    # train: flat_map to materialize 5 augmented samples per file
    train_ds = (
        train_files
        .flat_map(_augment_from_path)
        .shuffle(2000)
        .batch(batch_size, drop_remainder=True)
        .prefetch(AUTOTUNE)
    )

    # val: no augmentation
    val_ds = (
        val_files
        .map(_load_val, num_parallel_calls=AUTOTUNE)
        .batch(batch_size, drop_remainder=False)
        .prefetch(AUTOTUNE)
    )

    return train_ds, val_ds, num_classes, wnids


# -------------------------------------------------------------------
# Model
# -------------------------------------------------------------------

def build_model(num_classes: int) -> tuple[keras.Model, keras.Model]:
    base = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    inp = keras.Input(shape=(224, 224, 3))
    x = base(inp, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    out = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inp, out, name="mobilenetv2_tiny_imagenet_aug")
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
        description="MobileNetV2 on Tiny-ImageNet-200 with handcrafted augmentations"
    )
    ap.add_argument("--data_root", required=True, help="tiny-imagenet-200 root")
    ap.add_argument("--run_name", default="tinyimagenet_aug")

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--image_size", type=int, nargs=2, default=(224, 224))

    ap.add_argument("--epochs_warmup", type=int, default=5)
    ap.add_argument("--epochs_finetune", type=int, default=60)
    ap.add_argument("--unfreeze_last", type=int, default=120)
    ap.add_argument("--lr_warmup", type=float, default=3e-4)
    ap.add_argument("--lr_finetune", type=float, default=1e-4)
    ap.add_argument("--early_stop_patience", type=int, default=10)

    ap.add_argument(
        "--monitor",
        type=str,
        default="val_loss",
        choices=["val_loss", "val_accuracy"],
        help="metric for LR schedule / early stopping / checkpointing",
    )
    ap.add_argument("--reduce_lr_patience", type=int, default=3)
    ap.add_argument("--reduce_lr_factor", type=float, default=0.5)
    ap.add_argument("--min_lr", type=float, default=1e-6)

    args = ap.parse_args()

    image_size = tuple(args.image_size)

    run_dir = Path("runs") / args.run_name
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, num_classes, wnids = make_datasets(
        root_dir=args.data_root,
        batch_size=args.batch_size,
        image_size=image_size,
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
    warm_hist = model.fit(
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

    # fine-tuning
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
        keras.callbacks.EarlyStopping(
            monitor=args.monitor,
            patience=args.early_stop_patience,
            restore_best_weights=True,
            verbose=1,
            mode=monitor_mode,
        ),
        keras.callbacks.CSVLogger(str(run_dir / "finetune_history.csv")),
    ]

    print(f"[fine-tune] unfreezing last {args.unfreeze_last} backbone layers (BN frozen)")
    print(f"[diag] trainables (fine-tune): {count_trainable_params(model):,}")

    fine_hist = model.fit(
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
    print(f"Saved to:\n  - {ckpt_dir / 'best.keras'}\n  - {final_path}\n"
          f"  - {run_dir / 'warmup_history.csv'}\n  - {run_dir / 'finetune_history.csv'}\n")


if __name__ == "__main__":
    main()
