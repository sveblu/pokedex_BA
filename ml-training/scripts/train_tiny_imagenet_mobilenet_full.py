#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MobileNetV2 on Tiny-ImageNet-200 with a memory-safe tf.data pipeline.

Key characteristics
- Fixed class ordering from wnids.txt to ensure stable label indices.
- Validation labels constructed from val_annotations.txt.
- No in-memory dataset cache (prevents host-pinned OOM); optional on-disk cache.
- Two-stage training: classifier head warm-up, then partial backbone fine-tuning.
- Conservative defaults for 6 GB GPUs; configurable via CLI flags.

Directory layout expected (standard Tiny-ImageNet):
  <root>/
    wnids.txt
    words.txt
    train/<WNID>/images/*.JPEG
    val/images/*.JPEG
    val/val_annotations.txt
"""

from __future__ import annotations
import os
import argparse
from pathlib import Path
from typing import Tuple

import tensorflow as tf
from tensorflow import keras

# ----------------------------- Runtime configuration -----------------------------

# Disable XLA to avoid large host-pinned buffers during data ingest.
# It can be re-enabled later once the pipeline is stable.
tf.config.optimizer.set_jit(False)

# Enable dynamic GPU memory growth for WSL / consumer GPUs.
gpus = tf.config.list_physical_devices("GPU")
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass  # Fallback silently if not supported

AUTOTUNE = tf.data.AUTOTUNE


# ----------------------------- Data pipeline ------------------------------------

def build_lookup_tables(root: Path) -> Tuple[tf.lookup.StaticHashTable, tf.lookup.StaticHashTable, list[str]]:
    """
    Builds two lookup tables:
      - wnid_to_index: maps WNID string -> class index [0..C-1]
      - valfile_to_index: maps validation filename (e.g., 'val_0.JPEG') -> class index
    Returns both tables and the ordered list of WNIDs.
    """
    wnids = [w.strip() for w in (root / "wnids.txt").read_text().splitlines() if w.strip()]
    assert len(wnids) > 0, "wnids.txt is empty or missing classes."

    # WNID -> integer index
    wnid_keys = tf.constant(wnids, dtype=tf.string)
    wnid_vals = tf.range(len(wnids), dtype=tf.int32)
    wnid_init = tf.lookup.KeyValueTensorInitializer(wnid_keys, wnid_vals)
    wnid_to_index = tf.lookup.StaticHashTable(wnid_init, default_value=-1)

    # Validation filename -> integer index (from val_annotations.txt)
    ann_path = root / "val" / "val_annotations.txt"
    assert ann_path.exists(), f"Missing: {ann_path}"
    val_lines = [ln.strip() for ln in ann_path.read_text().splitlines() if ln.strip()]
    val_files, val_inds = [], []
    for ln in val_lines:
        # Format: "<filename>\t<wnid>\t<x>\t<y>\t<w>\t<h>"
        parts = ln.split("\t")
        fname = parts[0]
        wnid = parts[1]
        idx = wnids.index(wnid)  # raises if inconsistent
        val_files.append(fname)
        val_inds.append(idx)

    vf_keys = tf.constant(val_files, dtype=tf.string)
    vf_vals = tf.constant(val_inds, dtype=tf.int32)
    vf_init = tf.lookup.KeyValueTensorInitializer(vf_keys, vf_vals)
    valfile_to_index = tf.lookup.StaticHashTable(vf_init, default_value=-1)

    return wnid_to_index, valfile_to_index, wnids


def make_datasets(
    root_dir: str,
    batch_size: int = 32,
    image_size: tuple[int, int] = (224, 224),
    cache_to_disk: bool = True,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, int, list[str]]:
    """
    Creates memory-safe tf.data pipelines for train and validation.

    Design decisions
    - No in-memory `Dataset.cache()` to avoid large host-pinned allocations.
    - Optional on-disk cache at ~/.tfdata_cache/tiny-imagenet-200/*.cache.
    - Fixed class indices via wnids.txt; validation labels from val_annotations.txt.
    """
    root = Path(root_dir)
    train_dir = root / "train"
    val_img_dir = root / "val" / "images"

    assert train_dir.exists(), f"Missing: {train_dir}"
    assert val_img_dir.exists(), f"Missing: {val_img_dir}"
    assert (root / "wnids.txt").exists(), f"Missing: {root / 'wnids.txt'}"
    assert (root / "val" / "val_annotations.txt").exists(), "val_annotations.txt missing"

    wnid_to_index, valfile_to_index, wnids = build_lookup_tables(root)
    num_classes = len(wnids)

    # Decoder and basic preprocessing
    def _decode_and_resize(img_bytes: tf.Tensor) -> tf.Tensor:
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, image_size, antialias=True)
        img = tf.cast(img, tf.float32) / 255.0
        return img

    # Train: derive label from directory name (.../train/<WNID>/images/<file>)
    def _load_train(path: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        img_bytes = tf.io.read_file(path)
        img = _decode_and_resize(img_bytes)
        parts = tf.strings.split(path, os.sep)
        wnid = parts[-3]  # "<WNID>"
        label = wnid_to_index.lookup(wnid)
        return img, label

    # Val: derive label from filename using val_annotations mapping
    def _load_val(path: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        img_bytes = tf.io.read_file(path)
        img = _decode_and_resize(img_bytes)
        fname = tf.strings.split(path, os.sep)[-1]
        label = valfile_to_index.lookup(fname)
        return img, label

    # Lists of files
    train_files = tf.data.Dataset.list_files(str(train_dir / "*" / "images" / "*.JPEG"), shuffle=True)
    val_files = tf.data.Dataset.list_files(str(val_img_dir / "*.JPEG"), shuffle=False)

    # Augmentations (lightweight; compatible with validation pipeline)
    aug = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.05),
            keras.layers.RandomZoom(0.1),
        ],
        name="augment",
    )

    def _map_train(path: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        img, label = _load_train(path)
        img = aug(img)
        return img, label

    # Assemble datasets (no in-memory cache)
    train_ds = (
        train_files
        .shuffle(20_000)
        .map(_map_train, num_parallel_calls=AUTOTUNE)
        .batch(batch_size, drop_remainder=True)
        .prefetch(2)
    )
    val_ds = (
        val_files
        .map(_load_val, num_parallel_calls=AUTOTUNE)
        .batch(batch_size, drop_remainder=False)
        .prefetch(2)
    )

    # Optional: on-disk cache (safe). Speeds up subsequent epochs without consuming RAM.
    if cache_to_disk:
        cache_root = Path.home() / ".tfdata_cache" / "tiny-imagenet-200"
        cache_root.mkdir(parents=True, exist_ok=True)
        train_ds = train_ds.cache(str(cache_root / "train.cache"))
        val_ds = val_ds.cache(str(cache_root / "val.cache"))

    return train_ds, val_ds, num_classes, wnids


# ----------------------------- Model definition ---------------------------------

def build_model(num_classes: int) -> keras.Model:
    """
    Constructs MobileNetV2 backbone with a classifier head.
    """
    base = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    x = keras.Input(shape=(224, 224, 3))
    y = base(x, training=False)  # frozen batchnorm statistics during warm-up
    y = keras.layers.GlobalAveragePooling2D()(y)
    y = keras.layers.Dropout(0.2)(y)
    out = keras.layers.Dense(num_classes, activation="softmax")(y)
    model = keras.Model(x, out, name="mobilenetv2_tiny_imagenet")
    return model, base


def unfreeze_tail(base: keras.Model, num_unfrozen: int = 40) -> None:
    """
    Unfreezes the last `num_unfrozen` layers of `base` for fine-tuning, while keeping
    BatchNormalization layers frozen to preserve stable statistics.
    """
    # Freeze all first
    for layer in base.layers:
        layer.trainable = False

    # Unfreeze tail except BatchNorm
    for layer in base.layers[-num_unfrozen:]:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True


# ----------------------------- Diagnostics --------------------------------------

def print_mapping_summary(wnids: list[str], limit: int = 10) -> None:
    """
    Prints the first few WNIDs and their indices for sanity checking.
    """
    print("\nWNIDs (order -> class_id):")
    for i, w in enumerate(wnids[:limit]):
        print(f"  {i:<2d} -> {w}")
    if len(wnids) > limit:
        print("  ...")
    print(f"\nClasses detected: {len(wnids)}\n")


def quick_distribution_check(model: keras.Model, ds: tf.data.Dataset, name: str, steps: int = 2) -> None:
    """
    Computes softmax mean and argmax histograms over a few batches to ensure
    outputs are non-degenerate after training.
    """
    import numpy as np
    probs_all = []
    preds_all = []
    for i, (xb, _) in enumerate(ds.take(steps)):
        p = model.predict(xb, verbose=0)
        probs_all.append(p)
        preds_all.append(p.argmax(axis=1))
    probs = np.concatenate(probs_all, axis=0)
    preds = np.concatenate(preds_all, axis=0)
    mean = probs.mean(axis=0)
    bincount = np.bincount(preds, minlength=probs.shape[1])
    topk = 5 if probs.shape[1] >= 5 else probs.shape[1]
    print(f"[diag] {name} softmax mean (first {topk}): {mean[:topk].round(3)}")
    print(f"[diag] {name} argmax counts (first {topk}): {bincount[:topk]}\n")


# ----------------------------- Training routine ---------------------------------

def main():
    parser = argparse.ArgumentParser(description="MobileNetV2 on Tiny-ImageNet-200 (memory-safe pipeline)")
    parser.add_argument("--data_root", type=str, required=True, help="Path to tiny-imagenet-200 root directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--epochs_warmup", type=int, default=5, help="Classifier head warm-up epochs")
    parser.add_argument("--epochs_finetune", type=int, default=20, help="Fine-tuning epochs")
    parser.add_argument("--unfreeze_layers", type=int, default=40, help="Number of backbone layers to unfreeze")
    parser.add_argument("--cache_to_disk", action="store_true", help="Enable on-disk dataset cache")
    parser.add_argument("--no_cache", action="store_true", help="Disable caching entirely")
    args = parser.parse_args()

    # Resolve cache policy
    cache_to_disk = False if args.no_cache else args.cache_to_disk

    # Build datasets
    train_ds, val_ds, num_classes, wnids = make_datasets(
        root_dir=args.data_root,
        batch_size=args.batch_size,
        image_size=(224, 224),
        cache_to_disk=cache_to_disk,
    )

    # Print a short mapping summary
    print_mapping_summary(wnids, limit=10)

    # Construct model
    model, base = build_model(num_classes)

    # ---------------- Warm-up: train classifier head only ----------------
    base.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    # Compute safe steps (avoid partial last batch for stability)
    train_steps = tf.data.experimental.cardinality(train_ds).numpy()
    val_steps = tf.data.experimental.cardinality(val_ds).numpy()
    print(f"[warm-up] trainable params: {model.trainable_weights and sum(int(tf.size(v)) for v in model.trainable_variables)}")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_warmup,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        verbose=2,
    )

    # ---------------- Fine-tuning: unfreeze tail ----------------
    unfreeze_tail(base, num_unfrozen=args.unfreeze_layers)
    # Compile with a lower LR; freeze BatchNorms implicitly respected
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    # Callbacks: learning rate scheduling, early stop, checkpointing
    ckpt_dir = Path("checkpoints/tiny_mobilenetv2_full")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_dir / "best.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    print(f"[fine-tune] unfreezing last {args.unfreeze_layers} backbone layers (BatchNorm frozen).")
    # Simple indicator of how many params will update
    trainable_params = sum(v.numpy().size for v in model.trainable_variables)
    print(f"[diag] trainables (fine-tune): {trainable_params:,} params")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_finetune,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=2,
    )

    # Save final model
    final_path = ckpt_dir / "final.keras"
    model.save(final_path)

    # Quick distribution checks to ensure outputs are non-degenerate
    quick_distribution_check(model, val_ds, name="VAL", steps=3)
    quick_distribution_check(model, train_ds, name="TRAIN", steps=3)

    # Final evaluation summary
    eval_train = model.evaluate(train_ds, verbose=0)
    eval_val = model.evaluate(val_ds, verbose=0)
    print(f"\nEVAL — train: acc={eval_train[1]:.3f}, val: acc={eval_val[1]:.3f}\n")
    print(f"✅ Training finished. Saved:\n  - {ckpt_dir / 'best.keras'}\n  - {final_path}\n")


if __name__ == "__main__":
    main()
