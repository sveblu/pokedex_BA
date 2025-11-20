#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
import argparse
from pathlib import Path
from typing import Tuple, List

import tensorflow as tf
from tensorflow import keras

# ----------------------------- Runtime configuration -----------------------------

tf.config.optimizer.set_jit(False)

for g in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

AUTOTUNE = tf.data.AUTOTUNE


# ----------------------------- Utilities ----------------------------------------

def count_trainable_params(model: keras.Model) -> int:
    total = 0
    for v in model.trainable_variables:
        n = 1
        for d in v.shape:
            n *= int(d) if d is not None else 1
        total += int(n)
    return total


# ----------------------------- Data pipeline ------------------------------------

def build_lookup_tables(root: Path) -> Tuple[tf.lookup.StaticHashTable, tf.lookup.StaticHashTable, List[str]]:
    wnids_path = root / "wnids.txt"
    assert wnids_path.exists(), f"Missing: {wnids_path}"
    wnids = [w.strip() for w in wnids_path.read_text().splitlines() if w.strip()]

    wnid_keys = tf.constant(wnids, dtype=tf.string)
    wnid_vals = tf.range(len(wnids), dtype=tf.int32)
    wnid_init = tf.lookup.KeyValueTensorInitializer(wnid_keys, wnid_vals)
    wnid_to_index = tf.lookup.StaticHashTable(wnid_init, default_value=-1)

    ann_path = root / "val" / "val_annotations.txt"
    assert ann_path.exists(), f"Missing: {ann_path}"
    val_lines = [ln.strip() for ln in ann_path.read_text().splitlines() if ln.strip()]
    val_files, val_inds = [], []
    for ln in val_lines:
        parts = ln.split("\t")
        fname = parts[0]
        wnid = parts[1]
        idx = wnids.index(wnid)
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
) -> Tuple[tf.data.Dataset, tf.data.Dataset, int, List[str]]:

    root = Path(root_dir)
    train_dir = root / "train"
    val_img_dir = root / "val" / "images"

    assert train_dir.exists(), f"Missing: {train_dir}"
    assert val_img_dir.exists(), f"Missing: {val_img_dir}"
    assert (root / "wnids.txt").exists(), f"Missing: {root / 'wnids.txt'}"
    assert (root / "val" / "val_annotations.txt").exists(), "val_annotations.txt missing"

    wnid_to_index, valfile_to_index, wnids = build_lookup_tables(root)
    num_classes = len(wnids)

    def _decode_and_resize(img_bytes: tf.Tensor) -> tf.Tensor:
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, image_size, antialias=True)
        img = tf.cast(img, tf.float32) / 255.0
        return img

    def _load_train(path: tf.Tensor):
        img_bytes = tf.io.read_file(path)
        img = _decode_and_resize(img_bytes)
        parts = tf.strings.split(path, os.sep)
        wnid = parts[-3]
        label = wnid_to_index.lookup(wnid)
        return img, label

    def _load_val(path: tf.Tensor):
        img_bytes = tf.io.read_file(path)
        img = _decode_and_resize(img_bytes)
        fname = tf.strings.split(path, os.sep)[-1]
        label = valfile_to_index.lookup(fname)
        return img, label

    train_files = tf.data.Dataset.list_files(str(train_dir / "*" / "images" / "*.JPEG"), shuffle=True)
    val_files = tf.data.Dataset.list_files(str(val_img_dir / "*.JPEG"), shuffle=False)

    # --- NO AUGMENTATIONS VERSION ---
    # Just decode + resize + batch + prefetch

    def _map_train(path: tf.Tensor):
        img, label = _load_train(path)
        return img, label

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

    if cache_to_disk:
        cache_root = Path.home() / ".tfdata_cache" / "tiny-imagenet-200-noaug"
        cache_root.mkdir(parents=True, exist_ok=True)
        train_ds = train_ds.cache(str(cache_root / "train.cache"))
        val_ds = val_ds.cache(str(cache_root / "val.cache"))

    return train_ds, val_ds, num_classes, wnids


# ----------------------------- Model definition ---------------------------------

def build_model(num_classes: int) -> tuple[keras.Model, keras.Model]:
    base = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    x = keras.Input(shape=(224, 224, 3))
    y = base(x, training=False)
    y = keras.layers.GlobalAveragePooling2D()(y)
    y = keras.layers.Dropout(0.2)(y)
    out = keras.layers.Dense(num_classes, activation="softmax")(y)
    model = keras.Model(x, out, name="mobilenetv2_tiny_imagenet")
    return model, base


def unfreeze_tail(base: keras.Model, num_unfrozen: int) -> None:
    for layer in base.layers:
        layer.trainable = False
    for layer in base.layers[-num_unfrozen:]:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True


# ----------------------------- Diagnostics --------------------------------------

def print_mapping_summary(wnids: List[str], limit: int = 10) -> None:
    print("\nWNIDs (order -> class_id):")
    for i, w in enumerate(wnids[:limit]):
        print(f"  {i:<2d} -> {w}")
    if len(wnids) > limit:
        print("  ...")
    print(f"\nClasses detected: {len(wnids)}\n")


def quick_distribution_check(model: keras.Model, ds: tf.data.Dataset, name: str, steps: int = 3) -> None:
    import numpy as np
    probs_all, preds_all = [], []
    for _, (xb, _) in enumerate(ds.take(steps)):
        p = model.predict(xb, verbose=0)
        probs_all.append(p)
        preds_all.append(p.argmax(axis=1))
    probs = np.concatenate(probs_all, axis=0)
    preds = np.concatenate(preds_all, axis=0)
    mean = probs.mean(axis=0)
    bincount = np.bincount(preds, minlength=probs.shape[1])
    print(f"[diag] {name} softmax mean (first 5): {mean[:5].round(3)}")
    print(f"[diag] {name} argmax counts (first 5): {bincount[:5]}\n")


# ----------------------------- Training routine ---------------------------------

def main():
    parser = argparse.ArgumentParser(description="MobileNetV2 on Tiny-ImageNet-200 (NO AUGMENTATIONS)")

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, nargs=2, default=(224, 224))
    parser.add_argument("--cache_to_disk", action="store_true")
    parser.add_argument("--no_cache", action="store_true")

    parser.add_argument("--epochs_warmup", type=int, default=5)
    parser.add_argument("--epochs_finetune", type=int, default=20)

    parser.add_argument("--unfreeze_last", type=int, default=40)
    parser.add_argument("--lr_warmup", type=float, default=3e-4)
    parser.add_argument("--lr_finetune", type=float, default=1e-4)
    parser.add_argument("--early_stop_patience", type=int, default=7)
    parser.add_argument("--monitor", type=str, default="val_loss",
                        choices=["val_loss", "val_accuracy"])
    parser.add_argument("--reduce_lr_patience", type=int, default=3)
    parser.add_argument("--reduce_lr_factor", type=float, default=0.5)
    parser.add_argument("--min_lr", type=float, default=1e-6)

    args = parser.parse_args()

    cache_to_disk = False if args.no_cache else args.cache_to_disk

    train_ds, val_ds, num_classes, wnids = make_datasets(
        root_dir=args.data_root,
        batch_size=args.batch_size,
        image_size=tuple(args.image_size),
        cache_to_disk=cache_to_disk,
    )

    print_mapping_summary(wnids, limit=10)

    model, base = build_model(num_classes)

    # ---------------- Warm-up ----------------
    base.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr_warmup),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    train_steps = tf.data.experimental.cardinality(train_ds).numpy()
    val_steps = tf.data.experimental.cardinality(val_ds).numpy()

    print(f"[warm-up] trainable params: {count_trainable_params(model):,}")

    hist_warm = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_warmup,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        verbose=2,
        callbacks=[
            keras.callbacks.CSVLogger(f"runs/{args.run_name}/warmup_history.csv", append=False)
        ],
    )

    # ---------------- Fine-tuning ----------------
    unfreeze_tail(base, num_unfrozen=args.unfreeze_last)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr_finetune),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    out_dir = Path("runs") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_dir / "best.keras"),
            monitor=args.monitor,
            mode="max" if args.monitor == "val_accuracy" else "min",
            save_best_only=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=args.monitor,
            factor=args.reduce_lr_factor,
            patience=args.reduce_lr_patience,
            min_lr=args.min_lr,
            verbose=1,
            mode="max" if args.monitor == "val_accuracy" else "min",
        ),
        keras.callbacks.CSVLogger(str(out_dir / "finetune_history.csv"), append=False),
    ]

    print(f"[fine-tune] unfreezing last {args.unfreeze_last} backbone layers.")

    hist_ft = model.fit(
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

    quick_distribution_check(model, val_ds, name="VAL", steps=3)
    quick_distribution_check(model, train_ds, name="TRAIN", steps=3)

    eval_train = model.evaluate(train_ds, verbose=0)
    eval_val = model.evaluate(val_ds, verbose=0)
    print(f"\nEVAL — train: acc={eval_train[1]:.3f}, val: acc={eval_val[1]:.3f}\n")
    print(f"Saved:\n  - {ckpt_dir / 'best.keras'}\n  - {final_path}\n")


if __name__ == "__main__":
    main()
# ----------------------------- End of file --------------------------------------