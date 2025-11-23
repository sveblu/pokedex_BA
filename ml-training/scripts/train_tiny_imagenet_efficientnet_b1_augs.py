#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
import argparse
from pathlib import Path
from typing import Tuple, List

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.efficientnet import preprocess_input


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
# Helpers
# -------------------------------------------------------------------

def count_trainable_params(model: keras.Model) -> int:
    total = 0
    for v in model.trainable_variables:
        n = 1
        for d in v.shape:
            n *= int(d) if d is not None else 1
        total += int(n)
    return total


# -------------------------------------------------------------------
# Label lookup
# -------------------------------------------------------------------

def build_lookup_tables(root: Path):
    wnids_path = root / "wnids.txt"
    wnids = [w.strip() for w in wnids_path.read_text().splitlines() if w.strip()]

    wnid_to_index = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            tf.constant(wnids),
            tf.range(len(wnids), dtype=tf.int32),
        ),
        default_value=-1,
    )

    ann_path = root / "val" / "val_annotations.txt"
    lines = [ln.strip() for ln in ann_path.read_text().splitlines() if ln.strip()]

    val_files, val_inds = [], []
    for ln in lines:
        fname, wnid, *_ = ln.split("\t")
        idx = wnids.index(wnid)
        val_files.append(fname)
        val_inds.append(idx)

    val_lookup = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            tf.constant(val_files),
            tf.constant(val_inds),
        ),
        default_value=-1,
    )

    return wnid_to_index, val_lookup, wnids


# -------------------------------------------------------------------
# Decode + EfficientNet preprocessing
# -------------------------------------------------------------------

def _decode_and_resize(img_bytes: tf.Tensor, image_size):
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, image_size, antialias=True)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img


# -------------------------------------------------------------------
# Dataset w/ augmentations
# -------------------------------------------------------------------

def make_datasets(root_dir, batch_size, image_size, cache_to_disk):

    root = Path(root_dir)
    wnid_to_index, valfile_to_index, wnids = build_lookup_tables(root)
    num_classes = len(wnids)

    train_files = tf.data.Dataset.list_files(
        str(root / "train" / "*" / "images" / "*.JPEG"),
        shuffle=True
    )
    val_files = tf.data.Dataset.list_files(
        str(root / "val" / "images" / "*.JPEG"),
        shuffle=False
    )

    aug = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.125, fill_mode="reflect"),
            keras.layers.RandomZoom(
                height_factor=(-0.3, 0.0),
                width_factor=(-0.3, 0.0),
                fill_mode="reflect",
            ),
        ]
    )

    def _load_train(path):
        img_bytes = tf.io.read_file(path)
        img = _decode_and_resize(img_bytes, image_size)
        wnid = tf.strings.split(path, os.sep)[-3]
        label = wnid_to_index.lookup(wnid)
        return img, label

    def _load_val(path):
        img_bytes = tf.io.read_file(path)
        img = _decode_and_resize(img_bytes, image_size)
        fname = tf.strings.split(path, os.sep)[-1]
        label = valfile_to_index.lookup(fname)
        return img, label

    train_raw = train_files.map(_load_train, AUTOTUNE)
    val_raw = val_files.map(_load_val, AUTOTUNE)

    if cache_to_disk:
        cache_root = Path.home() / ".tfdata_cache" / "tiny-imagenet-efnetb1-aug"
        cache_root.mkdir(parents=True, exist_ok=True)
        train_raw = train_raw.cache(str(cache_root / "train.cache"))
        val_raw = val_raw.cache(str(cache_root / "val.cache"))

    train_ds = (
        train_raw
        .map(lambda x, y: (aug(x, training=True), y), AUTOTUNE)
        .shuffle(2000)
        .batch(batch_size, drop_remainder=True)
        .prefetch(2)
    )

    val_ds = val_raw.batch(batch_size).prefetch(2)

    return train_ds, val_ds, num_classes, wnids


# -------------------------------------------------------------------
# Model (EfficientNet-B1 @ 240x240)
# -------------------------------------------------------------------

def build_model(num_classes):
    base = keras.applications.EfficientNetB1(
        input_shape=(240, 240, 3),      # 🔥 correct default resolution
        include_top=False,
        weights="imagenet",
    )
    inp = keras.Input(shape=(240, 240, 3))
    x = base(inp, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inp, out)
    return model, base


def unfreeze_tail(base, n):
    for layer in base.layers:
        layer.trainable = False
    for layer in base.layers[-n:]:
        if not isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = True


# -------------------------------------------------------------------
# Training
# -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--run_name", default="tinyimagenet_efficientnet_b1_aug")
    ap.add_argument("--batch_size", type=int, default=32)

    ap.add_argument("--epochs_warmup", type=int, default=5)
    ap.add_argument("--epochs_finetune", type=int, default=30)
    ap.add_argument("--unfreeze_last", type=int, default=160)

    ap.add_argument("--lr_warmup", type=float, default=3e-4)
    ap.add_argument("--lr_finetune", type=float, default=1e-4)

    ap.add_argument("--no_cache", action="store_true")
    ap.add_argument("--monitor", type=str, default="val_loss",
                    choices=["val_loss", "val_accuracy"])
    ap.add_argument("--reduce_lr_patience", type=int, default=3)
    ap.add_argument("--reduce_lr_factor", type=float, default=0.5)
    ap.add_argument("--min_lr", type=float, default=1e-6)

    args = ap.parse_args()

    image_size = (240, 240)   # 🔥 correct B1 default

    run_dir = Path("runs") / args.run_name
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, num_classes, wnids = make_datasets(
        args.data_root,
        args.batch_size,
        image_size,
        cache_to_disk=not args.no_cache,
    )

    model, base = build_model(num_classes)

    # Warm-up
    base.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(args.lr_warmup),
        loss="sparse_categorical_crossentropy",
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
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
            keras.callbacks.CSVLogger(str(run_dir / "warmup_history.csv"))
        ],
    )

    # Fine-tuning
    unfreeze_tail(base, args.unfreeze_last)
    model.compile(
        optimizer=keras.optimizers.Adam(args.lr_finetune),
        loss="sparse_categorical_crossentropy",
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    )

    monitor_mode = "max" if args.monitor == "val_accuracy" else "min"

    cb = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_dir / "best.keras"),
            monitor=args.monitor,
            mode=monitor_mode,
            save_best_only=True,
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

    print(f"[fine-tune] unfreezing last {args.unfreeze_last} layers")
    print(f"[diag] trainables: {count_trainable_params(model):,}")

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_finetune,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=cb,
        verbose=2,
    )

    model.save(str(ckpt_dir / "final.keras"))
    print("\nSaved best and final checkpoints.\n")


if __name__ == "__main__":
    main()
