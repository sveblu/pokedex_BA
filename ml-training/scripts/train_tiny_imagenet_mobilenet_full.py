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
# (UNCHANGED — omitted here for brevity, keep your full code exactly as-is)
# -------------------------------------------------------------------------------

# KEEP ALL THE SAME FUNCTIONS: build_lookup_tables, make_datasets, build_model,
# unfreeze_tail, print_mapping_summary, quick_distribution_check


# ----------------------------- Training routine ---------------------------------

def main():
    parser = argparse.ArgumentParser(description="MobileNetV2 on Tiny-ImageNet-200 (memory-safe pipeline)")

    # NEW: allow user to choose output directory
    parser.add_argument("--run_name", type=str, default="tinyimagenet_run",
                        help="Folder name inside runs/ where logs and models will be saved")

    # ---------------- existing args (unchanged) ----------------
    parser.add_argument("--data_root", type=str, required=True)
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

    # ---------------- NEW: run directory ----------------
    run_dir = Path("runs") / args.run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

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

    warmup_csv = keras.callbacks.CSVLogger(str(run_dir / "warmup_history.csv"))

    train_steps = tf.data.experimental.cardinality(train_ds).numpy()
    val_steps = tf.data.experimental.cardinality(val_ds).numpy()

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_warmup,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        verbose=2,
        callbacks=[warmup_csv],
    )

    # ---------------- Fine-tuning ----------------
    unfreeze_tail(base, num_unfrozen=args.unfreeze_last)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr_finetune),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    ft_callbacks = [
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
        keras.callbacks.EarlyStopping(
            monitor=args.monitor,
            patience=args.early_stop_patience,
            restore_best_weights=True,
            verbose=1,
            mode="max" if args.monitor == "val_accuracy" else "min",
        ),
        keras.callbacks.CSVLogger(str(run_dir / "finetune_history.csv")),  # NEW
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_finetune,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=ft_callbacks,
        verbose=2,
    )

    # ---------------- Saving ----------------
    final_path = ckpt_dir / "final.keras"
    model.save(final_path)

    eval_train = model.evaluate(train_ds, verbose=0)
    eval_val = model.evaluate(val_ds, verbose=0)
    print(f"\nEVAL — train: acc={eval_train[1]:.3f}, val: acc={eval_val[1]:.3f}\n")
    print(f"Saved:\n  - {ckpt_dir / 'best.keras'}\n  - {final_path}\n")


if __name__ == "__main__":
    main()
