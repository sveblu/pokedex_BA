#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # hide INFO & WARNING, show only ERROR

import argparse
import json
import sys
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

AUTOTUNE = tf.data.AUTOTUNE


# -------------------------------------------------------------------
# Small helpers
# -------------------------------------------------------------------

def non_empty_subdirs(p: Path):
    out = []
    for d in sorted([x for x in p.iterdir() if x.is_dir()]):
        if any(d.rglob("*")):
            out.append(d.name)
    return out


def build_ds(root, img_size=(224, 224), batch=32):
    root = Path(root)
    train_root, val_root = root / "train", root / "val"
    assert train_root.is_dir() and val_root.is_dir(), "train/ and val/ required"

    train_classes = non_empty_subdirs(train_root)
    val_classes = non_empty_subdirs(val_root)

    only_train = sorted(set(train_classes) - set(val_classes))
    only_val = sorted(set(val_classes) - set(train_classes))

    if only_train or only_val:
        print("⚠ class mismatch detected", file=sys.stderr)
        if only_train:
            print("  present only in train (ignored):", only_train, file=sys.stderr)
        if only_val:
            print("  present only in val (ignored):  ", only_val, file=sys.stderr)

    classes = [c for c in train_classes if c in val_classes]
    if not classes:
        raise RuntimeError("No overlapping non-empty classes between train/ and val/.")

    tr_raw = tf.keras.utils.image_dataset_from_directory(
        train_root,
        image_size=img_size,
        batch_size=batch,
        shuffle=True,
        class_names=classes,
    )
    va_raw = tf.keras.utils.image_dataset_from_directory(
        val_root,
        image_size=img_size,
        batch_size=batch,
        shuffle=False,
        class_names=classes,
    )

    class_names = tr_raw.class_names
    num_classes = len(class_names)

    aug = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.08),
            keras.layers.RandomZoom(0.15),
            keras.layers.RandomTranslation(0.05, 0.05),
            keras.layers.RandomContrast(0.1),
        ],
        name="aug",
    )

    def norm(x, y):
        x = tf.cast(x, tf.float32) / 255.0
        return x, y

    tr = (
        tr_raw
        .map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
        .map(norm, num_parallel_calls=AUTOTUNE)
        .prefetch(2)
    )
    va = va_raw.map(norm, num_parallel_calls=AUTOTUNE).prefetch(2)

    return tr, va, num_classes, class_names


def freeze_bn(model):
    for l in model.layers:
        if isinstance(l, keras.layers.BatchNormalization):
            l.trainable = False


def unfreeze_tail(model, n_layers):
    unfrozen = 0
    # skip last layer (new head)
    for l in reversed(model.layers[:-1]):
        if unfrozen >= n_layers:
            break
        if not isinstance(l, keras.layers.BatchNormalization):
            l.trainable = True
        unfrozen += 1


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Finetune Tiny-ImageNet-pretrained model on Pokémon dataset"
    )
    ap.add_argument("--pokemon_root", required=True,
                    help="Root with train/ and val/ Pokémon folders")
    ap.add_argument(
        "--pretrain_run",
        required=True,
        help="Name of Tiny-ImageNet run under runs/ (e.g. tinyimagenet_mobilenetv2_v1_acc)",
    )
    ap.add_argument(
        "--pretrain_ckpt",
        default="best.keras",
        help="Checkpoint filename inside runs/<pretrain_run>/checkpoints/ (default: best.keras)",
    )
    ap.add_argument(
        "--run_name",
        required=True,
        help="Output run name under runs/ for this Pokémon finetune",
    )

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--image_size", type=int, nargs=2, default=(224, 224),
                    help="Image size (H W), e.g. 224 224 or 240 240 or 260 260")

    ap.add_argument("--epochs_warmup", type=int, default=5)
    ap.add_argument("--epochs_finetune", type=int, default=30)
    ap.add_argument("--unfreeze_last", type=int, default=60)
    ap.add_argument("--lr_warmup", type=float, default=3e-4)
    ap.add_argument("--lr_finetune", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=3e-4)
    ap.add_argument("--early_stop_patience", type=int, default=6)
    ap.add_argument("--tensorboard", action="store_true")

    args = ap.parse_args()

    img_size = tuple(args.image_size)

    # ------------------------------------------------------------------
    # Paths: pretrain checkpoint + Pokémon run output
    # ------------------------------------------------------------------
    pretrain_dir = Path("runs") / args.pretrain_run / "checkpoints"
    base_model_path = pretrain_dir / args.pretrain_ckpt
    if not base_model_path.is_file():
        raise FileNotFoundError(
            f"Could not find base model: {base_model_path} "
            f"(check --pretrain_run / --pretrain_ckpt)."
        )

    out = Path("runs") / args.run_name
    ckpt = out / "checkpoints"
    out.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_ds, val_ds, num_classes, class_names = build_ds(
        args.pokemon_root,
        img_size=img_size,
        batch=args.batch_size,
    )

    # ------------------------------------------------------------------
    # Load base model and replace head
    # ------------------------------------------------------------------
    old = keras.models.load_model(base_model_path)
    features = old.layers[-1].input  # tensor before old Dense(200)

    logits = keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="pokemon_head",
    )(features)

    model = keras.Model(
        inputs=old.input,
        outputs=logits,
        name=f"{old.name}_pokemon",
    )

    # ------------------------------------------------------------------
    # Warm-up head
    # ------------------------------------------------------------------
    for l in model.layers:
        l.trainable = False
    model.layers[-1].trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(args.lr_warmup),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    warm_cbs = [
        keras.callbacks.CSVLogger(str(out / "warmup_history.csv")),
    ]
    if args.tensorboard:
        warm_cbs.append(
            keras.callbacks.TensorBoard(log_dir=str(out / "tb" / "warmup"))
        )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_warmup,
        callbacks=warm_cbs,
        verbose=2,
    )

    # ------------------------------------------------------------------
    # Fine-tune tail
    # ------------------------------------------------------------------
    for l in model.layers:
        l.trainable = False
    freeze_bn(model)
    unfreeze_tail(model, args.unfreeze_last)

    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=args.lr_finetune,
            weight_decay=args.weight_decay,
            clipnorm=1.0,
        ),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    ft_cbs = [
        keras.callbacks.ModelCheckpoint(
            str(ckpt / "best_by_acc.keras"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
        keras.callbacks.ModelCheckpoint(
            str(ckpt / "best_by_loss.keras"),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            mode="min",
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(out / "finetune_history.csv")),
    ]
    if args.tensorboard:
        ft_cbs.append(
            keras.callbacks.TensorBoard(log_dir=str(out / "tb" / "finetune"))
        )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_finetune,
        callbacks=ft_cbs,
        verbose=2,
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    model.save(ckpt / "final.keras")
    (out / "classes.json").write_text(json.dumps(class_names, indent=2))

    print(
        f"\nSaved:\n"
        f"  - {ckpt / 'best_by_acc.keras'}\n"
        f"  - {ckpt / 'best_by_loss.keras'}\n"
        f"  - {ckpt / 'final.keras'}\n"
        f"  - {out / 'classes.json'}\n"
    )


if __name__ == "__main__":
    main()
