#!/usr/bin/env python3
# train_finetune_pokemon.py

import argparse, json
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

AUTOTUNE = tf.data.AUTOTUNE

def build_ds(root, img_size=(224,224), batch=32):
    root = Path(root)
    tr = tf.keras.utils.image_dataset_from_directory(
        root / "train", image_size=img_size, batch_size=batch, shuffle=True)
    va = tf.keras.utils.image_dataset_from_directory(
        root / "val", image_size=img_size, batch_size=batch, shuffle=False)

    aug = keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.08),
        keras.layers.RandomZoom(0.15),
        keras.layers.RandomTranslation(0.05, 0.05),
        keras.layers.RandomContrast(0.1),
    ], name="aug")

    norm = lambda x,y: (tf.cast(x, tf.float32)/255.0, y)
    tr = tr.map(lambda x,y: (aug(x, training=True), y), AUTOTUNE).map(norm, AUTOTUNE).prefetch(2)
    va = va.map(norm, AUTOTUNE).prefetch(2)

    return tr, va, len(tr.class_names), tr.class_names

def freeze_bn(model):
    for l in model.layers:
        if isinstance(l, keras.layers.BatchNormalization):
            l.trainable = False

def unfreeze_tail(model, n_layers):
    unfrozen = 0
    for l in reversed(model.layers[:-1]):  # skip new head
        if unfrozen >= n_layers: break
        if not isinstance(l, keras.layers.BatchNormalization):
            l.trainable = True
        unfrozen += 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pokemon_root", required=True)
    ap.add_argument("--base_model", required=True, help=".keras from Tiny-ImageNet pretraining")
    ap.add_argument("--run_name", default="pokemon_v1")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs_warmup", type=int, default=5)
    ap.add_argument("--epochs_finetune", type=int, default=30)
    ap.add_argument("--unfreeze_last", type=int, default=60)
    ap.add_argument("--lr_warmup", type=float, default=3e-4)
    ap.add_argument("--lr_finetune", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=3e-4)
    ap.add_argument("--early_stop_patience", type=int, default=6)
    ap.add_argument("--tensorboard", action="store_true")
    args = ap.parse_args()

    out = Path("runs")/args.run_name
    ckpt = out/"checkpoints"
    out.mkdir(parents=True, exist_ok=True); ckpt.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, num_classes, class_names = build_ds(args.pokemon_root, batch=args.batch_size)

    old = keras.models.load_model(args.base_model)
    features = old.layers[-1].input            # tensor before old Dense(200)
    logits = keras.layers.Dense(num_classes, activation="softmax", name="pokemon_head")(features)
    model = keras.Model(inputs=old.input, outputs=logits, name="mobilenetv2_pokemon")

    # warm-up head
    for l in model.layers: l.trainable = False
    model.layers[-1].trainable = True
    model.compile(optimizer=keras.optimizers.Adam(args.lr_warmup),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
    warm_cbs = [keras.callbacks.CSVLogger(str(out/"warmup_history.csv"))]
    if args.tensorboard: warm_cbs += [keras.callbacks.TensorBoard(log_dir=str(out/"tb"/"warmup"))]
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_warmup, callbacks=warm_cbs, verbose=2)

    # fine-tune tail
    for l in model.layers: l.trainable = False
    freeze_bn(model); unfreeze_tail(model, args.unfreeze_last)
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=args.lr_finetune, weight_decay=args.weight_decay, clipnorm=1.0),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    ft_cbs = [
        keras.callbacks.ModelCheckpoint(str(ckpt/"best_by_acc.keras"), monitor="val_accuracy", mode="max", save_best_only=True),
        keras.callbacks.ModelCheckpoint(str(ckpt/"best_by_loss.keras"), monitor="val_loss", mode="min", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, mode="min", verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=args.early_stop_patience, restore_best_weights=True, mode="max", verbose=1),
        keras.callbacks.CSVLogger(str(out/"finetune_history.csv")),
    ]
    if args.tensorboard: ft_cbs += [keras.callbacks.TensorBoard(log_dir=str(out/"tb"/"finetune"))]
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_finetune, callbacks=ft_cbs, verbose=2)

    model.save(ckpt/"final.keras")
    (out/"classes.json").write_text(json.dumps(class_names, indent=2))
    print(f"\nSaved:\n  - {ckpt/'best_by_acc.keras'}\n  - {ckpt/'best_by_loss.keras'}\n  - {ckpt/'final.keras'}\n  - {out/'classes.json'}")

if __name__ == "__main__":
    main()
