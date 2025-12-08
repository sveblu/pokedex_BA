#!/usr/bin/env python3
"""
Make detailed training curves + confusion matrix for ONE finetuned run.

Usage:
  python scripts/make_plots_from_history_detailed.py \
      --run_dir runs/pokemon_mobilenetv2_aug_v1_acc \
      --val_root data/pokemon/val
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix

AUTOTUNE = tf.data.AUTOTUNE


# ------------------------------------------------------------
# History loading
# ------------------------------------------------------------

def load_history(run_dir: Path) -> pd.DataFrame:
    """Load warmup_history.csv + finetune_history.csv into one DataFrame."""
    warm_path = run_dir / "warmup_history.csv"
    fine_path = run_dir / "finetune_history.csv"

    if not fine_path.exists():
        raise FileNotFoundError(f"Missing {fine_path}")

    df_fine = pd.read_csv(fine_path)

    if warm_path.exists() and warm_path.stat().st_size > 0:
        df_warm = pd.read_csv(warm_path)
        df_warm["epoch_global"] = df_warm["epoch"] + 1
        offset = int(df_warm["epoch_global"].max())
        df_fine["epoch_global"] = df_fine["epoch"] + 1 + offset
        history = pd.concat([df_warm, df_fine], ignore_index=True)
    else:
        df_fine["epoch_global"] = df_fine["epoch"] + 1
        history = df_fine

    return history


# ------------------------------------------------------------
# Training curves with detailed annotations
# ------------------------------------------------------------

def plot_training_curves(history: pd.DataFrame, out_path: Path) -> None:
    sns.set_style("whitegrid")

    epochs = history["epoch_global"].to_numpy()
    acc = history["accuracy"].to_numpy()
    val_acc = history["val_accuracy"].to_numpy()
    loss = history["loss"].to_numpy()
    val_loss = history["val_loss"].to_numpy()
    lr = history.get("lr", pd.Series([np.nan] * len(history))).to_numpy()

    best_acc_idx = int(np.argmax(val_acc))
    best_acc = float(val_acc[best_acc_idx])
    best_acc_epoch = int(epochs[best_acc_idx])

    best_loss_idx = int(np.argmin(val_loss))
    best_loss = float(val_loss[best_loss_idx])
    best_loss_epoch = int(epochs[best_loss_idx])

    fig, (ax_acc, ax_loss) = plt.subplots(
        1, 2, figsize=(14, 5), dpi=150, constrained_layout=True
    )

    # ---------- Accuracy ----------
    ax_acc.plot(epochs, acc, label="train acc", color="C0")
    ax_acc.plot(epochs, val_acc, label="val acc", color="C1")
    ax_acc.set_title("Accuracy over epochs")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0.0, 1.05)
    ax_acc.legend(loc="lower right")

    ax_acc.annotate(
        f"Best val acc\n(ep {best_acc_epoch}: {best_acc:.3f})",
        xy=(best_acc_epoch, best_acc),
        xytext=(best_acc_epoch + 1, best_acc - 0.1),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=9,
        ha="left",
        va="top",
    )

    for e, a_tr, a_val in zip(epochs, acc, val_acc):
        if e % 5 == 0:
            ax_acc.text(
                e,
                a_tr + 0.02,
                f"{a_tr:.2f}",
                color="C0",
                fontsize=7,
                ha="center",
            )
            ax_acc.text(
                e,
                a_val - 0.04,
                f"{a_val:.2f}",
                color="C1",
                fontsize=7,
                ha="center",
            )

    ax_acc.axvline(
        best_acc_epoch,
        color="gray",
        linestyle="--",
        linewidth=0.8,
        alpha=0.6,
    )

    # ---------- Loss ----------
    ax_loss.plot(epochs, loss, label="train loss", color="C0")
    ax_loss.plot(epochs, val_loss, label="val loss", color="C1")
    ax_loss.set_title("Loss over epochs")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(loc="upper right")

    ax_loss.annotate(
        f"Best val loss\n(ep {best_loss_epoch}: {best_loss:.3f})",
        xy=(best_loss_epoch, best_loss),
        xytext=(best_loss_epoch + 1, best_loss + 0.3),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=9,
        ha="left",
        va="bottom",
    )

    # optional LR on twin axis
    if not np.all(np.isnan(lr)):
        ax_lr = ax_loss.twinx()
        ax_lr.plot(epochs, lr, color="C2", alpha=0.4, label="lr")
        ax_lr.set_ylabel("Learning rate")
        ax_lr.set_yscale("log")
        ax_lr.tick_params(axis="y", labelsize=8)
        ax_lr.legend(loc="upper center", fontsize=7)

    fig.suptitle("Training curves (detailed)", fontsize=14)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"✔ saved {out_path}")


# ------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------

def infer_image_size_from_model(model: keras.Model) -> tuple[int, int]:
    """Infer H,W from model input shape."""
    # model.input_shape is usually (None, H, W, C)
    ishape = getattr(model, "input_shape", None)
    if ishape is None:
        # fall back to first layer input
        ishape = getattr(model.layers[0], "input_shape", None)

    if ishape is None or len(ishape) < 3:
        print("⚠ Could not infer image size from model, defaulting to 224x224")
        return (224, 224)

    h, w = ishape[1], ishape[2]
    if h is None or w is None:
        print("⚠ Model input shape has None dims, defaulting to 224x224")
        return (224, 224)

    return (int(h), int(w))


def plot_confusion_matrix(model_path: Path, val_root: Path, out_path: Path) -> None:
    print(f"→ Loading model: {model_path}")
    model = keras.models.load_model(model_path)

    img_size = infer_image_size_from_model(model)
    print(f"→ Using image size {img_size} for evaluation")

    print("→ Loading val dataset…")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_root,
        image_size=img_size,
        batch_size=64,
        shuffle=False,
    )
    class_names = val_ds.class_names
    num_classes = len(class_names)

    # Same normalization as in Pokémon finetune scripts: x / 255
    val_ds = val_ds.map(
        lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
        num_parallel_calls=AUTOTUNE,
    ).prefetch(2)

    y_true, y_pred = [], []
    for xb, yb in val_ds:
        probs = model.predict(xb, verbose=0)
        y_true.append(yb.numpy())
        y_pred.append(np.argmax(probs, axis=-1))

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    # Print sanity-check accuracy vs CSV val_accuracy
    overall_acc = (y_true == y_pred).mean()
    print(f"→ Overall val accuracy from confusion matrix: {overall_acc:.4f}")

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
    ).astype(float)
    cm /= cm.sum(axis=1, keepdims=True) + 1e-12  # normalize rows

    plt.figure(figsize=(10, 8), dpi=150)
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Fraction of class"},
    )
    plt.title("Detailed Confusion Matrix (percentages)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"✔ saved {out_path}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, help="Path to one run folder (with CSVs)")
    parser.add_argument("--val_root", required=True, help="Path to val/ directory")
    args = parser.parse_args()

    run = Path(args.run_dir)
    val_root = Path(args.val_root)

    history = load_history(run)
    plot_training_curves(history, run / "training_curves_detailed.png")

    model_path = run / "checkpoints" / "best_by_acc.keras"
    if model_path.exists():
        plot_confusion_matrix(model_path, val_root, run / "confusion_matrix_detailed.png")
    else:
        print(f"⚠ best_by_acc model not found at {model_path}, skipping confusion matrix.")


if __name__ == "__main__":
    main()
