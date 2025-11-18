import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras

plt.style.use("seaborn-v0_8")

def load_history(run_dir: Path):
    warm = run_dir / "warmup_history.csv"
    fine = run_dir / "finetune_history.csv"

    df_list = []
    if warm.exists() and warm.stat().st_size > 0:
        df_warm = pd.read_csv(warm)
        df_warm["epoch"] = df_warm.index + 1
        df_list.append(df_warm)

    df_fine = pd.read_csv(fine)
    df_fine["epoch"] = range(
        (df_list[-1]["epoch"].iloc[-1] + 1) if df_list else 1,
        (df_list[-1]["epoch"].iloc[-1] + 1 + len(df_fine)) if df_list else len(df_fine) + 1
    )
    df_list.append(df_fine)

    history = pd.concat(df_list, ignore_index=True)
    return history


def plot_training_curves(history: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax[0].plot(history["epoch"], history["accuracy"], label="train acc")
    ax[0].plot(history["epoch"], history["val_accuracy"], label="val acc")
    ax[0].set_title("Accuracy over epochs")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend()

    # Annotate best val accuracy
    best_epoch = history["val_accuracy"].idxmax()
    best_val = history["val_accuracy"].max()
    ax[0].annotate(
        f"Best: {best_val:.3f}",
        xy=(history["epoch"][best_epoch], best_val),
        xytext=(history["epoch"][best_epoch], best_val + 0.03),
        arrowprops=dict(arrowstyle="->"),
        fontsize=10
    )

    # Loss
    ax[1].plot(history["epoch"], history["loss"], label="train loss")
    ax[1].plot(history["epoch"], history["val_loss"], label="val loss")
    ax[1].set_title("Loss over epochs")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    ax[1].legend()

    # Annotate lowest val loss
    best_loss_epoch = history["val_loss"].idxmin()
    best_loss = history["val_loss"].min()
    ax[1].annotate(
        f"Best: {best_loss:.3f}",
        xy=(history["epoch"][best_loss_epoch], best_loss),
        xytext=(history["epoch"][best_loss_epoch], best_loss + 0.3),
        arrowprops=dict(arrowstyle="->"),
        fontsize=10
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"✔ saved {out_path}")


def plot_confusion_matrix(model_path: Path, val_root: Path, out_path: Path):
    print("→ Loading model…")
    m = keras.models.load_model(model_path)

    print("→ Loading val dataset…")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_root,
        image_size=(224, 224),
        batch_size=64,
        shuffle=False,
    )
    class_names = val_ds.class_names

    val_ds = val_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))

    y_true = []
    y_pred = []

    for x, y in val_ds:
        logits = m.predict(x, verbose=0)
        preds = np.argmax(logits, axis=1)
        y_true.extend(y.numpy())
        y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(15, 12))

    sns.heatmap(
        cm_percent,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True
    )

    ax.set_title("Detailed Confusion Matrix (percentages)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"✔ saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--val_root", required=True)
    args = parser.parse_args()

    run = Path(args.run_dir)
    val_root = Path(args.val_root)

    model_path = run / "checkpoints" / "best_by_acc.keras"

    history = load_history(run)
    plot_training_curves(history, run / "training_curves_detailed.png")
    plot_confusion_matrix(model_path, val_root, run / "confusion_matrix_detailed.png")


if __name__ == "__main__":
    main()
