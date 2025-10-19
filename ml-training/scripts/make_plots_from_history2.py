#!/usr/bin/env python3
# make_plots_from_history2.py

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Keras CSVLogger usually has these columns; keep only what we need
    keep = [c for c in df.columns if c in {"epoch","accuracy","loss","val_accuracy","val_loss","lr","learning_rate"}]
    df = df[keep]
    # unify lr column name if present
    if "learning_rate" in df.columns and "lr" not in df.columns:
        df = df.rename(columns={"learning_rate":"lr"})
    if "epoch" not in df.columns:
        # CSVLogger doesn't include epoch by default; fabricate 1..N
        df.insert(0, "epoch", range(1, len(df)+1))
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup_csv", default="runs/pokemon_v1/warmup_history.csv")
    ap.add_argument("--finetune_csv", default="runs/pokemon_v1/finetune_history.csv")
    ap.add_argument("--out_png", default="runs/pokemon_v1/deploy/training_curves.png")
    args = ap.parse_args()

    warm = Path(args.warmup_csv)
    fine = Path(args.finetune_csv)
    out  = Path(args.out_png)
    out.parent.mkdir(parents=True, exist_ok=True)

    dfs = []
    if warm.exists() and warm.stat().st_size > 0:
        w = load_csv(warm)
        dfs.append(w)
        offset = int(w["epoch"].iloc[-1])
    else:
        offset = 0

    f = load_csv(fine)
    f = f.copy()
    f["epoch"] = f["epoch"] + offset
    dfs.append(f)

    hist = pd.concat(dfs, ignore_index=True)

    # Plot
    fig = plt.figure(figsize=(9,6))
    gs = fig.add_gridspec(3, 1, height_ratios=[2,2,1], hspace=0.35)

    ax1 = fig.add_subplot(gs[0,0])
    ax1.plot(hist["epoch"], hist["loss"], label="train")
    if "val_loss" in hist:
        ax1.plot(hist["epoch"], hist["val_loss"], label="val")
    ax1.set_ylabel("loss"); ax1.legend()

    ax2 = fig.add_subplot(gs[1,0])
    if "accuracy" in hist:
        ax2.plot(hist["epoch"], hist["accuracy"], label="train")
    if "val_accuracy" in hist:
        ax2.plot(hist["epoch"], hist["val_accuracy"], label="val")
    ax2.set_ylabel("accuracy"); ax2.legend()

    ax3 = fig.add_subplot(gs[2,0])
    if "lr" in hist:
        ax3.plot(hist["epoch"], hist["lr"])
    ax3.set_ylabel("lr"); ax3.set_xlabel("epoch")

    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"✔ saved {out}")

if __name__ == "__main__":
    main()
