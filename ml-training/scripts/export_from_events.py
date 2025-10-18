from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
from pathlib import Path

# --- Paths ---
EVENTS_DIR = "checkpoints/tblogs"    # folder with events.out.tfevents.*
OUT_CSV    = "runs/offline_plots/exported_history.csv"
# -------------

ea = event_accumulator.EventAccumulator(EVENTS_DIR)
ea.Reload()

# Scalars you care about
tags = ["epoch_accuracy", "epoch_loss", "epoch_val_accuracy", "epoch_val_loss", "learning_rate"]

data = {}
for tag in tags:
    if tag in ea.Tags()["scalars"]:
        events = ea.Scalars(tag)
        data[tag] = [e.value for e in events]

df = pd.DataFrame(data)
Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False)

print(f"✔ Exported {len(df)} epochs to {OUT_CSV}")
