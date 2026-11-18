"""Learning curves from tables/train_log.csv -> figures/training_curves.png"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import matplotlib.pyplot as plt
import config as C

log = pd.read_csv(C.TABLES / "train_log.csv")
x = range(1, len(log) + 1)                       # epochs counted across both phases
p2_start = (log["phase"] == 1).sum() + 0.5       # where phase 2 begins
best = log["val_auc"].idxmax() + 1               # saved model

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))

a1.plot(x, log["val_auc"], "k-o", lw=2, ms=4, label="Overall")
for b, c in zip(C.SIZE_NAMES, ["tab:blue", "tab:orange", "tab:red"]):
    a1.plot(x, log[f"auc_{b}"], "-o", color=c, ms=3, label=f"{b} px")
a1.set_ylabel("Validation AUC")

a2.plot(x, log["train_loss"], "-o", ms=3, label="Train (weighted)")
a2.plot(x, log["val_loss"], "-o", ms=3, label="Validation")
a2.set_ylabel("BCE loss")

for a in (a1, a2):
    a.axvline(p2_start, color="grey", ls="--", lw=1)
    a.axvline(best, color="green", ls=":", lw=1.5)
    a.set_xlabel("Epoch (phase 1: 1-5, phase 2: 6+)")
    a.legend(fontsize=8)
a1.text(p2_start + 0.2, a1.get_ylim()[0] + 0.003, "phase 2", color="grey", fontsize=8)

plt.tight_layout()
plt.savefig(C.FIGURES / "training_curves.png", dpi=200)
print("saved", C.FIGURES / "training_curves.png")