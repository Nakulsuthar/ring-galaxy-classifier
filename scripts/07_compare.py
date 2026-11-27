"""
Step 7: compare the old and new inner/outer models on the same test rings.

- test:    held-out rings from labels.csv (as before)
- test_q1: held-out rings from the Q1 labelling round (never seen by either model)
AUC with bootstrap 95% intervals, overall and per size bin, plus the area-only baseline
-> tables/compare_ringtype.csv, figures/compare_roc_q1.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
import config as C
from model import RingNet, Galaxies, EVAL_TF

rng = np.random.default_rng(C.SEED)
MODELS = {"old": C.DATA / "best_model_ringtype.pt",       # trained on labels.csv only
          "new": C.DATA / "best_model_ringtype2.pt"}      # + Q1 labels
SPLITS = C.DATA / "splits_ringtype2.csv"


def auc_ci(y, s, n_boot=1000):
    """AUC and its 95% interval from bootstrap resampling."""
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            boots.append(roc_auc_score(y[i], s[i]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return roc_auc_score(y, s), lo, hi


# ---------------- predict both test sets with both models ----------------
df = pd.read_csv(SPLITS)
test = df[df["split"].isin(["test", "test_q1"])].reset_index(drop=True)
loader = DataLoader(Galaxies(test, EVAL_TF), batch_size=C.BATCH_SIZE,
                    shuffle=False, num_workers=C.NUM_WORKERS)

for name, path in MODELS.items():
    ckpt = torch.load(path, map_location=C.DEVICE, weights_only=False)
    model = RingNet().to(C.DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"{name}: {path.name} (phase {ckpt['phase']} epoch {ckpt['epoch']}, val AUC {ckpt['val_auc']:.3f})")
    ps = []
    with torch.no_grad():
        for x, _, _ in loader:
            ps.append(torch.sigmoid(model(x.to(C.DEVICE))).cpu())
    test[name] = torch.cat(ps).numpy()

# ---------------- AUC table ----------------
rows = []
for split in ["test", "test_q1"]:
    t = test[test["split"] == split]
    y = t["y"].to_numpy()
    subsets = [("all", np.ones(len(t), bool))] + \
              [(f"{b} px", (t["size_bin"] == b).to_numpy()) for b in C.SIZE_NAMES]
    for sub, m in subsets:
        if len(np.unique(y[m])) < 2:
            continue
        row = {"split": split, "subset": sub,
               "n_inner": int((y[m] == 1).sum()), "n_outer": int((y[m] == 0).sum())}
        for col in list(MODELS) + ["area"]:
            a, lo, hi = auc_ci(y[m], t[col].to_numpy()[m])
            row[col] = f"{a:.3f} ({lo:.2f}-{hi:.2f})"
        rows.append(row)

res = pd.DataFrame(rows).rename(columns={"area": "area only"})
res.to_csv(C.TABLES / "compare_ringtype.csv", index=False)
print("\n", res.to_string(index=False))

# ---------------- ROC on the Q1 test rings ----------------
t = test[test["split"] == "test_q1"]
y = t["y"].to_numpy()
fig, ax = plt.subplots(figsize=(5.5, 5.5))
for col, style in [("old", "b-"), ("new", "k-"), ("area", "--")]:
    fpr, tpr, _ = roc_curve(y, t[col])
    label = "area only" if col == "area" else f"{col} model"
    ax.plot(fpr, tpr, style, color="grey" if col == "area" else None, lw=2 if col == "new" else 1.5,
            label=f"{label} (AUC {roc_auc_score(y, t[col]):.3f})")
ax.plot([0, 1], [0, 1], ":", color="lightgrey")
ax.set_xlabel("False positive rate (outer called inner)")
ax.set_ylabel("True positive rate (inner found)")
ax.set_title(f"Held-out Q1 rings ({(y == 1).sum()} inner, {(y == 0).sum()} outer)", fontsize=10)
ax.legend(fontsize=8, loc="lower right")
plt.tight_layout()
plt.savefig(C.FIGURES / "compare_roc_q1.png", dpi=200)
print(f"\nsaved {C.TABLES / 'compare_ringtype.csv'}, {C.FIGURES / 'compare_roc_q1.png'}")