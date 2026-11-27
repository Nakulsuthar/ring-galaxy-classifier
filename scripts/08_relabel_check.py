"""
Step 8: check the relabelled old val/test rings.

1. agreement between the original ring types (labels.csv) and the blind relabel
2. AUC of the old and new inner/outer models on the old TEST rings,
   using the original labels vs the relabelled ones (same galaxies)
-> tables/relabel_agreement.csv, tables/relabel_auc.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import config as C
from model import RingNet, Galaxies, EVAL_TF

rng = np.random.default_rng(C.SEED)
RELABEL = C.SRC / "tables" / "labels_relabel_old.csv"
MODELS = {"old model": C.DATA / "best_model_ringtype.pt", "new model": C.DATA / "best_model_ringtype2.pt"}


def auc_ci(y, s, n_boot=1000):
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            boots.append(roc_auc_score(y[i], s[i]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return roc_auc_score(y, s), lo, hi


# ---------------- 1. agreement ----------------
old = pd.read_csv(C.DATA / "splits_ringtype2.csv")
old = old[old["split"].isin(["val", "test"]) & (old["source"] != "q1_pred_r1")]
new = pd.read_csv(RELABEL)[["object_id", "label", "ring_type", "notes"]]

# new verdict in one column: inner / outer / both / unclear / maybe / bad / not ring
new["new"] = np.where(new["label"] == "ring", new["ring_type"].fillna("unclear"),
                      np.where(new["label"].isin(["maybe", "bad"]), new["label"], "not ring"))
d = old.merge(new[["object_id", "new"]], on="object_id", how="inner")
print(f"relabelled: {len(d)} of {len(old)} old val/test rings")

tab = pd.crosstab(d["ring_type"], d["new"], margins=True)
tab.to_csv(C.TABLES / "relabel_agreement.csv")
print("\noriginal type (rows) vs blind relabel (columns):\n", tab)

typed = d[d["new"].isin(["inner", "outer"])]
same = (typed["ring_type"] == typed["new"]).mean()
print(f"\nof {len(typed)} still given a clear inner/outer type: {same:.0%} unchanged, {1 - same:.0%} swapped")
print(f"no longer a clear inner/outer type (both/unclear/maybe/not ring/bad): "
      f"{(~d['new'].isin(['inner', 'outer'])).mean():.0%}")

# ---------------- 2. re-score the old TEST rings ----------------
t = d[d["split"] == "test"].reset_index(drop=True)
loader = DataLoader(Galaxies(t, EVAL_TF), batch_size=C.BATCH_SIZE, shuffle=False, num_workers=C.NUM_WORKERS)
for name, path in MODELS.items():
    ckpt = torch.load(path, map_location=C.DEVICE, weights_only=False)
    model = RingNet().to(C.DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    ps = []
    with torch.no_grad():
        for x, _, _ in loader:
            ps.append(torch.sigmoid(model(x.to(C.DEVICE))).cpu())
    t[name] = torch.cat(ps).numpy()

rows = []
y_old = t["y"].to_numpy()                                   # original labels, all test rings
clean = t["new"].isin(["inner", "outer"]).to_numpy()        # rings with a clear new type
y_new = (t.loc[clean, "new"] == "inner").astype(int).to_numpy()
for lab_name, y, m in [("original labels", y_old, np.ones(len(t), bool)),
                       ("relabelled (clear only)", y_new, clean)]:
    row = {"labels": lab_name, "n_inner": int((y == 1).sum()), "n_outer": int((y == 0).sum())}
    for col in list(MODELS) + ["area"]:
        a, lo, hi = auc_ci(y, t[col].to_numpy()[m])
        row[col] = f"{a:.3f} ({lo:.2f}-{hi:.2f})"
    rows.append(row)

res = pd.DataFrame(rows).rename(columns={"area": "area only"})
res.to_csv(C.TABLES / "relabel_auc.csv", index=False)
print("\nold TEST rings, original vs relabelled:\n", res.to_string(index=False))

# ---------------- 3. ROC: new model, original vs relabelled labels ----------------
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

p = t["new model"].to_numpy()
fig, ax = plt.subplots(figsize=(5.5, 5.5))

for y, m, colour, name in [(y_old, np.ones(len(t), bool), "tab:blue", "original labels"),
                           (y_new, clean, "tab:orange", "relabelled (clear only)")]:
    fpr, tpr, _ = roc_curve(y, p[m])
    ax.plot(fpr, tpr, color=colour, lw=2,
            label=f"{name} (AUC {roc_auc_score(y, p[m]):.3f})")

ax.plot([0, 1], [0, 1], ":", color="lightgrey")
ax.set_xlabel("False positive rate (outer called inner)")
ax.set_ylabel("True positive rate (inner found)")
ax.set_title(f"New (ringtype2) model on old test rings: original ({len(t)}) vs relabelled ({clean.sum()})", fontsize=10)
ax.legend(fontsize=8, loc="lower right")
plt.tight_layout()
plt.savefig(C.FIGURES / "relabel_roc.png", dpi=200)
print(f"\nsaved {C.FIGURES / 'relabel_roc.png'}")
