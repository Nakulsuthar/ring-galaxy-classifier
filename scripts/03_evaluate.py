"""
step 3: evaluate the best model on the untouched set

- AUC overall ( with bootstrapping 95% interval) and against an area only baseline
- AUC per bin size
- AUC for all rings vs each type of negative
- AUC per source 
- precision / recall at P > 0.5, and expected precision when rings are rare
-> tables/test_metrics.csv, data/test_predicitions.csv, figures/test_roc.png, figures/test_mistakes.png"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
import config as C
from model import RingNet, Galaxies, EVAL_TF

rng = np.random.default_rng(C.SEED)

def auc_ci(y, s, n_boot=1000):
    """AUC and its 95% interval from bootstrap resampling (small samples are noisy)."""
    y, s = np.asarray(y), np.asarray(s)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))          # resample galaxies with replacement
        if y[i].min() != y[i].max():                  # need both classes to compute AUC
            boots.append(roc_auc_score(y[i], s[i]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return roc_auc_score(y, s), lo, hi   


# load the best model 
# weights_only = False, our own file and it contains plain numbers besides the weights
ckpt = torch.load(C.DATA / "best_model.pt", map_location=C.DEVICE, weights_only=False)
model = RingNet().to(C.DEVICE)
model.load_state_dict(ckpt["state_dict"])
model.eval()
print(f"model: phase {ckpt['phase']} epoch {ckpt['epoch']}, val AUC {ckpt['val_auc']:.3f}")

# predict on the test set 
df = pd.read_csv(C.DATA / "splits.csv")
test = df[df["split"] == "test"].reset_index(drop=True)
loader = DataLoader(Galaxies(test, EVAL_TF), batch_size=C.BATCH_SIZE, shuffle=False, num_workers=C.NUM_WORKERS)
ps = []
with torch.no_grad():
    for x, _,_ in loader:
        ps.append(torch.sigmoid(model(x.to(C.DEVICE))).cpu())
test["p_ring"] = torch.cat(ps).numpy()

# each type of negative set 
test["neg_type"] = np.where(test["label"] == "spiral", "spiral",
                            test["notes"].fillna("unspecified"))
test.loc[test["y"] == 1, "neg_type"] = ""
test.to_csv(C.DATA / "test_predictions.csv", index=False)      # in data/, so not on GitHub

y, p = test["y"].to_numpy(), test["p_ring"].to_numpy()
ring, neg = y == 1, y == 0
print(f"test set: {ring.sum()} rings, {neg.sum()} non-rings")

# metrics
rows = []

def add(group, name, positives, negatives):
    """ AUC of the rings in positives against the non rings in negatives"""
    m = positives | negatives 
    row = {"group": group, "subset": name, "n_ring": int(positives.sum()), "n_neg": int(negatives.sum())}
    if positives.any() and negatives.any():
        row["auc"], row["lo95"], row["hi95"] = auc_ci(y[m], p[m])
    rows.append(row)

add("all", "model", ring, neg)

# baseline: rank galaxies by size only
a, lo, hi = auc_ci(y, test["area"].to_numpy())
rows.append({"group": "all", "subset": "area only", "n_ring": int(ring.sum()),
             "n_neg": int(neg.sum()), "auc": a, "lo95": lo, "hi95": hi})

for b in C.SIZE_NAMES:
    m = (test["size_bin"] == b).to_numpy()
    add("size bin", b, ring & m, neg & m)

for t in ["spiral", "other", "edge-on", "smooth", "merger", "unspecified"]:
    add("rings vs", t, ring, neg & (test["neg_type"] == t).to_numpy())

for s in sorted(test["source"].unique()):
    m = (test["source"] == s).to_numpy()
    add("source", s, ring & m, neg & m)

metrics = pd.DataFrame(rows).round(3)
metrics.to_csv(C.TABLES / "test_metrics.csv", index=False)
print("\n", metrics.to_string(index=False))

# precision / recall at P > 0.5
pred = p >= 0.5
tp, fp, fn = (pred & ring).sum(), (pred & neg).sum(), (~pred & ring).sum()
recall, fpr = tp / (tp + fn), fp / neg.sum()
print(f"\nat P > 0.5: precision {tp/(tp+fp):.3f}, recall {recall:.3f}, "
      f"false positive rate {fpr:.3f}")

# what precision would be if rings were rare, as in a real survey
for f in [0.03, 0.05, 0.10]:
    prec = recall * f / (recall * f + fpr * (1 - f))
    print(f"  if {f:.0%} of galaxies were rings: expected precision {prec:.2f}")

# ---------------- 4. ROC curves ----------------
fig, ax = plt.subplots(figsize=(5.5, 5.5))
fpr_, tpr_, _ = roc_curve(y, p)
ax.plot(fpr_, tpr_, "k-", lw=2, label=f"all (AUC {roc_auc_score(y, p):.3f})")
for b, c in zip(C.SIZE_NAMES, ["tab:blue", "tab:orange", "tab:red"]):
    m = (test["size_bin"] == b).to_numpy()
    fpr_, tpr_, _ = roc_curve(y[m], p[m])
    ax.plot(fpr_, tpr_, color=c, label=f"{b} px (AUC {roc_auc_score(y[m], p[m]):.3f})")
m = ring | (test["neg_type"] == "spiral").to_numpy()
fpr_, tpr_, _ = roc_curve(y[m], p[m])
ax.plot(fpr_, tpr_, color="purple", ls="-.", label=f"rings vs spirals (AUC {roc_auc_score(y[m], p[m]):.3f})")
fpr_, tpr_, _ = roc_curve(y, test["area"])
ax.plot(fpr_, tpr_, color="grey", ls="--", label=f"area only (AUC {roc_auc_score(y, test['area']):.3f})")
ax.plot([0, 1], [0, 1], ":", color="lightgrey")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate (completeness)")
ax.legend(fontsize=8, loc="lower right")
plt.tight_layout()
plt.savefig(C.FIGURES / "test_roc.png", dpi=200)

# ---------------- 5. worst mistakes ----------------
fps = test[neg].sort_values("p_ring", ascending=False).head(12)    # non-rings scored highest
fns = test[ring].sort_values("p_ring").head(12)                    # rings scored lowest

fig, axes = plt.subplots(4, 6, figsize=(12, 8.8))
for ax, (_, r) in zip(axes[:2].ravel(), fps.iterrows()):
    ax.imshow(Image.open(r["path"]))
    ax.set_title(f"{r['neg_type']}  p={r['p_ring']:.2f}\nA={r['area']:.0f}px", fontsize=7)
for ax, (_, r) in zip(axes[2:].ravel(), fns.iterrows()):
    rt = r["ring_type"] if isinstance(r["ring_type"], str) else "ring"
    ax.imshow(Image.open(r["path"]))
    ax.set_title(f"{rt}  p={r['p_ring']:.2f}\nA={r['area']:.0f}px", fontsize=7)
for ax in axes.ravel():
    ax.axis("off")
fig.text(0.005, 0.73, "False positives\n(non-rings, highest P)", rotation=90, va="center", fontsize=9)
fig.text(0.005, 0.27, "False negatives\n(rings, lowest P)", rotation=90, va="center", fontsize=9)
plt.tight_layout(rect=(0.02, 0, 1, 1))
plt.savefig(C.FIGURES / "test_mistakes.png", dpi=150)

print(f"\nsaved {C.TABLES / 'test_metrics.csv'}, {C.FIGURES / 'test_roc.png'}, "
      f"{C.FIGURES / 'test_mistakes.png'}")
