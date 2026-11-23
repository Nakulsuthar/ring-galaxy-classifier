"""
Step 4: predict P(ring) (and P(inner)) for random unlabelled Q1 galaxies.

- random sample of N galaxies with area >= MIN_AREA that are NOT in labels.csv
- ring model (best_model.pt)              -> p_ring
- ringtype model (best_model_ringtype.pt) -> p_inner, if it exists
  (if both models share the same encoder, it runs once: about half the time)
- saved every CHUNK galaxies, so it can be stopped (Ctrl+C) and restarted without losing work
-> data/q1_predictions.csv

usage: python3.13 scripts/04_predict.py [N]      (default 70000; keep N the same when resuming)
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import pyarrow.parquet as pq
from torch.utils.data import DataLoader
import config as C
from model import RingNet, Galaxies, EVAL_TF

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
CHUNK = 2000                                   # galaxies per save
OUT = C.DATA / "q1_predictions.csv"            # in data/, so not on GitHub
RING_MODEL = C.DATA / "best_model.pt"
TYPE_MODEL = C.DATA / "best_model_ringtype.pt"

# ---------------- 1. pick random unlabelled galaxies ----------------
cols = pq.read_schema(C.MORPH_PATH).names
area_col = next(c for c in ["segmentation_area", "seg_area", "area"] if c in cols)
tile_col = next(c for c in ["tile_index", "tile"] if c in cols)
cat = (pd.read_parquet(C.MORPH_PATH, columns=["object_id", tile_col, area_col])
         .rename(columns={tile_col: "tile", area_col: "area"})
         .drop_duplicates("object_id"))
cat["object_id"] = cat["object_id"].astype("int64")
cat = cat[cat["area"] >= C.MIN_AREA]

labelled = set(pd.read_csv(C.LABELS_PATH)["object_id"].astype("int64"))
cat = cat[~cat["object_id"].isin(labelled)]              # nothing we've already labelled
print(f"eligible: {len(cat)} galaxies (area >= {C.MIN_AREA}, not labelled)")

# same seed -> same sample every run, which is what makes restarting work
sample = cat.sample(n=min(N, len(cat)), random_state=C.SEED).reset_index(drop=True)
sample["path"] = [str(C.image_path(t, o)) for t, o in zip(sample["tile"], sample["object_id"])]
exists = sample["path"].map(lambda p: Path(p).exists())
print(f"sampled {len(sample)}, missing images: {(~exists).sum()}")
sample = sample[exists].reset_index(drop=True)
sample["y"] = 0                                          # dummy label, Galaxies expects a y column

# ---------------- 2. resume if stopped earlier ----------------
done = set()
if OUT.exists():
    done = set(pd.read_csv(OUT, usecols=["object_id"])["object_id"])
    print(f"resuming: {len(done)} already predicted")
todo = sample[~sample["object_id"].isin(done)].reset_index(drop=True)

# ---------------- 3. load models ----------------
def load(path):
    ckpt = torch.load(path, map_location=C.DEVICE, weights_only=False)
    m = RingNet().to(C.DEVICE)
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    print(f"loaded {path.name}: phase {ckpt['phase']} epoch {ckpt['epoch']}, val AUC {ckpt['val_auc']:.3f}")
    return m

ring_model = load(RING_MODEL)
type_model, shared = None, False
if TYPE_MODEL.exists():
    type_model = load(TYPE_MODEL)
    a, b = ring_model.encoder.state_dict(), type_model.encoder.state_dict()
    shared = all(torch.equal(a[k], b[k]) for k in a)     # same encoder weights?
    print(f"shared encoder: {shared}" + ("  (encoder runs once)" if shared else ""))

# ---------------- 4. predict in chunks ----------------
t0 = time.time()
for start in range(0, len(todo), CHUNK):
    chunk = todo.iloc[start:start + CHUNK]
    loader = DataLoader(Galaxies(chunk, EVAL_TF), batch_size=64,
                        shuffle=False, num_workers=C.NUM_WORKERS)
    pr, pi = [], []
    with torch.no_grad():
        for x, _, _ in loader:
            x = x.to(C.DEVICE)
            feats = ring_model.encoder(x)                                     # 1024 features
            pr.append(torch.sigmoid(ring_model.head(feats).squeeze(1)).cpu())
            if type_model is not None:
                logit = type_model.head(feats).squeeze(1) if shared else type_model(x)
                pi.append(torch.sigmoid(logit).cpu())

    out = chunk[["object_id", "tile", "area"]].copy()
    out["p_ring"] = torch.cat(pr).numpy()
    out["p_inner"] = torch.cat(pi).numpy() if pi else np.nan
    out.to_csv(OUT, mode="a", header=not OUT.exists(), index=False)       # append this chunk

    n = start + len(chunk)
    rate = n / (time.time() - t0)
    print(f"{len(done) + n}/{len(sample)} | {rate:.0f} galaxies/s | ~{(len(todo) - n) / rate / 60:.0f} min left")

# ---------------- 5. summary ----------------
pred = pd.read_csv(OUT)
print(f"\npredicted {len(pred)} galaxies -> {OUT}")
for t in [0.5, 0.8, 0.9, 0.95, 0.99]:
    k = (pred["p_ring"] > t).sum()
    print(f"  P(ring) > {t}: {k:6d}  ({k / len(pred):.1%})")