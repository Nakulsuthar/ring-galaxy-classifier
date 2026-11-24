"""
Step 5: choose Q1 candidates to label.

- the TOP_N galaxies with the highest P(ring)
- plus BAND random galaxies from just below them (0.5 < P < the top-N cut), so the
  labelled set isn't only the easiest, clearest rings
- shuffled
-> data/candidates_{ROUND}.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import config as C

ROUND = "q1_r1"
TOP_N = 1000         # highest scoring galaxies
BAND = 300           # random galaxies from just below the top N cut

p = pd.read_csv(C.DATA / "q1_predictions.csv")
p = p.sort_values("p_ring", ascending=False).reset_index(drop=True)

top = p.head(TOP_N).assign(group="top")
cut = top["p_ring"].min()
pool = p[(p["p_ring"] > 0.5) & (p["p_ring"] < cut)]
band = pool.sample(n=min(BAND, len(pool)), random_state=C.SEED).assign(group="band")

cand = pd.concat([top, band]).sample(frac=1, random_state=C.SEED).reset_index(drop=True)
cand["path"] = [str(C.image_path(t, o)) for t, o in zip(cand["tile"], cand["object_id"])]
cand.to_csv(C.DATA / f"candidates_{ROUND}.csv", index=False)

print(f"top {len(top)}: P(ring) > {cut:.3f}")
print(f"band {len(band)}: 0.5 < P(ring) < {cut:.3f} (from {len(pool)} galaxies)")
print(f"saved {len(cand)} candidates -> {C.DATA / f'candidates_{ROUND}.csv'}")