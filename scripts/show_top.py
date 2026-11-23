"""Gallery of Q1 galaxies above a P(ring) threshold.
usage: python3.13 scripts/show_top.py [threshold]   (default 0.99)"""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import config as C

T = float(sys.argv[1]) if len(sys.argv) > 1 else 0.99
p = pd.read_csv(C.DATA / "q1_predictions.csv")
top = p[p["p_ring"] > T].sort_values("p_ring", ascending=False).reset_index(drop=True)
print(f"{len(top)} galaxies with P(ring) > {T}")

ncol = 8
nrow = math.ceil(len(top) / ncol)
fig, axes = plt.subplots(nrow, ncol, figsize=(2 * ncol, 2.2 * nrow), squeeze=False)
for ax in axes.ravel():
    ax.axis("off")

# itertuples keeps each column's own type, so the 18-digit object_id stays an exact integer
# (iterrows would turn it into a float and round it, giving a wrong file path)
for i, (ax, r) in enumerate(zip(axes.ravel(), top.itertuples())):
    ax.imshow(Image.open(C.image_path(r.tile, r.object_id)))
    ax.set_title(f"#{i}  p={r.p_ring:.3f}\nP(in)={r.p_inner:.2f}  A={r.area:.0f}", fontsize=7)

plt.tight_layout()
out = C.FIGURES / f"q1_top_p{str(T).replace('.', '')}.png"
plt.savefig(out, dpi=130)
print("saved", out)