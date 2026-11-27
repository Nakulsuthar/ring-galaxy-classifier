"""
Step 6: label Q1 candidates by eye.

Shows each galaxy twice: the full cutout, and the centre zoomed x2 (helps with inner rings).
P(ring) is hidden on purpose, so it doesn't bias your label.
Every key press is saved immediately; close and rerun any time to resume.

rings:     i inner    o outer    b both    r ring, type unclear
not ring:  n other    s spiral   e edge-on    c smooth    x merger    l lens / S0
other:     m maybe    d bad cutout
           z undo     q quit
-> get_training_data/tables/labels_{ROUND}.csv
   (same columns as labels.csv, plus p_ring, p_inner, group)
"""
import sys, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import config as C

ROUND = "relabel_old"
CAND = C.DATA / f"candidates_{ROUND}.csv"
OUT = C.SRC / "tables" / f"labels_{ROUND}.csv"      # next to labels.csv, outside the public repo
SOURCE = "relabel_old"

#        key: (label,    ring_type, notes)
KEYS = {"i": ("ring",   "inner",   ""),
        "o": ("ring",   "outer",   ""),
        "b": ("ring",   "both",    ""),
        "r": ("ring",   "",        ""),
        "n": ("no",     "",        "other"),
        "s": ("spiral", "",        ""),
        "e": ("no",     "",        "edge-on"),
        "c": ("no",     "",        "smooth"),
        "x": ("no",     "",        "merger"),
        "l": ("no",     "",        "lens"),
        "m": ("maybe",  "",        ""),
        "d": ("bad",    "",        "")}
COLS = ["object_id", "tile", "label", "ring_type", "source", "notes", "labelled_on",
        "p_ring", "p_inner", "group"]

# turn off matplotlib's own keyboard shortcuts (s = save, q = quit, l = log scale, ...)
for k in list(plt.rcParams):
    if k.startswith("keymap."):
        plt.rcParams[k] = []

# ---------------- what's left to label ----------------
cand = pd.read_csv(CAND)
done = set(pd.read_csv(OUT)["object_id"]) if OUT.exists() else set()
todo = cand[~cand["object_id"].isin(done)].reset_index(drop=True)
print(f"{len(cand)} candidates | {len(done)} already labelled | {len(todo)} to go")
if todo.empty:
    sys.exit("all candidates labelled")

i = 0     # position in todo (= number labelled this session)

# ---------------- window ----------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 6))
fig.text(0.5, 0.03,
         "rings: i inner   o outer   b both   r ring (type?)     |     "
         "not: n other   s spiral   e edge-on   c smooth   x merger   l lens/S0\n"
         "m maybe   d bad cutout     |     z undo   q quit",
         ha="center", fontsize=9)


def show():
    img = Image.open(todo["path"].iat[i]).convert("RGB")
    w, h = img.size
    crop = img.crop((w // 4, h // 4, 3 * w // 4, 3 * h // 4))      # central half
    for ax in (ax1, ax2):
        ax.clear()
        ax.axis("off")
    ax1.imshow(img)
    ax1.set_title("full cutout")
    ax2.imshow(crop)
    ax2.set_title("centre x2")
    fig.suptitle(f"{len(done) + i + 1} / {len(cand)}     area {todo['area'].iat[i]:.0f} px")
    fig.canvas.draw_idle()


def on_key(event):
    global i
    k = event.key

    if k == "q":
        plt.close(fig)
        return

    if k == "z":                                         # undo the last label of this session
        if i == 0:
            print("nothing to undo this session")
            return
        lab = pd.read_csv(OUT)
        lab = lab[lab["object_id"] != todo["object_id"].iat[i - 1]]
        lab.to_csv(OUT, index=False)
        i -= 1
        print("undone")
        show()
        return

    if k not in KEYS:
        return

    label, rtype, notes = KEYS[k]
    row = {"object_id": int(todo["object_id"].iat[i]),   # .iat keeps the exact 18-digit id
           "tile": int(todo["tile"].iat[i]),
           "label": label, "ring_type": rtype, "source": SOURCE, "notes": notes,
           "labelled_on": datetime.date.today().isoformat(),
           "p_ring": todo["p_ring"].iat[i], "p_inner": todo["p_inner"].iat[i],
           "group": todo["group"].iat[i]}
    pd.DataFrame([row], columns=COLS).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
    print(f"{len(done) + i + 1}: {label} {rtype} {notes}".strip())

    i += 1
    if i >= len(todo):
        print("all candidates labelled")
        plt.close(fig)
        return
    show()


fig.canvas.mpl_connect("key_press_event", on_key)
show()
plt.show()

# ---------------- summary ----------------
if OUT.exists():
    lab = pd.read_csv(OUT)
    print(f"\nlabelled so far: {len(lab)}")
    print(pd.crosstab(lab["group"], lab["label"]))
    for g in ["top", "band"]:
        s = lab[lab["group"] == g]
        if len(s):
            print(f"precision ({g}): {(s['label'] == 'ring').mean():.2f}  (rings / labelled, n={len(s)})")
    print("\nring types:", lab.loc[lab["label"] == "ring", "ring_type"].fillna("unclear").value_counts().to_dict())