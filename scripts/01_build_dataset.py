""" STEP1: Build the labelled dataset and split it into train val test 

labels.csv + morphology catalogue (segmentation area) + image paths 
-> keep ring (1) and no/spiral (0)
-> drop galaxies with area < MIN_AREA
-> assign a bin size 
-> split 70/15/15, stratified by label and size bin
-> data/splits.csv

TASK switch in config.py
TASK = "ring":      ring (1) vs no/spiral (0), as above
TASK = "ringtype":  inner (1) vs outer (0), typed rings only; reuses the ring run's split
TASK = "ringtype2": as ringtype, PLUS the typed rings from the Q1 labelling round,
                    split 60/10/30 into train / val / test_q1 (test_q1 never trained on)
-> data/splits{TAG}.csv, tables/split_summary{TAG}.csv
"""

import sys 
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for importing config.py

import pandas as pd 
import pyarrow.parquet as pq 
from sklearn.model_selection import train_test_split
import config as C

print(f"TASK = {C.TASK}")

# seg area (read once, used for old and new labels)
cols = pq.read_schema(C.MORPH_PATH).names
area_col = next((c for c in ["segmentation_area", "seg_area", "area"] if c in cols), None)
if area_col is None:
    sys.exit(f"No area column in morphology catalogue. Columns are: {cols}")
morph = (pd.read_parquet(C.MORPH_PATH, columns=["object_id", area_col]).rename(columns={area_col: "area"}).drop_duplicates("object_id"))
morph["object_id"] = morph["object_id"].astype("int64")


def add_area_and_cuts(lab, name):
    """NEW: area + image path + cuts, as a function so old and Q1 labels get identical treatment"""
    df = lab.merge(morph, on="object_id", how="left")

    # cuts 
    n0 = len(df)
    df = df[df["area"].notna()].copy()        # for no area 
    n1 = len(df)

    df["path"] = [str(C.image_path(t, o)) for t, o in zip(df["tile"], df["object_id"])]
    df = df[df["path"].map(lambda p: Path(p).exists())].copy()                  # for no image 
    n2 = len(df)

    df = df[df["area"] >= C.MIN_AREA].copy()
    n3 = len(df)          # for seg area cut 

    print(f"{name}: no area: -{n0-n1} | no image -{n1-n2} | area < {C.MIN_AREA}: -{n2-n3} | left: {n3}")
    df["size_bin"] = pd.cut(df["area"], bins=C.SIZE_BINS, labels=C.SIZE_NAMES, right=False).astype(str)
    return df


def split_two_step(d, strat, first_test, second_test):
    """NEW: stratified split; falls back to stratifying by class only if a box is too small"""
    try:
        a, rest = train_test_split(d, test_size=first_test, stratify=strat, random_state=C.SEED)
        b, c = train_test_split(rest, test_size=second_test, stratify=strat.loc[rest.index], random_state=C.SEED)
    except ValueError:
        print("  (some size-bin boxes too small: stratifying by class only)")
        a, rest = train_test_split(d, test_size=first_test, stratify=d["cls"], random_state=C.SEED)
        b, c = train_test_split(rest, test_size=second_test, stratify=rest["cls"], random_state=C.SEED)
    return a, b, c


# labels 
lab = pd.read_csv(C.LABELS_PATH)
if C.TASK == "ring":
    lab = lab[lab["label"].isin(C.POSITIVE | C.NEGATIVE)].copy()
    lab["y"] = lab["label"].isin(C.POSITIVE).astype(int)
    lab["cls"] = lab["label"]                                   # ring / no / spiral, used for stratifying
elif C.TASK in ("ringtype", "ringtype2"):                       # typed rings only
    lab = lab[(lab["label"] == "ring") & lab["ring_type"].isin(C.RING_TYPES)].copy()
    lab["y"] = (lab["ring_type"] == "inner").astype(int)       # inner = 1, outer = 0
    lab["cls"] = lab["ring_type"]                               # inner / outer
else:
    sys.exit(f"unknown TASK: {C.TASK}")
print(f"labels kept: {len(lab)} ({lab['cls'].value_counts().to_dict()})")

df = add_area_and_cuts(lab, "labels.csv")

# stratfied split
# this basically makes 9 boxes like this 
#  seg are       700–1500 px   1500–3000 px   >3000 px
#  ring            box 1          box 2         box 3
#  no              box 4          box 5         box 6
#  spiral          box 7          box 8         box 9

# good for not having messed up training and making sure uniformity is consistent

if C.TASK == "ring":
    strat = df["cls"] + "_" + df["size_bin"]      # ring no spiral x size bin 
    f = C.SPLIT
    train, val, test = split_two_step(df, strat, 1 - f["train"], f["test"] / (f["val"] + f["test"]))
    df["split"] = "train"
    df.loc[val.index, "split"] = "val"
    df.loc[test.index, "split"] = "test"
else:
    # ringtype reuses the ring run's split instead of making a new one.
    # the ringtype model starts from the ring model, which trained on 70% of all rings,
    # so a new split could put rings it already saw into the test set.
    ring_split = pd.read_csv(C.RING_SPLITS_PATH, usecols=["object_id", "split"])
    df = df.merge(ring_split, on="object_id", how="left")
    if df["split"].isna().any():
        sys.exit(f"{df['split'].isna().sum()} rings missing from {C.RING_SPLITS_PATH}; run once with TASK='ring' first")

# NEW: add the typed rings from the Q1 labelling round
if C.TASK == "ringtype2":
    q1 = pd.read_csv(C.Q1_LABELS_PATH)
    q1 = q1[(q1["label"] == "ring") & q1["ring_type"].isin(C.RING_TYPES)].copy()   # 'both' and unclear left out
    q1["y"] = (q1["ring_type"] == "inner").astype(int)
    q1["cls"] = q1["ring_type"]
    print(f"Q1 labels kept: {len(q1)} ({q1['cls'].value_counts().to_dict()})")
    q1 = add_area_and_cuts(q1, "Q1 labels")

    # 60 / 10 / 30 into train / val / test_q1, stratified by inner/outer x size bin.
    # test_q1 is never trained on: it's the fair test of whether the new labels helped.
    f = C.Q1_SPLIT
    strat = q1["cls"] + "_" + q1["size_bin"]
    tr, va, te = split_two_step(q1, strat, 1 - f["train"], f["test_q1"] / (f["val"] + f["test_q1"]))
    q1["split"] = "train"
    q1.loc[va.index, "split"] = "val"
    q1.loc[te.index, "split"] = "test_q1"
    df = pd.concat([df, q1], ignore_index=True)

# save and summary 
keep = ["object_id", "tile", "label", "y", "ring_type", "notes", "source", "area", "size_bin", "path", "split"]
df[keep].to_csv(C.SPLITS_PATH, index=False)

summary = pd.crosstab([df["split"], df["cls"]], df["size_bin"], margins=True)
summary = summary[[c for c in C.SIZE_NAMES + ["All"] if c in summary.columns]]
summary.to_csv(C.TABLES / f"split_summary{C.TAG}.csv")

print("\n", summary)
print(f"\nfraction y=1 per split ({'ring' if C.TASK == 'ring' else 'inner'}):\n", df.groupby("split")["y"].mean().round(3))
print("\nmedian area by class:\n", df.groupby("cls")["area"].median().round(0))
print("\nclass by source (watch for sources with only one class):\n", pd.crosstab(df["source"], df["cls"]))
print(f"\nsaved {C.SPLITS_PATH}")

# AUC = the probability that a randomly chosen ring gets a higher P(ring) score than a randomly chosen non-ring.