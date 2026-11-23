""" STEP1: Build the labelled dataset and split it into train val test 

labels.csv + morphology catalogue (segmentation area) + image paths 
-> keep ring (1) and no/spiral (0)
-> drop galaxies with area < MIN_AREA
-> assign a bin size 
-> split 70/15/15, stratified by label and size bin
-> data/splits.csv

NEW: TASK switch in config.py
TASK = "ring":     ring (1) vs no/spiral (0), as above
TASK = "ringtype": inner (1) vs outer (0), typed rings only; reuses the ring run's split
                   so test rings stay test rings (no leakage from the ring model)
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


# labels 
lab = pd.read_csv(C.LABELS_PATH)
if C.TASK == "ring":
    lab = lab[lab["label"].isin(C.POSITIVE | C.NEGATIVE)].copy()
    lab["y"] = lab["label"].isin(C.POSITIVE).astype(int)
    lab["cls"] = lab["label"]                                   # ring / no / spiral, used for stratifying
elif C.TASK == "ringtype":                                      # typed rings only
    lab = lab[(lab["label"] == "ring") & lab["ring_type"].isin(C.RING_TYPES)].copy()
    lab["y"] = (lab["ring_type"] == "inner").astype(int)       # inner = 1, outer = 0
    lab["cls"] = lab["ring_type"]                               # inner / outer
else:
    sys.exit(f"unknown TASK: {C.TASK}")
print(f"labels kept: {len(lab)} ({lab['cls'].value_counts().to_dict()})")

# seg area 
cols = pq.read_schema(C.MORPH_PATH).names
area_col = next((c for c in ["segmentation_area", "seg_area", "area"] if c in cols), None)
if area_col is None:
    sys.exit(f"No area column in morphology catalogue. Columns are: {cols}")

morph = (pd.read_parquet(C.MORPH_PATH, columns=["object_id", area_col]).rename(columns={area_col: "area"}).drop_duplicates("object_id"))
morph["object_id"] = morph["object_id"].astype("int64")

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

print(f"no area: -{n0-n1} | no image -{n1-n2} | area < {C.MIN_AREA}: -{n2-n3} | left: {n3}")

df["size_bin"] = pd.cut(df["area"], bins=C.SIZE_BINS, labels=C.SIZE_NAMES, right=False).astype(str)

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
    train, rest = train_test_split(df, train_size=f["train"], stratify=strat, random_state=C.SEED)
    val, test = train_test_split(rest, test_size=f["test"] / (f["val"] + f["test"]), stratify=strat.loc[rest.index], random_state=C.SEED)

    df["split"] = "train"
    df.loc[val.index, "split"] = "val"
    df.loc[test.index, "split"] = "test"
else:
    # NEW: ringtype reuses the ring run's split instead of making a new one.
    # the ringtype model starts from the ring model, which trained on 70% of all rings,
    # so a new split could put rings it already saw into the test set.
    ring_split = pd.read_csv(C.RING_SPLITS_PATH, usecols=["object_id", "split"])
    df = df.merge(ring_split, on="object_id", how="left")
    if df["split"].isna().any():
        sys.exit(f"{df['split'].isna().sum()} rings missing from {C.RING_SPLITS_PATH}; run once with TASK='ring' first")

# save and summary 
keep = ["object_id", "tile", "label", "y", "ring_type", "notes", "source", "area", "size_bin", "path", "split"]
df[keep].to_csv(C.SPLITS_PATH, index=False)                     # splits.csv or splits_ringtype.csv

summary = pd.crosstab([df["split"], df["cls"]], df["size_bin"], margins=True)
summary = summary[[c for c in C.SIZE_NAMES + ["All"] if c in summary.columns]]
summary.to_csv(C.TABLES / f"split_summary{C.TAG}.csv")

print("\n", summary)
print(f"\nfraction y=1 per split ({'ring' if C.TASK == 'ring' else 'inner'}):\n", df.groupby("split")["y"].mean().round(3))
print("\nmedian area by class:\n", df.groupby("cls")["area"].median().round(0))
print("\nclass by source:\n", pd.crosstab(df["source"], df["cls"]))  
print(f"\nsaved {C.SPLITS_PATH}")

# AUC = the probability that a randomly chosen ring gets a higher P(ring) score than a randomly chosen non-ring.