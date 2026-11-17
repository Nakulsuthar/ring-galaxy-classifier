""" STEP1: Build the labelled dataset and split it into train val test 

labels.csv + morphology catalogue (segmentation area) + image paths 
-> keep ring (1) and no/spiral (0)
-> drop galaxies with area < MIN_AREA
-> assign a bin size 
-> split 70/15/15, stratified by label and size bin
-> data/splits.csv
"""

import sys 
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for importing config.py

import pandas as pd 
import pyarrow.parquet as pq 
from sklearn.model_selection import train_test_split
import config as C


# labels 
lab = pd.read_csv(C.LABELS_PATH)
lab = lab[lab["label"].isin(C.POSITIVE | C.NEGATIVE)].copy()
lab["y"] = lab["label"].isin(C.POSITIVE).astype(int)
print(f"labels kept: {len(lab)} ({lab['label'].value_counts().to_dict()})")

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
# stratfied split 
# this basically makes 9 boxes like this 
#  seg are       700–1500 px   1500–3000 px   >3000 px
#  ring            box 1          box 2         box 3
#  no              box 4          box 5         box 6
#  spiral          box 7          box 8         box 9

# good for not having messed up training and making sure uniformity is consistent


strat = df["label"] + "_" + df["size_bin"]      # ring no spiral x size bin 
f = C.SPLIT
train, rest = train_test_split(df, train_size=f["train"], stratify=strat, random_state=C.SEED)
val, test = train_test_split(rest, test_size=f["test"] / (f["val"] + f["test"]), stratify=strat.loc[rest.index], random_state=C.SEED)

df["split"] = "train"
df.loc[val.index, "split"] = "val"
df.loc[test.index, "split"] = "test"

# save and summary 
keep = ["object_id", "tile", "label", "y", "ring_type", "notes", "source", "area", "size_bin", "path", "split"]
df[keep].to_csv(C.DATA / "splits.csv", index=False)

summary = pd.crosstab([df["split"], df["label"]], df["size_bin"], margins=True)
summary = summary[[c for c in C.SIZE_NAMES + ["All"] if c in summary.columns]]
summary.to_csv(C.TABLES / "split_summary.csv")

print("\n", summary)
print("\nring fraction per split:\n", df.groupby("split")["y"].mean().round(3))
print("\nmedian area by label:\n", df.groupby("label")["area"].median().round(0))
print(f"\nsaved {C.DATA / 'splits.csv'}")

