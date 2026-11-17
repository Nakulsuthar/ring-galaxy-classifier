""" Paths and settings shared by every script """ 
from pathlib import Path
import torch 

# Paths 

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
for d in (DATA, TABLES, FIGURES):
    d.mkdir(exist_ok=True)


# exisiting stuff 

SRC = ROOT.parent / "get_training_data"
LABELS_PATH = SRC / "tables" / "labels.csv"
MORPH_PATH = SRC / "data" / "morphology_catalogue.parquet"
IMAGE_DIR = SRC / "data" / "cutouts_jpg_gz"
EMB_PATH = SRC / "data" / "E1024_all.npy"
KEY_PATH = SRC / "data" / "E1024_keys.npy"

def image_path(tile, object_id):
    """ Path to a galaxy's JPG from its tile and object ID"""
    oid = int(object_id)
    oid_str = f"NEG{abs(oid)}" if oid < 0 else str(oid)
    return IMAGE_DIR / str(int(tile)) / f"{int(tile)}_{oid_str}_gz_arcsinh_vis_y.jpg"


# Labels 

POSITIVE = {"ring"}         # 1
NEGATIVE = {"no", "spiral"} # 0
# maybe bad and lens are left out 
RING_TYPES = ["inner", "outer"]  # for second stage model 

# sample cuts 
MIN_AREA = 700  # seg area limit 
SIZE_BINS = [700, 1500, 3000, 1e9]
SIZE_NAMES = ["700-1500", "1500-3000", ">3000"]

# split 
SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42 

# model
MODEL_NAME = "hf-hub:mwalmsley/zoobot-encoder-euclid-convnext-base"
IMG_SIZE   = 224
MEAN, STD  = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]   # ImageNet normalisation, same as the E1024 embeddings

# training 
# model is ConvNeXt-Base which is pretrained on galazy zoo euclid already + a new linear head (1024 -> 1) giving p(ring)

# input 224x224x3 (H x W x C (Color channels))
 # after stem the images are shrunk into a 56x56 grid where each grid has 128 numbers where each number is a feature
 # each layer then halves the grid and doubles the features
 # early layers will be frozen to avoid overfitting plus its a pretrained model so no point 
 
# STAGE 1 
# All of zoobot is frozen. Only the new head (1024 -> 1) learns 
# Runs for 5 EPOCHS at a large LR 
# Purpose is to get the new hwad before touching zoobots layers

# STAGE 2 
# Unfreeze the last layer of zoobot and train it with LR = 1e-5 and EPOCHS = 15 

 



BATCH_SIZE = 32
EPOCHS_HEAD = 5  # stage 1 where im only training the new head 
EPOCHS_FINE = 15 # stage 2 where i also unfreeze the last encoder blocks
LR_HEAD = 1e-3
LR_FINE = 1e-5
WEIGHT_DECAY = 0.05
UNFREEZE_BLOCKS = 1 # How many of the 4 ConvNeXt stages to unfreeze stafe 2 
PATIENCE = 5 # early stopping on validation AUC
NUM_WORKERS = 0 

DEVICE = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")