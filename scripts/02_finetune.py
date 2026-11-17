"""
Step 2: fine tune zoobot to classify rings in Phases
 
Phase 1: encoder frozen, train only the new head   (LR_HEAD, EPOCHS_HEAD)
Phase 2: unfreeze last block + head which has 27 M weights  (LR_FINE, EPOCHS_FINE)
    early stopping on validation AUC (PATIENCE) 
    
Loss is weighted so rings and non rings count equally inside each size bin, 
so "bigger = ring" gives the model no information . 
-> data/best_model.pt, tables/train_log.csv
"""

import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # so config and model import

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, log_loss
import config as C
from model import RingNet, set_phase, Galaxies, TRAIN_TF, EVAL_TF

torch.manual_seed(C.SEED)
np.random.seed(C.SEED)
print("device", C.DEVICE)

# data
df = pd.read_csv(C.DATA / "splits.csv")
train_df = df[df["split"] == "train"].reset_index(drop=True)
val_df = df[df["split"] == "val"].reset_index(drop=True)

# size balanced weights: inside each size bin, rings and non rings get equal total weights 
# like in 700-1500 bin is 22% rings -> each ring counts 2.3x each non ring counts 0.6x

n_bin = train_df.groupby("size_bin")["y"].transform("size")           
n_cell = train_df.groupby(["size_bin", "y"])["y"].transform("size")
w = n_bin / (2 * n_cell)
train_df["w"] = w / w.mean()
print(train_df.groupby(["size_bin", "y"])["w"].first().round(2).unstack())

# loaders: galaxies turns table rows into (image, label, weight), dataloader groups them in batches of 32 
# train is shuffled in each epoch, with random rotation and flips 
# val is fixed order with no augmentation, so the score is consistent between epochs
train_loader = DataLoader(Galaxies(train_df, TRAIN_TF), batch_size=C.BATCH_SIZE,shuffle=True,num_workers=C.NUM_WORKERS)
val_loader   = DataLoader(Galaxies(val_df, EVAL_TF), batch_size=C.BATCH_SIZE,shuffle=False, num_workers=C.NUM_WORKERS)

# model 
# bce or binary cross entropy is a standard way of measuring how wrong a predicted probability is 
model = RingNet().to(C.DEVICE)
bce = nn.BCEWithLogitsLoss(reduction="none")

def train_epoch(phase, opt):
    """ One pass over the training set. Returns the mean weighted loss"""
    model.train()
    if phase == 1:
        model.encoder.eval()
    total, n = 0.0, 0
    for x, y, w in train_loader:
        x, y, w = x.to(C.DEVICE), y.to(C.DEVICE), w.to(C.DEVICE)
        if phase == 1: 
            with torch.no_grad():
                feats = model.encoder(x)
            logits = model.head(feats).squeeze(1)
        else:
            logits = model(x)
        loss = (bce(logits, y) * w).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / n 


def evaluate():
    """ P(ring) for every validation galaxy"""
    model.eval()
    ps = []
    with torch.no_grad():
        for x, _, _ in val_loader:
            ps.append(torch.sigmoid(model(x.to(C.DEVICE))).cpu())
    return torch.cat(ps).numpy()

# training 
ys = val_df["y"].to_numpy()
log, best_auc = [], -1.0

for phase, epochs, lr in [(1, C.EPOCHS_HEAD, C.LR_HEAD), (2, C.EPOCHS_FINE, C.LR_FINE)]:
    set_phase(model,phase)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=C.WEIGHT_DECAY)
    print(f"\n==== Phase {phase}: {sum(p.numel() for p in params)/1e6:.3f} M trainable params, lr={lr} ==")
    bad = 0    # epochs since the last improvment 

    for ep in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(phase, opt)

        # val: overall AUC, loss and AUC per bin size
        ps = evaluate()
        auc = roc_auc_score(ys, ps)
        val_loss = log_loss(ys, np.clip(ps, 1e-6, 1 - 1e-6))
        bin_auc = {}
        for b in C.SIZE_NAMES:
            m = (val_df["size_bin"] == b).to_numpy()
            if m.any():
                bin_auc[b] = roc_auc_score(ys[m], ps[m])

        # log this epoch so it survives any crash 
        row = {"phase": phase, "epoch": ep, "train_loss": train_loss, "val_loss": val_loss,
                       "val_auc": auc, **{f"auc_{b}": v for b, v in bin_auc.items()},
                       "minutes": (time.time() - t0) / 60}
        log.append(row)
        pd.DataFrame(log).round(4).to_csv(C.TABLES / "train_log.csv", index=False)

        bins_txt = "  ".join(f"{b}: {v:.3f}" for b, v in bin_auc.items())
        print(f"P{phase} ep{ep:2d} | train {train_loss:.3f} | val {val_loss:.3f} | "
                      f"AUC {auc:.3f} [{bins_txt}] | {row['minutes']:.1f} min", end="")

        # store and keep the best model by validation AUC, early stopping in phase 2 
        if auc > best_auc:
            best_auc, bad = auc, 0 
            torch.save({"state_dict": model.state_dict(), "phase": phase, "epoch": ep,
                        "val_auc": auc}, C.DATA / "best_model.pt")
            print("  <- best, saved!!")
        else: 
            bad += 1 
            print()
            if phase == 2 and bad >= C.PATIENCE:
                print(f"no improvement for {C.PATIENCE} epochs, stopping")
                break 


print(f"\nbest val AUC {best_auc:.3f}, saved to {C.DATA / "best_model.pt"}")