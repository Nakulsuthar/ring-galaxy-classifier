""" Network, image loading and freezing logic"""
import torch 
import torch.nn as nn 
import timm 
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T 
import config as C 

# images 
# same processing as the E1024 embeddings, resize to 224, ImageNet normalisation 
EVAL_TF = T.Compose([T.Resize((C.IMG_SIZE, C.IMG_SIZE)), T.ToTensor(), T.Normalize(C.MEAN, C.STD),])

# training only: galaxies have no preferred orientation, so random rotation gives 
# new examples for free and stop the model learning orientation 
TRAIN_TF = T.Compose([T.RandomRotation(180),T.RandomHorizontalFlip(),T.RandomVerticalFlip(),T.Resize((C.IMG_SIZE, C.IMG_SIZE)),T.ToTensor(),T.Normalize(C.MEAN, C.STD),])

class Galaxies(Dataset):
    """ Returns (image tensor, label, loss weight) for each row of a splits table."""
    def __init__(self, df, tf):
        self.paths = df["path"].tolist()
        self.y = df["y"].to_numpy("float32")
        self.w = df["w"].to_numpy("float32") if "w" in df else [1.0] * len(df)
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), torch.tensor(self.y[i]), torch.tensor(self.w[i])


# network 
# ringNet is the ring classifer 
# it is two pirces stacked together
# 1. Encoder: basically what zoobot already does, returns 1024 numbers which then are the features described.
# 2. Head: one new linear layer (1024 -> 1), starts random. score is given as a function of weights and features 
# then the sigmoid is applied on the score to give a number between 0 to 1 which will be the P(ring)
class RingNet(nn.Module):
    """ Zoobot encoder (image -> 1024 features) + a new linear head (1024 -> 1 logit)"""
    def __init__(self,pretrained=True):
        super().__init__()
        self.encoder = timm.create_model(C.MODEL_NAME, pretrained=pretrained, num_classes=0)
        self.head = nn.Linear(self.encoder.num_features, 1)

    def forward(self, x):
        return self.head(self.encoder(x)).squeeze(1)      #logit, sigmoid gives p(ring)


def set_phase(model, phase):
    """ Phase1: only the head trains. Phase 2: last UNFREEZE_BLOCKS blocks + head train."""
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True 
    if phase == 2:
        for block in model.encoder.stages[-C.UNFREEZE_BLOCKS:]:
            for p in block.parameters():
                p.requires_grad = True
        if hasattr(model.encoder, "head"):
            for p in model.encoder.head.parameters():
                p.requires_grad = True 



