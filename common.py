"""Shared helpers for the FedAvg server and the edge clients."""
import os, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import f1_score

DATA   = Path(os.environ.get("DATA_ROOT", "/data"))      # mounted read-only
SHARED = Path(os.environ.get("SHARED_DIR", "/shared"))   # round files exchanged here
OUT    = Path(os.environ.get("OUT_DIR", "/outputs"))     # final results (server only)
IMG, BS = 128, 32
SEED = int(os.environ.get("SEED", 42))              # change per run for multi-seed experiments
IMAGE_COL, LABEL_COL, GROUP_COL = "image_id", "dx", "lesion_id"
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]   # same sorted order as the notebook
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TRAIN_TF = transforms.Compose([transforms.Resize((IMG, IMG)), transforms.RandomHorizontalFlip(),
                               transforms.RandomVerticalFlip(),
                               transforms.RandomRotation(20),          # same augmentation as the centralized baseline
                               transforms.ToTensor(), NORM])
EVAL_TF  = transforms.Compose([transforms.Resize((IMG, IMG)), transforms.ToTensor(), NORM])

def load_frame(csv_name):
    df = pd.read_csv(DATA / csv_name)
    index = {p.stem: p for p in (DATA / "dataset").rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    df["path"] = df[IMAGE_COL].astype(str).map(index.get)
    df = df.dropna(subset=["path"]).reset_index(drop=True)
    df["y"] = df[LABEL_COL].map({c: i for i, c in enumerate(CLASSES)})
    return df

class DS(Dataset):
    def __init__(self, frame, tf): self.f, self.tf = frame.reset_index(drop=True), tf
    def __len__(self): return len(self.f)
    def __getitem__(self, i):
        r = self.f.iloc[i]
        return self.tf(Image.open(r["path"]).convert("RGB")), int(r["y"])

def build_model():
    m = models.mobilenet_v3_small(weights=None)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, len(CLASSES))
    init = DATA / "init.pth"
    if init.exists(): m.load_state_dict(torch.load(init, map_location="cpu"))
    return m

@torch.no_grad()
def evaluate(model, frame):
    model.eval(); dl = DataLoader(DS(frame, EVAL_TF), BS, shuffle=False, num_workers=0)
    ce, loss, preds, labels = nn.CrossEntropyLoss(), 0.0, [], []
    for x, y in dl:
        out = model(x); loss += ce(out, y).item() * len(y)
        preds += out.argmax(1).tolist(); labels += y.tolist()
    return loss / len(labels), float(np.mean(np.array(preds) == np.array(labels))), f1_score(labels, preds, average="macro"), preds, labels

def atomic_save(obj, path):            # write then rename, so a reader never sees a half-written file
    tmp = Path(str(path) + ".tmp"); torch.save(obj, tmp); os.replace(tmp, path)

def wait_for(path, poll=2):
    while not Path(path).exists(): time.sleep(poll)