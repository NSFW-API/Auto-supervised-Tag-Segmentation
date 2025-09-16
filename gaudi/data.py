import os
import csv
import random
from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import json

def load_tag_descriptions(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    # normalize to list[str]
    out = {}
    for k, v in d.items():
        if isinstance(v, str):
            out[k] = [v]
        elif isinstance(v, list):
            out[k] = [str(x) for x in v]
        else:
            out[k] = [str(v)]
    return out

class ManifestDataset(Dataset):
    def __init__(self, manifest_csv: str, tag_vocab: List[str], image_size: int = 336):
        self.rows = []
        with open(manifest_csv, "r", encoding="utf-8") as f:
            rr = csv.DictReader(f)
            for r in rr:
                self.rows.append({"image": r["image_path"], "tags": (r["tags"] or "").split()})
        self.tag2idx = {t:i for i,t in enumerate(tag_vocab)}
        self.size = image_size
        self.tf = T.Compose([
            T.Resize(self.size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(self.size),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        img = Image.open(r["image"]).convert("RGB")
        x = self.tf(img)
        y = torch.zeros(len(self.tag2idx), dtype=torch.float32)
        for t in r["tags"]:
            if t in self.tag2idx:
                y[self.tag2idx[t]] = 1.0
        return x, y, r["image"], r["tags"]

def make_dataloaders(manifest_csv: str, tag_vocab: List[str], batch_size: int=32, image_size:int=336, workers:int=4):
    ds = ManifestDataset(manifest_csv, tag_vocab, image_size=image_size)
    # simple split
    n = len(ds)
    n_val = max(1, int(0.05*n))
    idx = list(range(n))
    random.shuffle(idx)
    val_idx = set(idx[:n_val])
    tr_idx = [i for i in idx if i not in val_idx]
    class Subset(torch.utils.data.Dataset):
        def __init__(self, ds, ids): self.ds, self.ids = ds, ids
        def __len__(self): return len(self.ids)
        def __getitem__(self, i): return self.ds[self.ids[i]]
    tr = Subset(ds, tr_idx)
    va = Subset(ds, list(val_idx))
    return (
        DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True),
        DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    )
