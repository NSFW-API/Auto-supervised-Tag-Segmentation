#!/usr/bin/env python3
import argparse, os, json, glob
import numpy as np
import torch, torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from gaudi.data import ManifestDataset, load_tag_descriptions
from gaudi.models.segmenter import SimpleTextUNet
from gaudi.models.image_text_model import ImageTextAligner

class SegDataset(Dataset):
    def __init__(self, manifest_csv: str, tag_vocab, pseudomask_root: str, image_size:int=384):
        self.rows = []
        self.tag2idx = {t:i for i,t in enumerate(tag_vocab)}
        # Build an index of available masks
        mask_map = {}
        for t in tag_vocab:
            d = os.path.join(pseudomask_root, t)
            if os.path.isdir(d):
                for fn in os.listdir(d):
                    mask_map.setdefault(fn, []).append((t, os.path.join(d, fn)))
        # manifest rows
        import csv
        with open(manifest_csv, "r", encoding="utf-8") as f:
            rr = csv.DictReader(f)
            for r in rr:
                stem = os.path.splitext(os.path.basename(r["image_path"]))[0] + ".png"
                if stem in mask_map:
                    self.rows.append((r["image_path"], mask_map[stem]))  # list of (tag,mask)
        self.tf_img = lambda im: torch.from_numpy(np.asarray(im.resize((image_size, image_size))).transpose(2,0,1)).float()/255.0
        self.image_size = image_size

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        img_path, pairs = self.rows[idx]
        im = Image.open(img_path).convert("RGB")
        im = im.resize((self.image_size, self.image_size), Image.BICUBIC)
        x = torch.from_numpy(np.asarray(im).transpose(2,0,1)).float()/255.0
        # randomly pick one tag per sample for diversity
        t, mpath = pairs[np.random.randint(len(pairs))]
        y = np.asarray(Image.open(mpath).convert("L").resize((self.image_size, self.image_size), Image.NEAREST))>127
        y = torch.from_numpy(y.astype(np.float32))[None,...]
        return x, t, y, img_path

def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    num = 2*(probs*targets).sum(dim=(1,2,3))
    den = (probs*probs + targets*targets).sum(dim=(1,2,3)) + eps
    return 1 - (num/den).mean()

def train(args):
    os.makedirs(args.out, exist_ok=True)
    # Load tag vocab
    with open(os.path.join(args.align_out, "tag_vocab.json"), "r", encoding="utf-8") as f:
        tag_vocab = json.load(f)["tags"]
    dataset = SegDataset(args.manifest, tag_vocab, args.pseudomasks, image_size=args.image_size)
    # split
    n = len(dataset)
    n_val = max(1, int(0.05*n))
    idx = list(range(n))
    import random; random.shuffle(idx)
    val_idx = set(idx[:n_val])
    tr_idx = [i for i in idx if i not in val_idx]
    class Subset(torch.utils.data.Dataset):
        def __init__(self, ds, ids): self.ds,self.ids=ds,ids
        def __len__(self): return len(self.ids)
        def __getitem__(self,i): return self.ds[self.ids[i]]
    tr = DataLoader(Subset(dataset,tr_idx), batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    va = DataLoader(Subset(dataset,list(val_idx)), batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg = SimpleTextUNet(img_backbone="resnet18", text_dim=args.embed_dim).to(device)
    # load aligner to get tag embeddings
    ck = torch.load(os.path.join(args.align_out, "checkpoints", "last.pt"), map_location="cpu")
    align = ImageTextAligner(num_tags=len(tag_vocab))
    align.load_state_dict(ck["model"], strict=False)
    tag_table = align.tag_emb.data.clone().to(device)  # [T,D]

    opt = torch.optim.AdamW(seg.parameters(), lr=args.lr, weight_decay=1e-4)
    writer = SummaryWriter(log_dir=os.path.join(args.out, "tb"))
    best = 1e9
    step = 0
    for ep in range(1, args.epochs+1):
        seg.train()
        for xb, tname, yb, _ in tqdm(tr, desc=f"seg train {ep}/{args.epochs}"):
            xb, yb = xb.to(device), yb.to(device)
            # gather tag embeddings
            tidx = torch.tensor([tag_vocab.index(t) for t in tname], dtype=torch.long, device=device)
            temb = tag_table[tidx]  # [B,D]
            opt.zero_grad(set_to_none=True)
            logit = seg(xb, temb)
            loss = 0.5*dice_loss(logit, yb) + 0.5*nn.functional.binary_cross_entropy_with_logits(logit, yb)
            loss.backward(); opt.step()
            writer.add_scalar("train/loss", loss.item(), step); step+=1
        # val
        seg.eval()
        vloss = 0.0
        with torch.no_grad():
            for xb, tname, yb, _ in va:
                xb, yb = xb.to(device), yb.to(device)
                tidx = torch.tensor([tag_vocab.index(t) for t in tname], dtype=torch.long, device=device)
                temb = tag_table[tidx]
                logit = seg(xb, temb)
                loss = 0.5*dice_loss(logit, yb) + 0.5*nn.functional.binary_cross_entropy_with_logits(logit, yb)
                vloss += loss.item()*xb.size(0)
        vloss /= max(1,len(va.dataset))
        writer.add_scalar("val/loss", vloss, ep)
        ckdir = os.path.join(args.out, "checkpoints"); os.makedirs(ckdir, exist_ok=True)
        torch.save({"model": seg.state_dict(), "args": vars(args)}, os.path.join(ckdir, "last.pt"))
        if vloss < best:
            best = vloss
            torch.save({"model": seg.state_dict(), "args": vars(args)}, os.path.join(ckdir, "best.pt"))
    writer.close()
    print("Segmenter training complete. Checkpoints saved to", os.path.join(args.out, "checkpoints"))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--pseudomasks", required=True)
    p.add_argument("--align-out", default="runs/align_s1", help="folder where train_align saved tag_vocab.json + checkpoints/last.pt")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--embed-dim", type=int, default=512)
    args = p.parse_args()
    train(args)
