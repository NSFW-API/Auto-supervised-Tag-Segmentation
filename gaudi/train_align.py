#!/usr/bin/env python3
import argparse, os, json, math, time
import torch, torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from gaudi.data import make_dataloaders, load_tag_descriptions
from gaudi.models.image_text_model import ImageTextAligner

def bce_multi_logits(logits, targets):
    return nn.functional.binary_cross_entropy_with_logits(logits, targets)

def train(args):
    os.makedirs(args.out, exist_ok=True)
    # tag vocab from tag-descriptions JSON
    tag_desc = load_tag_descriptions(args.tag_desc)
    tag_vocab = sorted(tag_desc.keys())
    with open(os.path.join(args.out, "tag_vocab.json"), "w", encoding="utf-8") as f:
        json.dump({"tags": tag_vocab}, f, ensure_ascii=False, indent=2)

    tr, va = make_dataloaders(args.manifest, tag_vocab, batch_size=args.batch_size, image_size=args.image_size, workers=args.workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageTextAligner(num_tags=len(tag_vocab), img_backbone=args.backbone, embed_dim=args.embed_dim, pretrained=not args.no_pretrain).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))

    writer = SummaryWriter(log_dir=os.path.join(args.out, "tb"))
    best_val = float("inf")
    step = 0
    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(tr, desc=f"Train {epoch}/{args.epochs}")
        total = 0.0
        for xb, yb, _, _ in pbar:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = bce_multi_logits(logits, yb)
            loss.backward()
            opt.step()
            total += loss.item()*xb.size(0)
            writer.add_scalar("train/loss", loss.item(), step)
            step += 1
        sched.step()

        # val
        model.eval()
        vloss = 0.0
        correct = 0.0
        total_items = 0.0
        with torch.no_grad():
            for xb, yb, _, _ in va:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = bce_multi_logits(logits, yb)
                vloss += loss.item()*xb.size(0)
                # simple F1 proxy: thresh @ 0.3
                preds = (logits.sigmoid() >= 0.3).float()
                correct += (preds.eq(yb).sum().item())
                total_items += yb.numel()
        vloss /= max(1, len(va.dataset))
        acc = correct / max(1, total_items)
        writer.add_scalar("val/loss", vloss, epoch)
        writer.add_scalar("val/acc_proxy", acc, epoch)

        # checkpoints
        ckpt_dir = os.path.join(args.out, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save({"model": model.state_dict(), "tag_vocab": tag_vocab, "args": vars(args)},
                   os.path.join(ckpt_dir, "last.pt"))
        if vloss < best_val:
            best_val = vloss
            torch.save({"model": model.state_dict(), "tag_vocab": tag_vocab, "args": vars(args)},
                       os.path.join(ckpt_dir, "best.pt"))

    writer.close()
    print("Training complete. Checkpoints at", os.path.join(args.out, "checkpoints"))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--tag-desc", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--image-size", type=int, default=336)
    p.add_argument("--backbone", default="vit_base_patch16_384")
    p.add_argument("--embed-dim", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--no-pretrain", action="store_true", help="initialize backbone from scratch")
    args = p.parse_args()
    train(args)
