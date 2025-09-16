#!/usr/bin/env python3
"""
Generate pseudo-masks from alignment model using text-conditioned activation maps
+ unsupervised region proposals + automatic validators.
"""
import argparse, os, json, math, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from tqdm import tqdm

from gaudi.data import load_tag_descriptions, ManifestDataset
from gaudi.models.image_text_model import ImageTextAligner
from gaudi.regionizer import region_proposals

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    HAS_CRF = True
except Exception:
    HAS_CRF = False

def load_ckpt(path):
    ck = torch.load(path, map_location="cpu")
    return ck

def heatmap_from_align(image_tensor: torch.Tensor, model: ImageTextAligner, tag_vec: torch.Tensor):
    """
    Produce a coarse heatmap by computing similarity with spatial feature map.
    We'll approximate by sliding the global projector over feature map with hooks.
    """
    # Get last conv/patch token map via forward hook: use timm forward_features
    feats = model.image.forward_features(image_tensor)  # [B,C,H',W'] for convnets; for ViT, timm returns sequence
    if feats.ndim == 4:
        C, H, W = feats.shape[1:]
        # project per-location
        f = feats
    else:
        # ViT: [B, tokens, C]; drop class token and reshape to grid (assume square)
        B, N, C = feats.shape
        S = int(math.sqrt(N-1))
        f = feats[:,1:,:].transpose(1,2).reshape(B, C, S, S)
        H, W = S, S
    # Use model.proj to map features to emb space
    # Collapse spatial by per-location projection
    B = f.shape[0]
    fmap = f
    # flatten spatial
    fmap2 = fmap.flatten(2).transpose(1,2)  # [B, HW, C]
    Wp = model.proj.weight.t()  # [F, D] -> [D, F] after transpose
    bp = model.proj.bias
    loc_emb = F.normalize(fmap2 @ Wp + bp, dim=-1)  # [B, HW, D]
    tag = F.normalize(tag_vec, dim=-1)             # [B, D]
    sim = (loc_emb * tag.unsqueeze(1)).sum(-1)     # [B, HW]
    sim = sim.reshape(B, H, W)
    sim = (sim - sim.amin(dim=(1,2), keepdim=True)) / (sim.amax(dim=(1,2), keepdim=True) - sim.amin(dim=(1,2), keepdim=True) + 1e-6)
    # upsample to input size
    sim_up = F.interpolate(sim.unsqueeze(1), size=image_tensor.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)  # [B,H,W]
    return sim_up

def crf_refine(image_np: np.ndarray, prob: np.ndarray) -> np.ndarray:
    if not HAS_CRF:
        return (prob > 0.5).astype(np.uint8)
    H,W = prob.shape
    d = dcrf.DenseCRF2D(W, H, 2)
    U = np.stack([1-prob, prob], axis=0)
    U = np.clip(U, 1e-6, 1-1e-6)
    U = -np.log(U)
    U = U.reshape(2, -1).astype(np.float32)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=50, srgb=13, rgbim=image_np.astype(np.uint8), compat=10)
    Q = d.inference(5)
    seg = np.argmax(np.array(Q).reshape(2, H, W), axis=0)
    return seg.astype(np.uint8)

def validators(image_np: np.ndarray, mask: np.ndarray, img_score: float, heatmap: np.ndarray, min_area_frac=0.002):
    H,W = mask.shape
    area = mask.sum() / (H*W + 1e-6)
    if area < min_area_frac or area > 0.8:
        return False, {"reason":"area"}
    # Consistency under flip
    mflip = np.fliplr(mask)
    iou = (mask & mflip).sum() / ((mask | mflip).sum() + 1e-6)
    # proxy causal: mean heat inside vs outside
    hin = heatmap[mask>0].mean() if mask.any() else 0.0
    hout = heatmap[mask==0].mean() if (mask==0).any() else 0.0
    if hin < hout + 0.05:
        return False, {"reason":"causal"}
    return True, {"iou_flip": float(iou), "hin": float(hin), "hout": float(hout)}

def run(args):
    os.makedirs(args.out, exist_ok=True)
    # Load vocab
    with open(args.align_ckpt, "rb") as f:
        ck = torch.load(f, map_location="cpu")
    tag_vocab = ck["tag_vocab"]
    tag2idx = {t:i for i,t in enumerate(tag_vocab)}
    # maps for descriptions (optional in this stage)
    tag_desc = load_tag_descriptions(args.tag_desc)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from gaudi.models.image_text_model import ImageTextAligner
    model = ImageTextAligner(num_tags=len(tag_vocab))
    model.load_state_dict(ck["model"], strict=False)
    model.eval().to(device)

    # Dataset (image paths only, not used as labels beyond tag list)
    ds = ManifestDataset(args.manifest, tag_vocab, image_size=args.image_size)

    stats = []
    for idx in tqdm(range(len(ds)), desc="pseudo"):
        x, y, img_path, tags = ds[idx]
        if not tags: continue
        with torch.no_grad():
            xb = x.unsqueeze(0).to(device)
            img_np = np.array(Image.open(img_path).convert("RGB"))
            # region proposals
            props = region_proposals(Image.fromarray(img_np), n_segs=args.n_segs)
            if len(props)==0: 
                continue
            # Precompute per-tag vector
            # Here we use learned tag_emb from checkpoint
            tag_emb_table = model.tag_emb.data  # [T,D]
            for t in tags:
                if t not in tag2idx: 
                    continue
                ti = tag2idx[t]
                tag_vec = tag_emb_table[ti:ti+1].to(device)  # [1,D]
                heat = heatmap_from_align(xb, model, tag_vec)[0].cpu().numpy()  # [H,W]
                # score proposals
                scores = []
                for m in props:
                    s = float(heat[m].mean())
                    scores.append(s)
                if not scores: 
                    continue
                # pick top K and make soft map
                K = min(args.topk, len(scores))
                top_idx = np.argsort(scores)[-K:][::-1]
                prob = np.zeros_like(heat)
                for i in top_idx:
                    prob = np.maximum(prob, heat * (props[i].astype(np.float32)))
                prob = (prob - prob.min()) / (prob.max() - prob.min() + 1e-6)
                # CRF refine (optional)
                seg = crf_refine(img_np, prob)
                ok, meta = validators(img_np, seg, 0.0, heat)
                if not ok:
                    continue
                # Save mask
                outdir = os.path.join(args.out, t)
                os.makedirs(outdir, exist_ok=True)
                stem = os.path.splitext(os.path.basename(img_path))[0]
                outpath = os.path.join(outdir, f"{stem}.png")
                Image.fromarray((seg*255).astype(np.uint8)).save(outpath)
                stats.append({"img": img_path, "tag": t, "mask": outpath, **meta})
    with open(os.path.join(args.out, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("Done pseudo-labels →", args.out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--tag-desc", required=True)
    p.add_argument("--align-ckpt", required=True)
    p.add_argument("--segmenter-ckpt", default=None, help="(optional) refine scoring with a trained segmenter")
    p.add_argument("--out", required=True)
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--n-segs", type=int, default=300)
    p.add_argument("--topk", type=int, default=5)
    args = p.parse_args()
    run(args)
