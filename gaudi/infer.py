#!/usr/bin/env python3
import argparse, os, json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from gaudi.models.image_text_model import ImageTextAligner
from gaudi.models.segmenter import SimpleTextUNet

def rle_encode(mask: np.ndarray) -> dict:
    # Basic uncompressed RLE for demo
    flat = mask.flatten(order="F").astype(np.uint8)
    counts = []
    last = 0
    run = 0
    for v in flat:
        if v==last:
            run+=1
        else:
            counts.append(run)
            run = 1
            last = v
    counts.append(run)
    return {"counts": counts, "size": list(mask.shape[::-1])}

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load align ckpt (for tag vocab + embeddings)
    a = torch.load(args.align_ckpt, map_location="cpu")
    tag_vocab = a["tag_vocab"]; tag2idx = {t:i for i,t in enumerate(tag_vocab)}
    align = ImageTextAligner(num_tags=len(tag_vocab))
    align.load_state_dict(a["model"], strict=False)
    tag_table = align.tag_emb.data.clone().to(device)

    seg = None
    if args.segmenter_ckpt:
        s = torch.load(args.segmenter_ckpt, map_location="cpu")
        seg = SimpleTextUNet()
        seg.load_state_dict(s["model"], strict=False)
        seg.to(device).eval()

    im = Image.open(args.image).convert("RGB")
    H,W = im.size[1], im.size[0]
    im_r = im.resize((args.image_size, args.image_size))
    x = torch.from_numpy(np.asarray(im_r).transpose(2,0,1)).float()/255.0
    x = x.unsqueeze(0).to(device)

    out = {}
    for t in args.tags.split():
        if t not in tag2idx: 
            continue
        ti = tag2idx[t]
        temb = tag_table[ti:ti+1]
        if seg is None:
            # Fallback: simple heatmap threshold
            from gaudi.pseudolabels import heatmap_from_align
            hm = heatmap_from_align(x, align, temb)[0].cpu().numpy()
            m = (hm > 0.5).astype(np.uint8)
        else:
            with torch.no_grad():
                logit = seg(x, temb)
                m = (torch.sigmoid(logit)[0,0].cpu().numpy() > 0.5).astype(np.uint8)
        # upscale to original
        m = np.array(Image.fromarray(m*255).resize(im.size, Image.NEAREST))>127
        out[t] = [ {"mask_rle": rle_encode(m), "score": 0.9} ]
        masks_for_viz[t] = m.copy()

    # Save
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    if args.viz:
        vis = im.copy().convert("RGBA")
        import random
        for t, lst in out.items():
            m = masks_for_viz.get(t)
            if m is None: 
                continue
            color = (random.randint(0,255), random.randint(0,255), random.randint(0,255), 90)
            overlay = Image.new("RGBA", im.size, (0,0,0,0))
            pix = overlay.load()
            H,W = m.shape
            for y in range(H):
                for x in range(W):
                    if m[y,x]:
                        pix[x,y] = color
            vis = Image.alpha_composite(vis, overlay)
        vis.save(args.viz)

    print("Wrote", args.out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--tags", required=True, help="space-separated tag list to localize")
    p.add_argument("--align_ckpt", required=True)
    p.add_argument("--segmenter_ckpt", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--viz", default=None)
    p.add_argument("--image_size", type=int, default=384)
    args = p.parse_args()
    main(args)
