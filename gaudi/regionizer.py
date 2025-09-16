from typing import List, Tuple, Dict
import numpy as np
from skimage.segmentation import slic, felzenszwalb
from skimage.measure import label, regionprops
from skimage import img_as_float
from PIL import Image

def region_proposals(image: Image.Image, n_segs: int = 300) -> List[np.ndarray]:
    """
    Return a list of boolean masks (H,W) as region proposals.
    Uses SLIC + Felzenszwalb to get a diverse set.
    """
    im = np.array(image.convert("RGB"))
    imgf = img_as_float(im)
    H, W = im.shape[:2]
    seg1 = slic(imgf, n_segments=n_segs, compactness=20, start_label=1, channel_axis=2)
    seg2 = felzenszwalb(imgf, scale=200, sigma=0.6, min_size=60)
    masks = []
    def seg_to_masks(seg):
        for r in regionprops(seg):
            if r.area < 30: continue
            m = np.zeros((H,W), dtype=bool); m[seg==r.label] = True
            masks.append(m)
    seg_to_masks(seg1); seg_to_masks(seg2)
    # de-duplicate by IoU
    uniq = []
    for m in masks:
        ok = True
        for u in uniq:
            inter = (m & u).sum(); union = (m | u).sum()
            if union>0 and inter/union > 0.9:
                ok = False; break
        if ok:
            uniq.append(m)
    return uniq[:800]
