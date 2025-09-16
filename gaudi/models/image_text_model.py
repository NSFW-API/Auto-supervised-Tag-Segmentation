from typing import List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class ImageTextAligner(nn.Module):
    """
    Simple CLIP-ish model:
      - Image encoder (timm backbone) → global embedding
      - Text embeddings are parameters we learn per-tag (initialized from external encoder or random)
      - Multi-label logits = (img_emb @ tag_emb.T)/temp
    """
    def __init__(self, num_tags: int, img_backbone: str = "vit_base_patch16_384", embed_dim: int = 512, pretrained: bool = True):
        super().__init__()
        self.image = timm.create_model(img_backbone, pretrained=pretrained, num_classes=0, global_pool="avg")
        feat_dim = self.image.num_features
        self.proj = nn.Linear(feat_dim, embed_dim)
        self.tag_emb = nn.Parameter(torch.randn(num_tags, embed_dim) * 0.02)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.image(images)                    # [B, F]
        emb = F.normalize(self.proj(feats), dim=-1)   # [B, D]
        tag = F.normalize(self.tag_emb, dim=-1)       # [T, D]
        scale = self.logit_scale.exp().clamp(1.0, 100.0)
        logits = scale * emb @ tag.t()                # [B, T]
        return logits

    @torch.no_grad()
    def image_embed(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.image(images)
        return F.normalize(self.proj(feats), dim=-1)
