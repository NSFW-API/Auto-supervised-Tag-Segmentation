import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class SimpleTextUNet(nn.Module):
    """
    Very small UNet-like decoder conditioned on a tag embedding.
    - Image encoder: timm backbone feature map (conv/ViT with feature hook -> here we use a small convnet)
    - Text conditioning: broadcast tag embedding and fuse via FiLM (scale+shift)
    """
    def __init__(self, img_backbone: str = "resnet18", text_dim: int = 512):
        super().__init__()
        self.backbone = timm.create_model(img_backbone, pretrained=True, features_only=True, out_indices=(1,2,3,4))
        chs = self.backbone.feature_info.channels()
        self.cond = nn.Sequential(nn.Linear(text_dim, 256), nn.ReLU(), nn.Linear(256, chs[-1]*2))
        self.up3 = nn.ConvTranspose2d(chs[-1], chs[-2], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(chs[-2], chs[-3], kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(chs[-3], chs[-4], kernel_size=2, stride=2)
        self.head = nn.Sequential(nn.Conv2d(chs[-4], 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 1, 1))

    def forward(self, images: torch.Tensor, tag_emb: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(images)
        x = feats[-1]
        # FiLM conditioning
        gamma_beta = self.cond(tag_emb)  # [B, 2C]
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1)
        x = x * (1 + gamma) + beta

        x = self.up3(x) + feats[-2]
        x = self.up2(x) + feats[-3]
        x = self.up1(x) + feats[-4]
        logit = self.head(x)
        logit = F.interpolate(logit, size=images.shape[-2:], mode="bilinear", align_corners=False)  # [B,1,H,W]
        return logit
