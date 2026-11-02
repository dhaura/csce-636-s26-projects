"""Model construction: any timm ImageNet classifier with a new 7-class head, optionally fused with patient metadata."""
import re

import timm
import torch
from torch import nn

# Families whose position embeddings are tied to the input size (timm resamples them when img_size is given).
_SIZED_FAMILIES = ("vit_", "eva02_", "deit", "beit")


def build_model(name, num_classes, img_size, pretrained=True, grad_checkpointing=False, drop_path_rate=0.0, n_meta=0):
    """timm model with a fresh head. n_meta > 0 returns a MetaFusion model that also takes a metadata vector."""
    kwargs = {"img_size": img_size} if name.startswith(_SIZED_FAMILIES) else {}
    if drop_path_rate:
        kwargs["drop_path_rate"] = drop_path_rate
    model = timm.create_model(name, pretrained=pretrained, num_classes=0 if n_meta else num_classes, **kwargs)
    if grad_checkpointing:
        model.set_grad_checkpointing(True)
    return MetaFusion(model, n_meta, num_classes) if n_meta else model


class MetaFusion(nn.Module):
    """Late fusion: [pooled image features, MLP(metadata)] -> linear classifier."""

    def __init__(self, backbone, n_meta, num_classes, hidden=64):
        super().__init__()
        self.backbone = backbone
        self.meta = nn.Sequential(nn.Linear(n_meta, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.head = nn.Linear(backbone.num_features + hidden, num_classes)

    def forward(self, x, meta):
        return self.head(torch.cat([self.backbone(x), self.meta(meta.to(x.dtype))], dim=1))


def convnext_layer_id(name, n_layers=14):
    """Layer index for layer-wise LR decay, grouped as in the ConvNeXt paper's fine-tuning code:
    stem 0, stage 0 -> 1, stage 1 -> 2, stage 2 -> 3..11 (three blocks per group), stage 3 -> 12, head 13."""
    name = name.removeprefix("backbone.")
    if name.startswith("stem"):
        return 0
    m = re.match(r"stages\.(\d+)\.(?:blocks\.(\d+)|downsample)", name)
    if m:
        stage, block = int(m.group(1)), int(m.group(2) or 0)
        return {0: 1, 1: 2, 2: 3 + block // 3, 3: n_layers - 2}[stage]
    return n_layers - 1   # norm_pre / head / metadata branch


def layer_decay_param_groups(model, lr, decay, weight_decay):
    """AdamW parameter groups with lr * decay**(top - layer); weight decay unchanged on every parameter."""
    groups = {}
    for name, p in model.named_parameters():
        layer = convnext_layer_id(name)
        g = groups.setdefault(layer, {"params": [], "lr": lr * decay ** (13 - layer), "weight_decay": weight_decay})
        g["params"].append(p)
    return [groups[k] for k in sorted(groups)]
