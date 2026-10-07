"""Backbone construction shared by training and evaluation scripts."""
import timm

# timm families whose position embeddings depend on the input size (resized when img_size is passed).
_SIZED_FAMILIES = ("vit_", "eva02_", "deit", "beit")


def build_model(name: str, num_classes: int, img_size: int, pretrained: bool = True, grad_checkpointing: bool = False):
    """timm model with a fresh `num_classes` head; transformers are built for `img_size` inputs."""
    kwargs = {"img_size": img_size} if name.startswith(_SIZED_FAMILIES) else {}
    model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes, **kwargs)
    if grad_checkpointing:
        model.set_grad_checkpointing(True)
    return model
