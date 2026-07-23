"""
CIFAR-100 transformer models.

Primary model:
    Swin Transformer Tiny from torchvision.

Baseline:
    Custom-sized Vision Transformer from torchvision components.

Both models are trained from scratch without pretrained weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torchvision.models import swin_t
from torchvision.models.vision_transformer import VisionTransformer


def count_parameters(
    model: nn.Module,
) -> int:
    """
    Count trainable parameters.

    Parameters
    ----------
    model : nn.Module
        Neural network.

    Returns
    -------
    int
        Number of trainable parameters.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def initialize_linear(
    layer: nn.Linear,
) -> None:
    """
    Initialize linear layer.

    Parameters
    ----------
    layer : nn.Linear
        Linear layer.
    """

    nn.init.trunc_normal_(
        layer.weight,
        std=0.02,
    )

    if layer.bias is not None:
        nn.init.constant_(
            layer.bias,
            0,
        )


def build_swin_model(
    config: dict,
) -> nn.Module:
    """
    Build CIFAR-100 Swin Transformer.

    Modifications:
    - Removed pretrained weights.
    - Changed patch embedding for CIFAR resolution.
    - Changed classifier to 100 classes.

    Parameters
    ----------
    config : dict
        Configuration dictionary.

    Returns
    -------
    nn.Module
        Swin Transformer model.
    """

    num_classes = config["dataset"]["num_classes"]

    patch_size = config["input"]["patch_size"]

    embed_dim = config["model"]["embed_dim"]

    model = swin_t(
        weights=None,
        num_classes=num_classes,
    )

    # CIFAR patch embedding
    model.features[0][0] = nn.Conv2d(
        in_channels=3,
        out_channels=embed_dim,
        kernel_size=patch_size,
        stride=patch_size,
    )

    model.head = nn.Linear(
        model.head.in_features,
        num_classes,
    )

    initialize_linear(
        model.head,
    )

    return model


def build_vit_model(
    config: dict,
) -> nn.Module:
    """
    Build parameter-matched Vision Transformer.

    Configuration:
    - 32x32 input
    - Patch size 4
    - 384 embedding dimension
    - 8 transformer layers
    - 6 attention heads

    This produces approximately the same parameter
    scale as Swin-T.

    Parameters
    ----------
    config : dict
        Configuration dictionary.

    Returns
    -------
    nn.Module
        Vision Transformer model.
    """

    num_classes = config["dataset"]["num_classes"]

    model = VisionTransformer(
        image_size=32,
        patch_size=4,
        num_layers=8,
        num_heads=6,
        hidden_dim=384,
        mlp_dim=1536,
        dropout=0.1,
        attention_dropout=0.0,
        num_classes=num_classes,
    )

    initialize_linear(
        model.heads.head,
    )

    return model


def build_model(
    config: dict,
) -> nn.Module:
    """
    Construct model from configuration.

    Parameters
    ----------
    config : dict
        Experiment configuration.

    Returns
    -------
    nn.Module
        Selected transformer model.
    """

    name = (
        config["experiment"]["name"]
        .lower()
    )

    if "swin" in name:
        return build_swin_model(config)

    if "vit" in name:
        return build_vit_model(config)

    raise ValueError(
        "Unknown model type."
    )


def verify_model(
    model: nn.Module,
    image_size: int = 32,
    classes: int = 100,
) -> None:
    """
    Verify output tensor shape.

    Parameters
    ----------
    model : nn.Module
        Model.

    image_size : int
        Input image resolution.

    classes : int
        Number of classes.

    Raises
    ------
    RuntimeError
        If output shape is incorrect.
    """

    model.eval()

    images = torch.randn(
        2,
        3,
        image_size,
        image_size,
    )

    with torch.no_grad():
        outputs = model(images)

    expected = (
        2,
        classes,
    )

    if outputs.shape != expected:
        raise RuntimeError(
            f"Expected {expected}, "
            f"received {outputs.shape}"
        )


if __name__ == "__main__":

    swin_config = {
        "experiment": {
            "name": "swin_transformer",
        },
        "dataset": {
            "num_classes": 100,
        },
        "input": {
            "patch_size": 2,
        },
        "model": {
            "embed_dim": 96,
        },
    }

    vit_config = {
        "experiment": {
            "name": "vision_transformer",
        },
        "dataset": {
            "num_classes": 100,
        },
    }

    swin_model = build_model(
        swin_config,
    )

    vit_model = build_model(
        vit_config,
    )

    print(
        "Swin parameters:",
        f"{count_parameters(swin_model):,}",
    )

    print(
        "ViT parameters:",
        f"{count_parameters(vit_model):,}",
    )

    verify_model(
        swin_model,
    )

    verify_model(
        vit_model,
    )

    print(
        "All model checks passed."
    )