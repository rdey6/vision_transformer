"""
Tests for transformer model implementations.

Covers:
- Forward pass correctness
- Output dimensions
- Patch compatibility
- Configuration validation
"""

from __future__ import annotations

import pytest
import torch

from src.models import (
    build_model,
    count_parameters,
)


@pytest.fixture
def swin_config():
    """
    Minimal Swin Transformer config.
    """

    return {
        "model": {
            "name": "swin",
            "num_classes": 100,
            "img_size": 32,
            "patch_size": 4,
            "window_size": 4,
            "embed_dim": 96,
            "depths": [
                2,
                2,
                6,
                2,
            ],
            "num_heads": [
                3,
                6,
                12,
                24,
            ],
            "dropout": 0.0,
        }
    }


@pytest.fixture
def vit_config():
    """
    Minimal Vision Transformer config.
    """

    return {
        "model": {
            "name": "vit",
            "num_classes": 100,
            "img_size": 32,
            "patch_size": 4,
            "embed_dim": 384,
            "depth": 6,
            "num_heads": 6,
            "dropout": 0.0,
        }
    }


def test_swin_output_shape(
    swin_config,
):
    """
    Verify Swin output shape.

    Expected:
    [batch_size, 100]
    """

    model = build_model(
        swin_config
    )


    images = torch.randn(
        4,
        3,
        32,
        32,
    )


    outputs = model(
        images
    )


    assert outputs.shape == (
        4,
        100,
    )


def test_vit_output_shape(
    vit_config,
):
    """
    Verify ViT output shape.

    Expected:
    [batch_size, 100]
    """

    model = build_model(
        vit_config
    )


    images = torch.randn(
        4,
        3,
        32,
        32,
    )


    outputs = model(
        images
    )


    assert outputs.shape == (
        4,
        100,
    )


def test_models_have_parameters(
    swin_config,
    vit_config,
):
    """
    Verify models contain trainable parameters.
    """

    swin = build_model(
        swin_config
    )

    vit = build_model(
        vit_config
    )


    swin_params = count_parameters(
        swin
    )

    vit_params = count_parameters(
        vit
    )


    assert swin_params > 0

    assert vit_params > 0


def test_patch_size_validation(
    vit_config,
):
    """
    Verify invalid patch size raises error.

    Patch size must divide image size.
    """

    vit_config["model"]["patch_size"] = 7


    with pytest.raises(
        ValueError
    ):

        build_model(
            vit_config
        )


def test_model_accepts_cifar_input(
    swin_config,
    vit_config,
):
    """
    Verify models accept CIFAR-100 resolution.

    CIFAR-100:
    3 x 32 x 32
    """

    models = [
        build_model(
            swin_config
        ),
        build_model(
            vit_config
        ),
    ]


    input_tensor = torch.randn(
        2,
        3,
        32,
        32,
    )


    for model in models:

        output = model(
            input_tensor
        )

        assert output.size(
            0
        ) == 2

        assert output.size(
            1
        ) == 100