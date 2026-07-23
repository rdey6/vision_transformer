"""
Tests for model parameter requirements.

Covers:
- Parameter count calculation
- ViT/Swin parameter matching
- Trainable parameter validation
"""

from __future__ import annotations

import pytest

from src.models import (
    build_model,
    count_parameters,
)


@pytest.fixture
def primary_config():
    """
    Primary Swin Transformer configuration.
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
    ViT baseline configuration.

    This should be adjusted so that
    parameters are within 10% of Swin.
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


def test_parameter_count_positive(
    primary_config,
    vit_config,
):
    """
    Verify both models have parameters.
    """

    primary_model = build_model(
        primary_config
    )

    vit_model = build_model(
        vit_config
    )


    primary_params = count_parameters(
        primary_model
    )

    vit_params = count_parameters(
        vit_model
    )


    assert primary_params > 0

    assert vit_params > 0



def test_all_parameters_trainable(
    primary_config,
    vit_config,
):
    """
    Verify models are trained from scratch.

    No frozen parameters should exist.
    """

    models = [
        build_model(primary_config),
        build_model(vit_config),
    ]


    for model in models:

        frozen_parameters = [
            parameter
            for parameter in model.parameters()
            if not parameter.requires_grad
        ]


        assert len(
            frozen_parameters
        ) == 0



def test_vit_parameter_matching(
    primary_config,
    vit_config,
):
    """
    Verify ViT parameter count is within
    10% of primary model.

    Requirement:
    0.9 * primary <= ViT <= 1.1 * primary
    """

    primary_model = build_model(
        primary_config
    )

    vit_model = build_model(
        vit_config
    )


    primary_params = count_parameters(
        primary_model
    )

    vit_params = count_parameters(
        vit_model
    )


    lower_bound = (
        primary_params
        *
        0.9
    )

    upper_bound = (
        primary_params
        *
        1.1
    )


    assert lower_bound <= vit_params <= upper_bound, (
        "\nParameter mismatch!\n"
        f"Primary model parameters: "
        f"{primary_params:,}\n"
        f"ViT parameters: "
        f"{vit_params:,}\n"
        f"Allowed range: "
        f"{lower_bound:,.0f} - "
        f"{upper_bound:,.0f}"
    )


def test_parameter_count_consistency(
    primary_config,
):
    """
    Verify parameter counting is deterministic.
    """

    model = build_model(
        primary_config
    )


    count_one = count_parameters(
        model
    )

    count_two = count_parameters(
        model
    )


    assert count_one == count_two