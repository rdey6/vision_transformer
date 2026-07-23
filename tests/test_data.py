"""
Tests for CIFAR-100 dataset pipeline.

Covers:
- Dataset split correctness
- Split disjointness
- Transform behavior
"""

from __future__ import annotations

import torch

from src.data import (
    build_datasets,
    get_class_distribution,
)

def get_test_config():
    """
    Minimal configuration matching project YAML structure.
    """
    return {
        "experiment": {
            "seed": 42,
        },
        "dataset": {
            "root": "data",
            "num_classes": 100,
            "val_size": 5000,
        },
        "input": {
            "image_size": 32,
        },
        "augmentation": {
            "random_crop": True,
            "crop_padding": 4,
            "random_horizontal_flip": True,
            "color_jitter": True,
            "random_erasing": True,
        },
        "normalization": {
            "mean": [
                0.5071,
                0.4867,
                0.4408,
            ],
            "std": [
                0.2675,
                0.2565,
                0.2761,
            ],
        },
    }

def test_dataset_split_sizes():
    """
    Verify train/validation/test sizes.

    Expected:
    - Train: 45000
    - Validation: 5000
    - Test: 10000
    """

    config = get_test_config()


    train_dataset, val_dataset, test_dataset = (
        build_datasets(config)
    )


    assert len(train_dataset) == 45000

    assert len(val_dataset) == 5000

    assert len(test_dataset) == 10000


def test_dataset_split_disjoint():
    """
    Verify no samples overlap between splits.
    """

    config = get_test_config()

    train_dataset, val_dataset, test_dataset = (
        build_datasets(config)
    )


    train_indices = set(
        train_dataset.indices
    )

    val_indices = set(
        val_dataset.indices
    )

    test_indices = set(
        range(
            len(test_dataset)
        )
    )


    assert train_indices.isdisjoint(
        val_indices
    )


    assert train_indices.isdisjoint(
        test_indices
    )


    assert val_indices.isdisjoint(
        test_indices
    )


def test_split_is_stratified():
    """
    Verify class distributions are approximately equal.

    CIFAR-100 has exactly:
    - 500 images/class in training
    """

    config = get_test_config()

    train_dataset, val_dataset, _ = (
        build_datasets(config)
    )


    train_distribution = (
        get_class_distribution(
            train_dataset
        )
    )

    val_distribution = (
        get_class_distribution(
            val_dataset
        )
    )


    for count in train_distribution.values():

        assert abs(count - 450) < 50


    for count in val_distribution.values():

        assert abs(count - 50) < 20


def test_transform_output_shape():
    """
    Verify transformed image dimensions.

    CIFAR-100 images should remain
    compatible with patch embedding.
    """

    config = get_test_config()

    train_dataset, _, _ = (
        build_datasets(config)
    )


    image, label = train_dataset[0]


    assert isinstance(
        image,
        torch.Tensor,
    )


    assert image.shape[0] == 3

    assert image.shape[1] > 0

    assert image.shape[2] > 0


    assert isinstance(
        label,
        int,
    )