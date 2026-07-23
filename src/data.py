"""
Utilities for loading and preprocessing the CIFAR-100 dataset.

This module provides reproducible dataset splitting, configurable data
augmentations, deterministic evaluation transforms, and PyTorch
DataLoaders for training transformer-based image classifiers.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import datasets
from torchvision import transforms


def set_random_seed(seed: int) -> None:
    """
    Set all random seeds for reproducible experiments.

    Parameters
    ----------
    seed : int
        Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """
    Initialize worker seeds for deterministic DataLoader behavior.

    Parameters
    ----------
    worker_id : int
        Worker identifier supplied by PyTorch.
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    config_path : str | Path
        Path to YAML configuration.

    Returns
    -------
    dict
        Configuration dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as yaml_file:
        config = yaml.safe_load(yaml_file)

    return config

def get_normalization(
    config: dict[str, Any],
) -> tuple[list[float], list[float]]:
    """
    Retrieve normalization statistics from config.

    Parameters
    ----------
    config : dict
        Configuration dictionary.

    Returns
    -------
    tuple[list[float], list[float]]
        Mean and standard deviation values.
    """

    normalization = config["normalization"]

    mean = normalization["mean"]
    std = normalization["std"]

    return mean, std

def build_train_transform(
    config: dict[str, Any],
) -> transforms.Compose:
    """
    Create training data augmentation pipeline.

    Parameters
    ----------
    config : dict
        Configuration dictionary.

    Returns
    -------
    transforms.Compose
        Training transforms.
    """

    augmentation = config["augmentation"]

    image_size = config["input"]["image_size"]

    mean, std = get_normalization(
        config,
    )

    transform_list: list[Any] = []

    if augmentation["random_crop"]:
        transform_list.append(
            transforms.RandomCrop(
                size=image_size,
                padding=augmentation["crop_padding"],
            )
        )

    if augmentation["horizontal_flip"]:
        transform_list.append(
            transforms.RandomHorizontalFlip()
        )

    if augmentation["color_jitter"]:
        transform_list.append(
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
            )
        )

    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean,
                std=std,
            ),
        ]
    )

    if augmentation["random_erasing"]:
        transform_list.append(
            transforms.RandomErasing(
                p=0.25,
                scale=(0.02, 0.2),
                ratio=(0.3, 3.3),
            )
        )

    return transforms.Compose(
        transform_list
    )


def build_eval_transform(
    config: dict[str, Any],
) -> transforms.Compose:
    """
    Create deterministic evaluation pipeline.

    Parameters
    ----------
    config : dict
        Configuration dictionary.

    Returns
    -------
    transforms.Compose
        Validation/test transforms.
    """

    mean, std = get_normalization(
        config,
    )

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean,
                std=std,
            ),
        ]
    )


def create_train_validation_split(
    dataset: datasets.CIFAR100,
    seed: int,
) -> tuple[list[int], list[int]]:
    """
    Create a reproducible stratified train-validation split.

    The official CIFAR-100 training set contains 50,000 images.
    This function produces:

    - 45,000 training samples
    - 5,000 validation samples

    Parameters
    ----------
    dataset : datasets.CIFAR100
        Training dataset.

    seed : int
        Random seed.

    Returns
    -------
    tuple[list[int], list[int]]
        Training indices and validation indices.
    """
    labels = np.asarray(dataset.targets)

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=5000,
        random_state=seed,
    )

    train_indices, validation_indices = next(
        splitter.split(np.zeros(len(labels)), labels)
    )

    return train_indices.tolist(), validation_indices.tolist()


def verify_disjoint_splits(
    train_indices: list[int],
    validation_indices: list[int],
    test_size: int,
) -> None:
    """
    Verify dataset splits are disjoint.

    Parameters
    ----------
    train_indices : list[int]
        Training indices.

    validation_indices : list[int]
        Validation indices.

    test_size : int
        Number of official test samples.

    Raises
    ------
    ValueError
        If overlap exists.
    """

    train_set = set(train_indices)

    validation_set = set(validation_indices)

    if train_set.intersection(validation_set):
        raise ValueError(
            "Train and validation sets overlap."
        )

    test_indices = set(
        range(
            50000,
            50000 + test_size,
        )
    )

    if train_set.intersection(test_indices):
        raise ValueError(
            "Train and test sets overlap."
        )

    if validation_set.intersection(test_indices):
        raise ValueError(
            "Validation and test sets overlap."
        )

def build_datasets(
    config: dict[str, Any],
) -> tuple[Subset, Subset, datasets.CIFAR100]:
    """
    Build training, validation, and testing datasets.

    Parameters
    ----------
    config : dict
        Experiment configuration.

    Returns
    -------
    tuple
        Training, validation, and testing datasets.
    """
    root = config["dataset"]["root"]
    seed = config["experiment"]["seed"]

    train_transform = build_train_transform(config)
    eval_transform = build_eval_transform(config,)

    train_dataset_full = datasets.CIFAR100(
        root=root,
        train=True,
        download=True,
        transform=train_transform,
    )

    validation_dataset_full = datasets.CIFAR100(
        root=root,
        train=True,
        download=False,
        transform=eval_transform,
    )

    test_dataset = datasets.CIFAR100(
        root=root,
        train=False,
        download=True,
        transform=eval_transform,
    )

    train_indices, validation_indices = create_train_validation_split(
        train_dataset_full,
        seed,
    )

    verify_disjoint_splits(
    train_indices,
    validation_indices,
    len(test_dataset),
    )

    train_dataset = Subset(
        train_dataset_full,
        train_indices,
    )

    validation_dataset = Subset(
        validation_dataset_full,
        validation_indices,
    )

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
    )


def build_dataloaders(
    config: dict[str, Any],
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Construct DataLoaders for all dataset splits.

    Parameters
    ----------
    config : dict
        Experiment configuration.

    Returns
    -------
    tuple
        Train, validation and test DataLoaders.
    """
    batch_size = config["training"]["batch_size"]

    train_dataset, validation_dataset, test_dataset = build_datasets(
        config
    )

    generator = torch.Generator()
    generator.manual_seed(config["experiment"]["seed"])

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=True,
    )

    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=True,
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
    )


def get_class_names(
    root: str = "data",
) -> list[str]:
    """
    Return CIFAR-100 class names.

    Parameters
    ----------
    root : str, default="data"
        Dataset root directory.

    Returns
    -------
    list[str]
        List containing all class names.
    """
    dataset = datasets.CIFAR100(
        root=root,
        train=True,
        download=True,
    )

    return dataset.classes

def get_class_distribution(dataset):
    """
    Return class counts for a dataset split.
    """

    labels = []

    for index in dataset.indices:
        labels.append(
            dataset.dataset.targets[index]
        )

    distribution = {}

    for label in labels:
        distribution[label] = (
            distribution.get(label, 0)
            + 1
        )

    return distribution

def main() -> None:
    """
    Quick sanity check for the data pipeline.
    """
    config = load_config(
        "configs/primary.yaml",
    )

    set_random_seed(
        config["experiment"]["seed"],
    )

    train_loader, validation_loader, test_loader = (
        build_dataloaders(config)
    )

    print(
        f"Training samples: {len(train_loader.dataset):,}"
    )
    print(
        f"Validation samples: {len(validation_loader.dataset):,}"
    )
    print(
        f"Testing samples: {len(test_loader.dataset):,}"
    )

    images, labels = next(iter(train_loader))

    print(f"Image batch shape : {images.shape}")
    print(f"Label batch shape : {labels.shape}")


if __name__ == "__main__":
    main()