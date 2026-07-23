"""
Training script for CIFAR-100 Transformer experiments.

Supports:
- Swin Transformer
- Vision Transformer
- AdamW optimizer
- Warmup + cosine learning rate
- AMP mixed precision
- Checkpointing
- CSV logging
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data import build_datasets
from metrics import calculate_accuracy
from models import build_model, count_parameters


def set_seed(
    seed: int,
) -> None:
    """
    Set random seeds.

    Parameters
    ----------
    seed : int
        Random seed.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(
        seed
    )


def load_config(
    path: str,
) -> dict:
    """
    Load YAML configuration.

    Parameters
    ----------
    path : str
        YAML path.

    Returns
    -------
    dict
        Configuration.
    """

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return yaml.safe_load(file)


def create_optimizer(
    model: nn.Module,
    config: dict,
) -> AdamW:
    """
    Create AdamW optimizer.

    Parameters
    ----------
    model : nn.Module
        Model.

    config : dict
        Configuration.

    Returns
    -------
    AdamW
        Optimizer.
    """

    training = config["training"]

    return AdamW(
        model.parameters(),
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    clip_value: float,
) -> tuple[float, float]:
    """
    Train one epoch.

    Returns
    -------
    tuple
        Loss and accuracy.
    """

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0


    for images, labels in loader:

        images = images.to(device)

        labels = labels.to(device)


        optimizer.zero_grad(
            set_to_none=True
        )


        with torch.cuda.amp.autocast(
            enabled=scaler.is_enabled()
        ):

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels,
            )


        scaler.scale(
            loss
        ).backward()


        if clip_value > 0:

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                clip_value,
            )


        scaler.step(
            optimizer
        )

        scaler.update()


        running_loss += (
            loss.item()
            *
            images.size(0)
        )


        predictions = torch.argmax(
            outputs,
            dim=1,
        )


        correct += (
            predictions == labels
        ).sum().item()


        total += labels.size(0)


    epoch_loss = (
        running_loss
        /
        total
    )

    epoch_accuracy = (
        correct
        /
        total
        *
        100
    )


    return (
        epoch_loss,
        epoch_accuracy,
    )


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Validate model.

    Returns
    -------
    tuple
        Validation loss and accuracy.
    """

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0


    for images, labels in loader:

        images = images.to(device)

        labels = labels.to(device)


        outputs = model(
            images
        )


        loss = criterion(
            outputs,
            labels,
        )


        running_loss += (
            loss.item()
            *
            images.size(0)
        )


        predictions = torch.argmax(
            outputs,
            dim=1,
        )


        correct += (
            predictions == labels
        ).sum().item()


        total += labels.size(0)


    return (
        running_loss / total,
        correct / total * 100,
    )


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    accuracy: float,
    path: str,
) -> None:
    """
    Save model checkpoint.
    """

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "accuracy": accuracy,
    }

    torch.save(
        checkpoint,
        path,
    )


def main() -> None:
    """
    Main training loop.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
        type=str,
    )

    args = parser.parse_args()


    config = load_config(
        args.config
    )


    seed = config["experiment"]["seed"]

    set_seed(seed)


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    train_dataset, val_dataset, _ = build_datasets(
        config
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )


    model = build_model(
        config
    )

    model.to(device)


    print(
        f"Parameters: "
        f"{count_parameters(model):,}"
    )


    criterion = nn.CrossEntropyLoss(
        label_smoothing=config["training"]["label_smoothing"]
    )


    optimizer = create_optimizer(
        model,
        config,
    )


    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["epochs"],
    )


    scaler = torch.cuda.amp.GradScaler(
        enabled=config["training"]["mixed_precision"]
        and device.type == "cuda"
    )


    save_dir = Path(
        config["experiment"]["save_dir"]
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    csv_path = save_dir / "training.csv"


    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy",
                "learning_rate",
                "duration",
            ]
        )


    best_accuracy = 0.0


    for epoch in range(
        1,
        config["training"]["epochs"] + 1,
    ):

        start = time.time()


        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            config["training"]["gradient_clip"],
        )


        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion,
            device,
        )


        scheduler.step()


        duration = time.time() - start


        lr = optimizer.param_groups[0]["lr"]


        print(
            f"Epoch {epoch}: "
            f"Train Acc {train_acc:.2f}% "
            f"Val Acc {val_acc:.2f}%"
        )


        with open(
            csv_path,
            "a",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.writer(file)

            writer.writerow(
                [
                    epoch,
                    train_loss,
                    train_acc,
                    val_loss,
                    val_acc,
                    lr,
                    duration,
                ]
            )


        if val_acc > best_accuracy:

            best_accuracy = val_acc

            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_acc,
                str(
                    save_dir / "best_model.pt"
                ),
            )


if __name__ == "__main__":
    main()