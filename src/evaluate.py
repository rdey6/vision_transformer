"""
Evaluation script for CIFAR-100 transformer experiments.

Generates:
- Test accuracy
- Top-5 accuracy
- Macro F1
- Per-class F1 metrics
- Confusion matrix
- Error analysis examples
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets

from data import build_datasets
from metrics import (
    calculate_class_metrics,
    calculate_confusion_matrix,
    calculate_top_k_accuracy,
    collect_predictions,
    evaluate_predictions,
    find_common_confusions,
    get_best_and_worst_classes,
    get_misclassified_examples,
)
from models import build_model


def load_config(
    path: str,
) -> dict:
    """
    Load YAML configuration.

    Parameters
    ----------
    path : str
        Configuration path.

    Returns
    -------
    dict
        Configuration dictionary.
    """

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
) -> torch.nn.Module:
    """
    Load trained weights.

    Parameters
    ----------
    model : torch.nn.Module
        Model.

    checkpoint_path : str
        Checkpoint path.

    device : torch.device
        Device.

    Returns
    -------
    torch.nn.Module
        Loaded model.
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return model


def get_class_names() -> list[str]:
    """
    Get CIFAR-100 class names.

    Returns
    -------
    list[str]
        Class labels.
    """

    dataset = datasets.CIFAR100(
        root="data",
        train=False,
        download=True,
    )

    return dataset.classes


def denormalize(
    image: torch.Tensor,
    mean: list[float],
    std: list[float],
) -> torch.Tensor:
    """
    Convert normalized image back to display format.

    Parameters
    ----------
    image : torch.Tensor
        Normalized tensor.

    mean : list[float]
        Normalization mean.

    std : list[float]
        Normalization std.

    Returns
    -------
    torch.Tensor
        Image tensor.
    """

    mean_tensor = torch.tensor(
        mean,
        device=image.device,
    )

    std_tensor = torch.tensor(
        std,
        device=image.device,
    )

    image = image.permute(
        1,
        2,
        0,
    )

    image = (
        image * std_tensor
        + mean_tensor
    )

    return torch.clamp(
        image,
        0,
        1,
    )


def save_confusion_matrix(
    matrix: np.ndarray,
    output_path: str,
) -> None:
    """
    Save confusion matrix plot.

    Parameters
    ----------
    matrix : np.ndarray
        Confusion matrix.

    output_path : str
        Output file.
    """

    plt.figure(
        figsize=(12, 10),
    )

    plt.imshow(
        matrix,
        interpolation="nearest",
    )

    plt.title(
        "CIFAR-100 Confusion Matrix"
    )

    plt.xlabel(
        "Predicted Class"
    )

    plt.ylabel(
        "True Class"
    )

    plt.colorbar()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def save_error_examples(
    examples: dict,
    class_names: list[str],
    mean: list[float],
    std: list[float],
    output_path: str,
) -> None:
    """
    Save confident incorrect predictions.

    Parameters
    ----------
    examples : dict
        Misclassified examples.

    class_names : list[str]
        CIFAR-100 labels.

    mean : list[float]
        Normalization mean.

    std : list[float]
        Normalization std.

    output_path : str
        Output file.
    """

    images = examples["images"]

    targets = examples["targets"]

    predictions = examples["predictions"]

    probabilities = examples["probabilities"]


    plt.figure(
        figsize=(12, 12),
    )


    for index in range(
        len(images)
    ):

        image = denormalize(
            images[index],
            mean,
            std,
        )


        plt.subplot(
            3,
            4,
            index + 1,
        )


        plt.imshow(
            image.cpu().numpy()
        )


        plt.axis(
            "off"
        )


        plt.title(
            (
                f"True: "
                f"{class_names[targets[index]]}\n"
                f"Pred: "
                f"{class_names[predictions[index]]}\n"
                f"Prob: "
                f"{probabilities[index]:.3f}"
            ),
            fontsize=8,
        )


    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def main() -> None:
    """
    Main evaluation function.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
    )

    args = parser.parse_args()


    config = load_config(
        args.config
    )


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    _, _, test_dataset = build_datasets(
        config
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )


    model = build_model(
        config
    )


    model = load_checkpoint(
        model,
        args.checkpoint,
        device,
    )


    model.to(device)


    images, targets, predictions, probabilities, logits = (
        collect_predictions(
            model,
            test_loader,
            device,
        )
    )


    metrics = evaluate_predictions(
        targets,
        predictions,
    )


    top5_accuracy = calculate_top_k_accuracy(
    logits,
    targets,
    k=5,
    )


    print("\nTest Results")
    print("----------------------")
    print(
        f"Top-1 Accuracy: "
        f"{metrics['accuracy']:.2f}%"
    )

    print(
        f"Macro F1: "
        f"{metrics['macro_f1']:.4f}"
    )

    print(
        f"Top-5 Accuracy: "
        f"{top5_accuracy:.2f}%"
    )


    class_names = get_class_names()


    class_metrics = calculate_class_metrics(
        targets,
        predictions,
        class_names,
    )


    best, worst = get_best_and_worst_classes(
        class_metrics,
    )


    print("\nHighest F1 Classes")

    for item in best:
        print(item)


    print("\nLowest F1 Classes")

    for item in worst:
        print(item)


    confusion = calculate_confusion_matrix(
        targets,
        predictions,
    )


    print("\nCommon Confusions")

    for item in find_common_confusions(
        confusion,
        class_names,
        count=3,
    ):
        print(item)


    output_dir = Path(
        "figures"
    )

    output_dir.mkdir(
        exist_ok=True,
    )


    save_confusion_matrix(
        confusion,
        str(
            output_dir
            /
            "confusion_matrix.png"
        ),
    )


    examples = get_misclassified_examples(
        images,
        targets,
        predictions,
        probabilities,
        count=12,
    )


    save_error_examples(
        examples,
        class_names,
        config["normalization"]["mean"],
        config["normalization"]["std"],
        str(
            output_dir
            /
            "error_examples.png"
        ),
    )


if __name__ == "__main__":
    main()