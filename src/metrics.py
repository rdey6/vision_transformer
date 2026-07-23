"""
Evaluation metrics for CIFAR-100 transformer experiments.

Includes:
- Accuracy metrics
- Top-k accuracy
- Per-class F1 analysis
- Confusion analysis
- Misclassification analysis
- Metric export utilities
"""

from __future__ import annotations

import csv
import json
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def calculate_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Calculate top-1 accuracy.

    Parameters
    ----------
    predictions : torch.Tensor
        Model logits.

    targets : torch.Tensor
        Ground truth labels.

    Returns
    -------
    float
        Accuracy percentage.
    """

    predicted_labels = torch.argmax(
        predictions,
        dim=1,
    )

    return (
        (predicted_labels == targets)
        .float()
        .mean()
        .item()
        * 100
    )


def calculate_top_k_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k: int = 5,
) -> float:
    """
    Calculate top-k accuracy.

    Parameters
    ----------
    predictions : torch.Tensor
        Model logits.

    targets : torch.Tensor
        Ground truth labels.

    k : int
        Number of predictions considered.

    Returns
    -------
    float
        Top-k accuracy percentage.
    """

    _, top_k_predictions = torch.topk(
        predictions,
        k=k,
        dim=1,
    )

    correct = (
        top_k_predictions
        == targets.unsqueeze(1)
    )

    return (
        correct.any(dim=1)
        .float()
        .mean()
        .item()
        * 100
    )


def collect_predictions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Collect predictions, probabilities, and logits.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.

    dataloader : DataLoader
        Evaluation dataloader.

    device : torch.device
        Device.

    Returns
    -------
    tuple
        Images, targets, predictions,
        probabilities, logits.
    """

    model.eval()

    all_images = []
    all_targets = []
    all_predictions = []
    all_probabilities = []
    all_logits = []

    with torch.no_grad():

        for images, targets in dataloader:

            images = images.to(device)

            targets = targets.to(device)


            outputs = model(
                images
            )


            probabilities = torch.softmax(
                outputs,
                dim=1,
            )


            predictions = torch.argmax(
                outputs,
                dim=1,
            )


            all_images.append(
                images.cpu()
            )

            all_targets.append(
                targets.cpu()
            )

            all_predictions.append(
                predictions.cpu()
            )

            all_probabilities.append(
                probabilities.cpu()
            )

            all_logits.append(
                outputs.cpu()
            )


    return (
        torch.cat(all_images),
        torch.cat(all_targets),
        torch.cat(all_predictions),
        torch.cat(all_probabilities),
        torch.cat(all_logits),
    )


def evaluate_predictions(
    targets: torch.Tensor,
    predictions: torch.Tensor,
) -> dict[str, float]:
    """
    Compute summary metrics.

    Parameters
    ----------
    targets : torch.Tensor
        True labels.

    predictions : torch.Tensor
        Predicted labels.

    Returns
    -------
    dict
        Accuracy and macro F1.
    """

    return {
        "accuracy": accuracy_score(
            targets.numpy(),
            predictions.numpy(),
        )
        * 100,

        "macro_f1": f1_score(
            targets.numpy(),
            predictions.numpy(),
            average="macro",
        ),
    }


def calculate_class_metrics(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    class_names: list[str],
) -> list[dict[str, Any]]:
    """
    Calculate class-level metrics.

    Parameters
    ----------
    targets : torch.Tensor
        True labels.

    predictions : torch.Tensor
        Predicted labels.

    class_names : list[str]
        Class names.

    Returns
    -------
    list[dict]
        Per-class results.
    """

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            targets.numpy(),
            predictions.numpy(),
            labels=range(
                len(class_names)
            ),
            zero_division=0,
        )
    )


    results = []

    for index, name in enumerate(class_names):

        results.append(
            {
                "class": name,
                "precision": float(
                    precision[index]
                ),
                "recall": float(
                    recall[index]
                ),
                "f1": float(
                    f1[index]
                ),
            }
        )


    return results


def get_best_and_worst_classes(
    class_metrics: list[dict[str, Any]],
    count: int = 5,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """
    Find highest and lowest F1 classes.

    Parameters
    ----------
    class_metrics : list[dict]
        Class metrics.

    count : int
        Number of classes.

    Returns
    -------
    tuple
        Best and worst classes.
    """

    sorted_metrics = sorted(
        class_metrics,
        key=lambda item: item["f1"],
        reverse=True,
    )

    return (
        sorted_metrics[:count],
        sorted_metrics[-count:],
    )


def calculate_confusion_matrix(
    targets: torch.Tensor,
    predictions: torch.Tensor,
) -> np.ndarray:
    """
    Generate confusion matrix.

    Parameters
    ----------
    targets : torch.Tensor
        True labels.

    predictions : torch.Tensor
        Predicted labels.

    Returns
    -------
    np.ndarray
        Confusion matrix.
    """

    return confusion_matrix(
        targets.numpy(),
        predictions.numpy(),
    )


def find_common_confusions(
    confusion: np.ndarray,
    class_names: list[str],
    count: int = 10,
) -> list[dict[str, Any]]:
    """
    Find most common incorrect predictions.

    Parameters
    ----------
    confusion : np.ndarray
        Confusion matrix.

    class_names : list[str]
        Class names.

    count : int
        Number of confusion pairs.

    Returns
    -------
    list[dict]
        Confusion pairs.
    """

    matrix = confusion.copy()

    np.fill_diagonal(
        matrix,
        0,
    )

    indices = np.argsort(
        matrix.flatten()
    )[::-1]


    results = []


    for index in indices:

        true_label = (
            index
            //
            matrix.shape[0]
        )

        predicted_label = (
            index
            %
            matrix.shape[0]
        )

        value = matrix[
            true_label,
            predicted_label,
        ]


        if value == 0:
            continue


        results.append(
            {
                "true_class": class_names[
                    true_label
                ],
                "predicted_class": class_names[
                    predicted_label
                ],
                "count": int(value),
            }
        )


        if len(results) == count:
            break


    return results


def get_misclassified_examples(
    images: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    probabilities: torch.Tensor,
    count: int = 12,
) -> dict[str, torch.Tensor]:
    """
    Select confident incorrect predictions.

    Parameters
    ----------
    images : torch.Tensor
        Images.

    targets : torch.Tensor
        True labels.

    predictions : torch.Tensor
        Predictions.

    probabilities : torch.Tensor
        Prediction probabilities.

    count : int
        Number of examples.

    Returns
    -------
    dict
        Selected examples.
    """

    incorrect = (
        targets != predictions
    )

    indices = torch.where(
        incorrect
    )[0]


    confidence = probabilities.max(
        dim=1
    ).values


    selected = indices[
        torch.argsort(
            confidence[indices],
            descending=True,
        )[:count]
    ]


    return {
        "images": images[selected],
        "targets": targets[selected],
        "predictions": predictions[selected],
        "probabilities": confidence[selected],
    }


def save_metrics_json(
    metrics: dict[str, Any],
    path: str,
) -> None:
    """
    Save summary metrics.

    Parameters
    ----------
    metrics : dict
        Metrics.

    path : str
        Output path.
    """

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )


def save_class_metrics_csv(
    metrics: list[dict[str, Any]],
    path: str,
) -> None:
    """
    Save class metrics.

    Parameters
    ----------
    metrics : list[dict]
        Class metrics.

    path : str
        Output path.
    """

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "class",
                "precision",
                "recall",
                "f1",
            ],
        )

        writer.writeheader()

        writer.writerows(
            metrics
        )