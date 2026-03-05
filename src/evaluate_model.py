"""Model evaluation utilities for classification tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.utils import ensure_directory



def evaluate_classification(y_true: pd.Series, y_pred: pd.Series) -> dict[str, Any]:
    """Compute standard classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }



def save_confusion_matrix_plot(
    y_true: pd.Series,
    y_pred: pd.Series,
    labels: list[str],
    output_path: Path,
) -> Path:
    """Save a confusion matrix figure."""
    ensure_directory(output_path.parent)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", ax=ax, xticks_rotation=45, colorbar=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path



def save_model_comparison_plot(reports: dict[str, dict[str, Any]], output_path: Path) -> Path:
    """Save model comparison chart (accuracy/precision/recall/F1)."""
    ensure_directory(output_path.parent)

    rows: list[dict[str, Any]] = []
    for name, metrics in reports.items():
        rows.append(
            {
                "model": name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
        )

    df = pd.DataFrame(rows)
    melt = df.melt(id_vars="model", var_name="metric", value_name="score")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=melt, x="model", y="score", hue="metric", ax=ax)
    ax.set_title("Model Performance Comparison")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
