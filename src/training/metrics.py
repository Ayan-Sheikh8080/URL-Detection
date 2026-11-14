"""Metrics for binary phishing classification."""

from sklearn.metrics import precision_recall_fscore_support


def phishing_precision_recall_f1(
    labels: list[int],
    predictions: list[int],
) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        pos_label=0,
        zero_division=0,
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
