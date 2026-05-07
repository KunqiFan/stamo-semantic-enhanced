from __future__ import annotations

from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error


def macro_f1_from_rows(rows: list[dict], target: str) -> float:
    y_true = [row["gold"][target] for row in rows]
    y_pred = [row["pred"][target] for row in rows]
    return float(f1_score(y_true, y_pred, average="macro"))


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }
