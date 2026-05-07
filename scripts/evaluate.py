from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predicted process semantics or stages")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--target", type=str, required=True, help="Target semantic field to evaluate")
    args = parser.parse_args()

    y_true = []
    y_pred = []

    with args.predictions.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            
            # Handle nested semantics format from train_compact_semantics.py
            if "gold_semantics" in row and "predicted_semantics" in row:
                if args.target in row["gold_semantics"]:
                    y_true.append(row["gold_semantics"][args.target])
                    y_pred.append(row["predicted_semantics"][args.target])
            # Handle flat format
            elif "gold" in row and "pred" in row:
                y_true.append(row["gold"])
                y_pred.append(row["pred"])

    if not y_true:
        print(f"Error: Target '{args.target}' not found in predictions file.")
        return

    print(f"=== Evaluation for: {args.target} ===")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    
    print("\nConfusion Matrix:")
    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Print a formatted confusion matrix
    print(f"{'':>15} " + " ".join([f"{label:>10}" for label in labels]))
    for i, true_label in enumerate(labels):
        print(f"{true_label:>15} " + " ".join([f"{val:>10}" for val in cm[i]]))

if __name__ == "__main__":
    main()
