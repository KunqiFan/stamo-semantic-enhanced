from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def load_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_delta_z(latent_dir: Path, sample_id: str) -> np.ndarray:
    path = latent_dir / f"{sample_id}.npz"
    return np.load(path)["delta_z"]


def can_stratify(labels: list[str]) -> bool:
    counts = Counter(labels)
    return len(labels) >= 4 and all(count >= 2 for count in counts.values())


def train_text_only(rows: list[dict], target: str) -> None:
    texts = [row.get("caption", "") or row.get("process_text", "") for row in rows]
    labels = [row["labels"][target] for row in rows]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    clf = LogisticRegression(max_iter=1000)

    if can_stratify(labels):
        x_train, x_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        x_train_vec = vectorizer.fit_transform(x_train)
        x_test_vec = vectorizer.transform(x_test)
        clf.fit(x_train_vec, y_train)
        preds = clf.predict(x_test_vec)
        print(classification_report(y_test, preds, digits=4))
        return

    print("Dataset too small for a stratified split; fitting on all rows for smoke testing.")
    x_vec = vectorizer.fit_transform(texts)
    clf.fit(x_vec, labels)
    preds = clf.predict(x_vec)
    print(classification_report(labels, preds, digits=4))


def train_delta_z_only(rows: list[dict], target: str, latent_dir: Path) -> None:
    x = np.stack([load_delta_z(latent_dir, row["sample_id"]) for row in rows])
    y = [row["labels"][target] for row in rows]
    clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=200, random_state=42)

    if can_stratify(y):
        idx = np.arange(len(rows))
        train_idx, test_idx = train_test_split(
            idx, test_size=0.2, random_state=42, stratify=y
        )
        clf.fit(x[train_idx], np.array(y)[train_idx])
        preds = clf.predict(x[test_idx])
        print(classification_report(np.array(y)[test_idx], preds, digits=4))
        return

    print("Dataset too small for a stratified split; fitting on all rows for smoke testing.")
    clf.fit(x, y)
    preds = clf.predict(x)
    print(classification_report(y, preds, digits=4))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--setting", choices=["text_only", "delta_z_only"], required=True)
    parser.add_argument("--target", type=str, default="object_motion")
    parser.add_argument("--latent_dir", type=Path, default=Path("data/interim/latents"))
    args = parser.parse_args()

    rows = load_manifest(args.manifest)
    if args.setting == "text_only":
        train_text_only(rows, args.target)
    else:
        train_delta_z_only(rows, args.target, args.latent_dir)


if __name__ == "__main__":
    main()
