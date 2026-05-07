"""Batch-generate text bridge descriptions for all samples in a manifest.

Reads a JSONL manifest, generates template_text and enriched_text for each sample
using gold labels (or predicted semantics), and writes an augmented manifest.

Usage:
    py scripts/build_bridge_texts.py \
        --manifest data/processed/droid_process_chain/train.jsonl \
        --out data/interim/droid_process_chain/bridge_texts/train.jsonl

    # With predicted semantics instead of gold:
    py scripts/build_bridge_texts.py \
        --manifest data/processed/droid_process_chain/train.jsonl \
        --predicted_semantics data/interim/droid_process_chain/semantics/train_cv_predicted.jsonl \
        --out data/interim/droid_process_chain/bridge_texts/train.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stamo_bridge.semantics.interface import SEMANTIC_FIELDS
from stamo_bridge.semantics.text_bridge import (
    generate_enriched_text,
    generate_llm_prompt,
    generate_template_text,
)


def load_predicted_semantics(path: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                result[row["sample_id"]] = row.get("predicted_semantics", {})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--predicted_semantics", type=Path, default=None,
                        help="Use predicted semantics instead of gold labels")
    parser.add_argument("--export_llm_prompts", type=Path, default=None,
                        help="Also export LLM prompts to this file for offline generation")
    args = parser.parse_args()

    predicted = load_predicted_semantics(args.predicted_semantics) if args.predicted_semantics else None

    args.out.parent.mkdir(parents=True, exist_ok=True)
    prompts_file = None
    if args.export_llm_prompts:
        args.export_llm_prompts.parent.mkdir(parents=True, exist_ok=True)
        prompts_file = args.export_llm_prompts.open("w", encoding="utf-8")

    count = 0
    with args.manifest.open("r", encoding="utf-8") as fin, \
         args.out.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            sid = row["sample_id"]

            if predicted and sid in predicted:
                sem_labels = predicted[sid]
            else:
                sem_labels = {f: row["labels"][f] for f in SEMANTIC_FIELDS}

            row["template_text"] = generate_template_text(sem_labels)
            row["enriched_text"] = generate_enriched_text(sem_labels, stage=None)

            fout.write(json.dumps(row) + "\n")

            if prompts_file:
                prompt_row = {
                    "sample_id": sid,
                    "prompt": generate_llm_prompt(sem_labels, stage=None),
                }
                prompts_file.write(json.dumps(prompt_row) + "\n")

            count += 1

    if prompts_file:
        prompts_file.close()
        print(f"Exported {count} LLM prompts to {args.export_llm_prompts}")

    print(f"Generated bridge texts for {count} samples -> {args.out}")


if __name__ == "__main__":
    main()
