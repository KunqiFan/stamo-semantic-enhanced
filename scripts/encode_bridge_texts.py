"""Encode bridge texts into dense vectors using Sentence-BERT.

Reads a bridge-text-augmented manifest (from build_bridge_texts.py),
encodes specified text fields with a sentence transformer, and saves
per-sample .npz files with the embeddings.

Usage:
    py scripts/encode_bridge_texts.py \
        --manifest data/interim/droid_process_chain/bridge_texts/train.jsonl \
        --out_dir data/interim/droid_process_chain/text_embeddings \
        --fields template_text enriched_text caption
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--fields", nargs="+", default=["template_text", "enriched_text"])
    parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model_name, device=args.device)
    print(f"Loaded {args.model_name} (dim={model.get_sentence_embedding_dimension()})")

    rows = []
    with args.manifest.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    print(f"Loaded {len(rows)} samples, encoding fields: {args.fields}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for field in args.fields:
        texts = [row.get(field, "") or "" for row in rows]
        embeddings = model.encode(texts, batch_size=args.batch_size, show_progress_bar=True)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        print(f"  [{field}] shape={embeddings.shape}")

        for i, row in enumerate(rows):
            out_path = args.out_dir / f'{row["sample_id"]}.npz'
            existing = dict(np.load(out_path)) if out_path.exists() else {}
            existing[field] = embeddings[i]
            np.savez_compressed(out_path, **existing)

    print(f"Saved text embeddings to {args.out_dir}")


if __name__ == "__main__":
    main()
