from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def normalize_row(row: dict, split: str, defaults: dict) -> dict:
    def field(name: str, fallback=None):
        key = defaults.get(f"{name}_key", name)
        return row.get(key, fallback)

    sample_id = row.get("sample_id") or f"{field('trajectory')}_{field('start')}_{field('end')}"
    labels = field("labels", {}) or {}

    normalized = {
        "sample_id": sample_id,
        "split": split,
        "task_name": defaults.get("task_name", "short_horizon_manipulation"),
        "image_t": field("image_t"),
        "image_tp": field("image_tp"),
        "trajectory_id": field("trajectory"),
        "start_step": field("start"),
        "end_step": field("end"),
        "caption": field("caption", ""),
        "process_text": field("process_text", ""),
        "action": field("action"),
        "ee_delta": field("ee_delta"),
        "labels": labels,
    }
    return normalized


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=ROOT / "data" / "processed" / "real_process_chain")
    args = parser.parse_args()

    config = load_json(args.config)
    defaults = config["defaults"]
    splits = config["splits"]

    for split_name, split_info in splits.items():
        pair_metadata = Path(split_info["pair_metadata"])
        if not pair_metadata.is_absolute():
            pair_metadata = ROOT / pair_metadata

        rows = [normalize_row(row, split_name, defaults) for row in iter_jsonl(pair_metadata)]
        write_jsonl(args.out_dir / f"{split_name}.jsonl", rows)
        print(f"{split_name}: wrote {len(rows)} rows to {args.out_dir / f'{split_name}.jsonl'}")


if __name__ == "__main__":
    main()
