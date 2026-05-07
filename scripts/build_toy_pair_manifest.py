from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
STAMO_TOY_ROOT = ROOT / "StaMo" / "toy_data"


def make_labels(start_idx: int, end_idx: int) -> dict[str, str]:
    gap = end_idx - start_idx
    motion_direction = (start_idx + end_idx) % 3

    if gap == 0:
        gripper_state = "open"
        object_motion = "still"
        stage_label = "approach"
        target_relation = "farther"
        contact_state = "no_contact"
    elif gap == 1:
        gripper_state = "closing"
        object_motion = "still"
        stage_label = "contact"
        target_relation = "closer"
        contact_state = "contact"
    elif gap == 2:
        gripper_state = "closed"
        object_motion = "moved"
        stage_label = "grasp"
        target_relation = "closer"
        contact_state = "contact"
    elif gap == 3:
        gripper_state = "closed"
        object_motion = "lifted"
        stage_label = "lift"
        target_relation = "closer"
        contact_state = "contact"
    else:
        gripper_state = "closed"
        object_motion = "placed" if motion_direction == 2 else "moved"
        stage_label = "place" if motion_direction == 2 else "move"
        target_relation = "reached" if motion_direction == 2 else "closer"
        contact_state = "contact"

    return {
        "stage_label": stage_label,
        "contact_state": contact_state,
        "gripper_state": gripper_state,
        "object_motion": object_motion,
        "target_relation": target_relation,
    }


def make_process_text(labels: dict[str, str]) -> str:
    return (
        f"contact_state={labels['contact_state']} -> "
        f"gripper_state={labels['gripper_state']} -> "
        f"object_motion={labels['object_motion']} -> "
        f"target_relation={labels['target_relation']}"
    )


def make_caption(labels: dict[str, str]) -> str:
    stage = labels["stage_label"]
    if stage in {"approach", "contact"}:
        return "The robot is getting ready to interact with the object."
    if stage in {"grasp", "lift"}:
        return "The robot is handling the object during a short manipulation step."
    return "The robot is adjusting the object's position relative to the workspace target."


def build_rows(image_dir: Path, split: str, max_gap: int) -> list[dict]:
    image_paths = sorted(image_dir.glob("*.png"))
    rows: list[dict] = []

    for start_idx, start_path in enumerate(image_paths):
        for gap in range(0, max_gap + 1):
            end_idx = start_idx + gap
            if end_idx >= len(image_paths):
                break

            labels = make_labels(start_idx, end_idx)
            rows.append(
                {
                    "sample_id": f"{split}_{start_idx:03d}_{end_idx:03d}",
                    "split": split,
                    "task_name": "toy_process_chain",
                    "image_t": str(start_path.resolve()),
                    "image_tp": str(image_paths[end_idx].resolve()),
                    "trajectory_id": f"{split}_toy_sequence",
                    "start_step": start_idx,
                    "end_step": end_idx,
                    "caption": make_caption(labels),
                    "process_text": make_process_text(labels),
                    "action": [float(gap), float(end_idx - start_idx), 0.0, 0.0],
                    "ee_delta": [float(gap), float(end_idx - start_idx), 0.0],
                    "labels": labels,
                }
            )
    return rows


def stratified_split(rows: list[dict], val_ratio: float) -> tuple[list[dict], list[dict]]:
    labels = [row["labels"]["stage_label"] for row in rows]
    train_rows, val_rows = train_test_split(
        rows,
        test_size=val_ratio,
        random_state=42,
        stratify=labels,
    )
    return train_rows, val_rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=ROOT / "data" / "processed" / "toy_process_chain")
    parser.add_argument("--max_gap", type=int, default=4)
    args = parser.parse_args()

    train_rows = build_rows(STAMO_TOY_ROOT / "train", "train", args.max_gap)
    eval_rows = build_rows(STAMO_TOY_ROOT / "eval", "test", args.max_gap)
    train_rows, val_rows = stratified_split(train_rows, val_ratio=0.2)

    write_jsonl(args.out_dir / "train.jsonl", train_rows)
    write_jsonl(args.out_dir / "val.jsonl", val_rows)
    write_jsonl(args.out_dir / "test.jsonl", eval_rows)

    print(f"Train rows: {len(train_rows)}")
    print(f"Val rows: {len(val_rows)}")
    print(f"Test rows: {len(eval_rows)}")
    print(f"Saved manifests to {args.out_dir}")


if __name__ == "__main__":
    main()
