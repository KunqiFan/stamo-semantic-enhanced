from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def heuristic_semantics(delta_z: np.ndarray) -> dict[str, str]:
    """Very small bootstrap heuristic for early pipeline testing."""
    mag = float(np.linalg.norm(delta_z))
    z_mean = float(delta_z.mean())
    z_max = float(delta_z.max())

    if mag < 4.0:
        object_motion = "still"
    elif z_max > 1.5:
        object_motion = "lifted"
    else:
        object_motion = "moved"

    return {
        "contact_state": "contact" if z_mean > 0.0 else "no_contact",
        "gripper_state": "closed" if z_max > 1.0 else "open",
        "object_motion": object_motion,
        "target_relation": "closer" if z_mean > -0.1 else "farther",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dir", type=Path, required=True)
    parser.add_argument("--out_path", type=Path, required=True)
    args = parser.parse_args()

    args.out_path.parent.mkdir(parents=True, exist_ok=True)

    with args.out_path.open("w", encoding="utf-8") as f:
        for npz_path in sorted(args.latent_dir.glob("*.npz")):
            data = np.load(npz_path)
            semantics = heuristic_semantics(data["delta_z"])
            row = {"sample_id": npz_path.stem, "predicted_semantics": semantics}
            f.write(json.dumps(row) + "\n")

    print(f"Saved semantics to {args.out_path}")


if __name__ == "__main__":
    main()
