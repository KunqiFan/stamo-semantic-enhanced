from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_step(command: list[str], cwd: Path) -> None:
    print(f"\n[RUN] {' '.join(command)}")
    subprocess.run(command, cwd=str(cwd), check=True)


def main() -> None:
    python = sys.executable

    run_step([python, "scripts/build_toy_pair_manifest.py"], ROOT)

    latent_dir = ROOT / "data" / "interim" / "toy_process_chain" / "latents"
    for split in ["train", "val", "test"]:
        run_step(
            [
                python,
                "scripts/extract_delta_z.py",
                "--manifest",
                f"data/processed/toy_process_chain/{split}.jsonl",
                "--out_dir",
                str(latent_dir),
            ],
            ROOT,
        )

    semantics_dir = ROOT / "data" / "interim" / "toy_process_chain" / "semantics"

    # ---------------------------------------------------------------
    # Stage A: train compact semantics with multiple classifiers
    # ---------------------------------------------------------------
    classifiers = ["logistic", "mlp", "pca_mlp", "rf", "hgb"]
    feature_keys = ["delta_z", "delta_pooled", "both"]

    for clf_name in classifiers:
        for fk in feature_keys:
            tag = f"{clf_name}_{fk}"
            print(f"\n{'='*60}")
            print(f"  Compact semantics: classifier={clf_name}  feature_key={fk}")
            print(f"{'='*60}")

            out_path = semantics_dir / f"test_predicted_semantics_{tag}.jsonl"
            run_step(
                [
                    python,
                    "scripts/train_compact_semantics.py",
                    "--train_manifest",
                    "data/processed/toy_process_chain/train.jsonl",
                    "--eval_manifest",
                    "data/processed/toy_process_chain/test.jsonl",
                    "--latent_dir",
                    str(latent_dir),
                    "--feature_key",
                    fk,
                    "--classifier",
                    clf_name,
                    "--out_path",
                    str(out_path),
                ],
                ROOT,
            )

    # ---------------------------------------------------------------
    # Stage B: process chain comparisons
    # ---------------------------------------------------------------

    # B1: text_only baseline
    print(f"\n{'='*60}")
    print("  Process chain: text_only")
    print(f"{'='*60}")
    run_step(
        [
            python,
            "scripts/train_process_chain.py",
            "--train_manifest",
            "data/processed/toy_process_chain/train.jsonl",
            "--eval_manifest",
            "data/processed/toy_process_chain/test.jsonl",
            "--setting",
            "text_only",
            "--target",
            "stage_label",
            "--text_key",
            "caption",
        ],
        ROOT,
    )

    # B2: semantics_only with best predicted semantics
    best_sem = semantics_dir / "test_predicted_semantics_hgb_both.jsonl"
    if not best_sem.exists():
        best_sem = semantics_dir / "test_predicted_semantics_logistic_delta_z.jsonl"

    print(f"\n{'='*60}")
    print(f"  Process chain: semantics_only (using {best_sem.name})")
    print(f"{'='*60}")
    run_step(
        [
            python,
            "scripts/train_process_chain.py",
            "--train_manifest",
            "data/processed/toy_process_chain/train.jsonl",
            "--eval_manifest",
            "data/processed/toy_process_chain/test.jsonl",
            "--setting",
            "semantics_only",
            "--target",
            "stage_label",
            "--predicted_semantics",
            str(best_sem),
        ],
        ROOT,
    )

    # B3: text_plus_semantics with STACKING fusion (the fix)
    print(f"\n{'='*60}")
    print(f"  Process chain: text_plus_semantics (stacking, using {best_sem.name})")
    print(f"{'='*60}")
    run_step(
        [
            python,
            "scripts/train_process_chain.py",
            "--train_manifest",
            "data/processed/toy_process_chain/train.jsonl",
            "--eval_manifest",
            "data/processed/toy_process_chain/test.jsonl",
            "--setting",
            "text_plus_semantics",
            "--target",
            "stage_label",
            "--text_key",
            "caption",
            "--predicted_semantics",
            str(best_sem),
            "--fusion",
            "stacking",
        ],
        ROOT,
    )

    # B4: text_plus_semantics with CONCAT fusion (comparison)
    print(f"\n{'='*60}")
    print(f"  Process chain: text_plus_semantics (concat, using {best_sem.name})")
    print(f"{'='*60}")
    run_step(
        [
            python,
            "scripts/train_process_chain.py",
            "--train_manifest",
            "data/processed/toy_process_chain/train.jsonl",
            "--eval_manifest",
            "data/processed/toy_process_chain/test.jsonl",
            "--setting",
            "text_plus_semantics",
            "--target",
            "stage_label",
            "--text_key",
            "caption",
            "--predicted_semantics",
            str(best_sem),
            "--fusion",
            "concat",
        ],
        ROOT,
    )

    # B5: text_plus_gold_semantics (upper bound)
    print(f"\n{'='*60}")
    print("  Process chain: text_plus_gold_semantics (upper bound)")
    print(f"{'='*60}")
    run_step(
        [
            python,
            "scripts/train_process_chain.py",
            "--train_manifest",
            "data/processed/toy_process_chain/train.jsonl",
            "--eval_manifest",
            "data/processed/toy_process_chain/test.jsonl",
            "--setting",
            "text_plus_semantics",
            "--target",
            "stage_label",
            "--text_key",
            "caption",
            "--fusion",
            "stacking",
        ],
        ROOT,
    )

    print("\nToy process-chain pipeline completed.")


if __name__ == "__main__":
    main()
