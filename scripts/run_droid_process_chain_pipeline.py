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

    # 1. Build DROID pair manifest
    print("\n" + "="*60)
    print("  Stage 1: Build DROID Pair Manifest")
    print("="*60)
    # Using 500 episodes as max to keep it reasonable for local testing, 
    # adjust as needed.
    run_step([
        python, "scripts/build_droid_pair_manifest.py",
        "--droid_dir", "data/raw/droid_100/1.0.0",
        "--out_dir", "data/processed/droid_process_chain",
        "--max_episodes", "100",  # We can increase this later
        "--require_language"
    ], ROOT)

    # 2. Extract Latents
    print("\n" + "="*60)
    print("  Stage 2: Extract Latents")
    print("="*60)
    latent_dir = ROOT / "data" / "interim" / "droid_process_chain" / "latents"
    for split in ["train", "val", "test"]:
        run_step([
            python, "scripts/extract_delta_z.py",
            "--manifest", f"data/processed/droid_process_chain/{split}.jsonl",
            "--out_dir", str(latent_dir)
        ], ROOT)

    # 3. Train Compact Semantics
    print("\n" + "="*60)
    print("  Stage 3: Train Compact Semantics")
    print("="*60)
    semantics_dir = ROOT / "data" / "interim" / "droid_process_chain" / "semantics"
    
    classifiers = ["logistic", "mlp", "pca_mlp", "rf", "hgb"]
    feature_keys = ["delta_z", "delta_pooled", "both"]

    for clf_name in classifiers:
        for fk in feature_keys:
            tag = f"{clf_name}_{fk}"
            out_path = semantics_dir / f"test_predicted_semantics_{tag}.jsonl"
            run_step([
                python, "scripts/train_compact_semantics.py",
                "--train_manifest", "data/processed/droid_process_chain/train.jsonl",
                "--eval_manifest", "data/processed/droid_process_chain/test.jsonl",
                "--latent_dir", str(latent_dir),
                "--feature_key", fk,
                "--classifier", clf_name,
                "--out_path", str(out_path)
            ], ROOT)

    # 4. Evaluate Process Chain
    print("\n" + "="*60)
    print("  Stage 4: Process Chain Evaluation")
    print("="*60)
    
    # B1: text_only baseline
    run_step([
        python, "scripts/train_process_chain.py",
        "--train_manifest", "data/processed/droid_process_chain/train.jsonl",
        "--eval_manifest", "data/processed/droid_process_chain/test.jsonl",
        "--setting", "text_only",
        "--target", "stage_label",
        "--text_key", "caption"
    ], ROOT)

    # B2: semantics_only with best predicted semantics
    best_sem = semantics_dir / "test_predicted_semantics_hgb_both.jsonl"
    if not best_sem.exists():
        best_sem = semantics_dir / "test_predicted_semantics_logistic_delta_z.jsonl"

    run_step([
        python, "scripts/train_process_chain.py",
        "--train_manifest", "data/processed/droid_process_chain/train.jsonl",
        "--eval_manifest", "data/processed/droid_process_chain/test.jsonl",
        "--setting", "semantics_only",
        "--target", "stage_label",
        "--predicted_semantics", str(best_sem)
    ], ROOT)

    # B3: text_plus_semantics with STACKING fusion
    run_step([
        python, "scripts/train_process_chain.py",
        "--train_manifest", "data/processed/droid_process_chain/train.jsonl",
        "--eval_manifest", "data/processed/droid_process_chain/test.jsonl",
        "--setting", "text_plus_semantics",
        "--target", "stage_label",
        "--text_key", "caption",
        "--predicted_semantics", str(best_sem),
        "--fusion", "stacking"
    ], ROOT)
    
    print("\nDROID process-chain pipeline completed.")


if __name__ == "__main__":
    main()
