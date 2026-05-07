# Semantic-Enhanced Visual Representation Learning for Robotic Manipulation

This project investigates whether injecting lightweight semantic supervision into diffusion-based visual compression pipelines can preserve task-relevant semantic information without sacrificing generation quality.

## Key Findings

1. **Diffusion-only compression discards semantics**: StaMo's Projector (96:1 compression) trained with only diffusion loss produces representations with ~54% semantic classification accuracy (near random).

2. **Semantic injection recovers semantics at zero inference cost**: Adding a lightweight SemanticHead (~0.6M params, 0.6% of trainable params) during training raises semantic accuracy to 95.9% (contact_state), while diffusion quality remains unchanged (PSNR 9.10 → 8.96).

3. **U-shaped λ sensitivity**: Ablation over λ ∈ {0.1, 0.5, 1.0, 2.0, 5.0} reveals gradient competition between diffusion and semantic objectives. λ=0.1 (semantic as gentle auxiliary) is optimal.

4. **Attachment position matters**: SemanticHead at Projector output (Position B, 68.8%) significantly outperforms DiTConditionHead output (Position A, 60.9%) and DINOv2 features (Position C, 53.7%).

5. **Compact semantics as state interface**: 12D semantic vectors outperform 512D delta_z in stage classification (78% vs 45%), few-shot learning (91.7% with 2% data), and action prediction (z-axis R²=0.76).

## Architecture

```
Image pair (t, t+Δ)
    ↓
DINOv2 ViT-B/14 (frozen, 86.4M params)
    ↓
Projector (cross-attention ×4, trainable)
    ↓
delta_z [2×1024]  ──→  DiT → Diffusion Loss
    └──→  SemanticHead (4× MLP) → Semantic Loss (×0.1)
```

Semantic attributes predicted:
- `contact_state`: no_contact / contact
- `gripper_state`: open / closing / closed
- `object_motion`: still / lifted / moved / placed
- `target_relation`: farther / closer / reached

## Project Structure

```
├── stamo/                  # StaMo model source (backbone, projector, renderer)
├── src/stamo_bridge/       # Semantic bridge library (schemas, interface, baselines)
├── scripts/                # Experiment scripts (training, evaluation, ablation)
│   └── rl_validation/      # RL validation scripts (PPO + StaMo encoder)
├── wrappers/               # Gym observation wrapper for StaMo encoder
├── configs/                # YAML configurations
├── results/                # Experiment results (JSON + ablation logs)
│   └── rl_validation/      # RL experiment progress reports
└── docs/                   # Research reports and documentation
```

## Results Summary

### Semantic Classification (Eval Set, 638 samples)

| Method | contact | gripper | motion | relation | Mean |
|--------|---------|---------|--------|----------|------|
| Diffusion-only delta_z | ~54% | ~50% | ~85% | ~45% | ~58.5% |
| DINOv2 raw features | 96.9% | 91.7% | 99.4% | 66.8% | 88.7% |
| **Ours (semantic-enhanced)** | **95.9%** | **91.5%** | **99.4%** | **64.6%** | **87.9%** |

### Generation Quality (unchanged)

| Metric | Diffusion-only | Semantic-enhanced | Δ |
|--------|---------------|-------------------|---|
| PSNR ↑ | 9.10 ± 1.12 | 8.96 ± 0.88 | -1.5% |
| LPIPS ↓ | 1.256 ± 0.028 | 1.230 ± 0.031 | -2.1% ✓ |

## Ongoing Work: RL Validation

We are running PPO policy learning experiments on ManiSkill3 (PushCube-v1) to verify that semantic-enhanced representations lead to faster convergence or higher success rates in downstream control tasks. See `results/rl_validation/progress_report.md` for details.

## Setup

```bash
pip install -r requirements.txt
```

**Data**: Not included due to size. See `docs/experiment_plan.md` for data preparation instructions.

**Checkpoints**: Train from scratch using scripts in `scripts/` or contact the author.

## Usage

### Train StaMo with semantic enhancement

```bash
python scripts/train_stamo_droid.py --config configs/droid.yaml
```

### Run ablation experiments

```bash
python scripts/run_lambda_ablation.py
python scripts/ablation_head_position.py
```

### Evaluate semantic quality

```bash
python scripts/evaluate.py --config configs/experiment.yaml
```

## Citation

If you find this work useful, please cite:

```
@misc{fan2026semantic,
  title={Semantic-Enhanced Visual Representation Learning for Robotic Manipulation},
  author={Kunqi Fan},
  year={2026},
  institution={Zhejiang University, Turing Class}
}
```

## License

This project is for academic research purposes.
