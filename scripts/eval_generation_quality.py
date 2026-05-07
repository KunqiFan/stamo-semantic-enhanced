"""Generation quality evaluation: pure-diffusion vs semantic-parallel checkpoints.

Compares PSNR / SSIM / LPIPS on the test set.
  image_t (condition) → eval_step → generated image_tp  vs  ground truth image_tp

Usage:
    cd StaMo && py ../scripts/eval_generation_quality.py
    cd StaMo && py ../scripts/eval_generation_quality.py --save_samples 8
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STAMO_ROOT = ROOT / "StaMo"
if str(STAMO_ROOT) not in sys.path:
    sys.path.insert(0, str(STAMO_ROOT))

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from stamo.renderer.model.renderer import RenderNet
from stamo.renderer.utils.data import PairImageData, collate_fn_pair
from stamo.renderer.utils.metrics import calculate_psnr, calculate_ssim

CHECKPOINTS = {
    "pure_diffusion": {
        "ckpt": str(STAMO_ROOT / "ckpts" / "droid" / "5000"),
        "semantic_enabled": False,
    },
    "semantic_parallel": {
        "ckpt": str(STAMO_ROOT / "ckpts" / "droid_semantic" / "5000"),
        "semantic_enabled": True,
    },
}


def build_model(config_path: Path, semantic_enabled: bool, ckpt_path: str):
    args = OmegaConf.load(str(config_path))
    args.deepspeed = False
    args.semantic_head.enabled = semantic_enabled
    model = RenderNet(args)
    model = model.to("cuda")
    model.load_checkpoint(ckpt_path)
    model.eval()
    return model, args


@torch.no_grad()
def evaluate_checkpoint(model, dataloader, generator, lpips_fn=None, save_samples=0, save_dir=None):
    psnr_list, ssim_list, lpips_list = [], [], []
    saved = 0

    for batch in dataloader:
        images_t = batch["images_t"].to("cuda")
        images_tp_gt = batch["images_tp"].to("cuda")

        outputs = model.eval_step({"images": images_t, "generator": generator}, {})
        gen_img = model.inv_vae_transform(outputs["images"])
        gen_img = torch.clamp(gen_img, 0, 1)

        psnr_list.append(calculate_psnr(gen_img, images_tp_gt).item())
        ssim_list.append(calculate_ssim(gen_img, images_tp_gt).item())

        if lpips_fn is not None:
            gen_lpips = gen_img * 2 - 1
            gt_lpips = images_tp_gt * 2 - 1
            lp = lpips_fn(gen_lpips, gt_lpips).mean().item()
            lpips_list.append(lp)

        if save_samples > 0 and saved < save_samples and save_dir:
            from torchvision.utils import save_image
            for i in range(min(images_t.size(0), save_samples - saved)):
                save_image(images_t[i].cpu(), save_dir / f"{saved}_cond.png")
                save_image(gen_img[i].cpu(), save_dir / f"{saved}_pred.png")
                save_image(images_tp_gt[i].cpu(), save_dir / f"{saved}_gt.png")
                saved += 1

    results = {
        "PSNR": (np.mean(psnr_list), np.std(psnr_list)),
        "SSIM": (np.mean(ssim_list), np.std(ssim_list)),
    }
    if lpips_list:
        results["LPIPS"] = (np.mean(lpips_list), np.std(lpips_list))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=STAMO_ROOT / "configs" / "droid.yaml")
    parser.add_argument("--save_samples", type=int, default=0, help="Number of sample images to save")
    parser.add_argument("--batch_size", type=int, default=1)
    cli = parser.parse_args()

    try:
        import lpips as _lpips
        lpips_fn = _lpips.LPIPS(net="alex").to("cuda")
        print("LPIPS (AlexNet) loaded")
    except ImportError:
        lpips_fn = None
        print("lpips not installed, skipping LPIPS metric")

    args_tmp = OmegaConf.load(str(cli.config))
    dataset = PairImageData(
        args_tmp.data.eval_manifest, flip_p=0.0, img_size=args_tmp.data.img_size,
    )
    dataloader = DataLoader(
        dataset, batch_size=cli.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn_pair, drop_last=False,
    )
    print(f"Test set: {len(dataset)} samples, {len(dataloader)} batches")

    generator = torch.Generator(device="cuda").manual_seed(args_tmp.seed)

    all_results = {}
    for name, cfg in CHECKPOINTS.items():
        print(f"\n{'='*50}")
        print(f"Evaluating: {name}")
        print(f"  Checkpoint: {cfg['ckpt']}")
        print(f"  Semantic head: {cfg['semantic_enabled']}")

        model, _ = build_model(cli.config, cfg["semantic_enabled"], cfg["ckpt"])

        save_dir = None
        if cli.save_samples > 0:
            save_dir = Path(ROOT / "eval_samples" / name)
            save_dir.mkdir(parents=True, exist_ok=True)

        generator.manual_seed(args_tmp.seed)
        results = evaluate_checkpoint(
            model, dataloader, generator,
            lpips_fn=lpips_fn, save_samples=cli.save_samples, save_dir=save_dir,
        )
        all_results[name] = results

        for metric, (mean, std) in results.items():
            print(f"  {metric}: {mean:.4f} +/- {std:.4f}")

        del model
        torch.cuda.empty_cache()

    print(f"\n{'='*50}")
    print("Comparison Table:")
    print(f"{'Metric':<10}", end="")
    for name in all_results:
        print(f"  {name:<25}", end="")
    print()
    metrics = list(next(iter(all_results.values())).keys())
    for metric in metrics:
        print(f"{metric:<10}", end="")
        for name in all_results:
            mean, std = all_results[name][metric]
            print(f"  {mean:.4f} +/- {std:.4f}     ", end="")
        print()


if __name__ == "__main__":
    main()
