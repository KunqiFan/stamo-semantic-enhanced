import copy
import inspect
import os
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torchvision.transforms as T
from diffusers.models import AutoencoderKL
from diffusers.schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    FlowMatchEulerDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils.torch_utils import randn_tensor
from tqdm.auto import tqdm

from stamo.renderer.model.backbone import DiTConditionHead, SD3TransformerBackbone, SemanticHead, VisionBackbone
from stamo.renderer.model.projector import Projector
from stamo.renderer.utils.overwatch import initialize_overwatch


overwatch = initialize_overwatch(__name__)


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class RenderNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        toy_mode = getattr(args.render_net, "toy_mode", False)

        self.vision_backbone = VisionBackbone(
            img_size=args.data.img_size,
            model_name=args.vision_backbone.model_name,
            pretrained=args.vision_backbone.pretrained,
            local_ckpt=args.vision_backbone.local_ckpt,
        )

        if toy_mode:
            toy_cfg = args.render_net.toy
            self.DiT = SD3TransformerBackbone(
                sample_size=toy_cfg.sample_size,
                patch_size=toy_cfg.patch_size,
                in_channels=toy_cfg.in_channels,
                num_layers=toy_cfg.num_layers,
                attention_head_dim=toy_cfg.attention_head_dim,
                num_attention_heads=toy_cfg.num_attention_heads,
                joint_attention_dim=args.projector.output_align_dim,
                caption_projection_dim=toy_cfg.caption_projection_dim,
                pooled_projection_dim=toy_cfg.pooled_projection_dim,
                out_channels=toy_cfg.in_channels,
                pos_embed_max_size=toy_cfg.pos_embed_max_size,
            )

            self.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=toy_cfg.num_train_timesteps)
            self.vae = AutoencoderKL(
                in_channels=3,
                out_channels=3,
                down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
                up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
                block_out_channels=(32, 64),
                layers_per_block=1,
                latent_channels=toy_cfg.in_channels,
                norm_num_groups=8,
                sample_size=args.data.img_size,
            )
        else:
            self.DiT = SD3TransformerBackbone.from_pretrained(args.render_net.sd3.local_ckpt, subfolder="transformer")
            self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                args.render_net.sd3.local_ckpt, subfolder="scheduler"
            )
            self.vae = AutoencoderKL.from_pretrained(args.render_net.sd3.local_ckpt, subfolder="vae")

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.scheduler_copy = copy.deepcopy(self.scheduler)

        self.projector = Projector(args, self.vision_backbone.patches, self.vision_backbone.channels)

        pooled_dim = getattr(getattr(args.render_net, "toy", {}), "pooled_projection_dim", 2048) if toy_mode else 2048
        self.dit_condition_head = DiTConditionHead(
            input_dim=args.projector.output_align_dim,
            pooled_dim=pooled_dim,
        )

        self.token_dropout = args.render_net.token_dropout
        self.num_token = args.projector.num_token

        semantic_cfg = getattr(args, "semantic_head", None)
        self.use_semantic_head = semantic_cfg is not None and getattr(semantic_cfg, "enabled", False)
        if self.use_semantic_head:
            self.semantic_position = getattr(semantic_cfg, "position", "pooled")
            if self.semantic_position == "pooled":
                sem_input_dim = pooled_dim  # 512 (toy) or 2048
            elif self.semantic_position == "proj":
                sem_input_dim = args.projector.output_align_dim  # 1024
            elif self.semantic_position == "dino":
                sem_input_dim = self.vision_backbone.channels  # 768
            else:
                raise ValueError(f"Unknown semantic_head.position: {self.semantic_position}")
            self.semantic_head = SemanticHead(
                input_dim=sem_input_dim,
                hidden_dim=getattr(semantic_cfg, "hidden_dim", 256),
            )
            self.semantic_lambda = getattr(semantic_cfg, "lambda_weight", 1.0)
        else:
            self.semantic_lambda = 0.0

        if toy_mode:
            tr_noise_scheduler = DDPMScheduler(num_train_timesteps=toy_cfg.num_train_timesteps)
            if args.render_net.eval_scheduler == "ddpm":
                noise_scheduler = DDPMScheduler(num_train_timesteps=toy_cfg.num_train_timesteps)
            elif args.render_net.eval_scheduler == "ddim":
                noise_scheduler = DDIMScheduler(num_train_timesteps=toy_cfg.num_train_timesteps)
                tr_noise_scheduler = DDIMScheduler(num_train_timesteps=toy_cfg.num_train_timesteps)
            else:
                noise_scheduler = PNDMScheduler(num_train_timesteps=toy_cfg.num_train_timesteps)
        else:
            tr_noise_scheduler = DDPMScheduler.from_pretrained(args.render_net.sd3.local_ckpt, subfolder="scheduler")
            if args.render_net.eval_scheduler == "ddpm":
                noise_scheduler = DDPMScheduler.from_pretrained(args.render_net.sd3.local_ckpt, subfolder="scheduler")
            elif args.render_net.eval_scheduler == "ddim":
                noise_scheduler = DDIMScheduler.from_pretrained(args.render_net.sd3.local_ckpt, subfolder="scheduler")
                tr_noise_scheduler = DDIMScheduler.from_pretrained(args.render_net.sd3.local_ckpt, subfolder="scheduler")
            else:
                noise_scheduler = PNDMScheduler.from_pretrained(args.render_net.sd3.local_ckpt, subfolder="scheduler")
        self.tr_noise_scheduler = tr_noise_scheduler
        self.val_noise_scheduler = noise_scheduler

        self.height, self.width = args.data.img_size, args.data.img_size

        self.seed = args.seed
        self.guidance_scale = args.render_net.guidance_scale
        self.num_inference_steps = args.render_net.num_inference_steps
        self.weighting_scheme = getattr(args.render_net, "weighting_scheme", "logit_normal")
        self.logit_mean = getattr(args.render_net, "logit_mean", 0.0)
        self.logit_std = getattr(args.render_net, "logit_std", 1.0)
        self.mode_scale = getattr(args.render_net, "mode_scale", 1.29)

        self.projector_feature_extractor = self.vision_backbone.transforms
        self.dit_feature_extractor = T.Normalize(mean=[0.5], std=[0.5])

        self.inv_vae_transform = T.Compose([T.Lambda(lambda img: img * 0.5 + 0.5)])

    def to(self, *args, **kwargs):
        model_converted = super().to(*args, **kwargs)
        self.device = next(self.parameters()).device
        self.dtype = next(self.DiT.parameters()).dtype
        return model_converted

    def set_trainable_params(self):
        self.DiT.enable_gradient_checkpointing()
        self.DiT.train()
        self.DiT.requires_grad_(True)

        self.projector.train()
        self.projector.requires_grad_(True)

        self.dit_condition_head.train()
        self.dit_condition_head.requires_grad_(True)

        self.vae.eval()
        self.vae.requires_grad_(False)

        self.vision_backbone.eval()
        self.vision_backbone.requires_grad_(False)

        if self.use_semantic_head:
            self.semantic_head.train()
            self.semantic_head.requires_grad_(True)

    def save_checkpoint(self, save_path: str, global_step: int) -> None:
        exclude_prefixes = ["vae", "projector", "vision_backbone"]
        save_dict = {"model": {}, "global_step": global_step}
        model_state_dict = self.state_dict()
        for k, v in model_state_dict.items():
            if not any(k.startswith(prefix) for prefix in exclude_prefixes):
                save_dict["model"][k] = v
        torch.save(save_dict, os.path.join(save_path, "RenderNet.pth"))
        torch.save(self.projector.state_dict(), os.path.join(save_path, "Projector.pth"))
        if self.use_semantic_head:
            torch.save(self.semantic_head.state_dict(), os.path.join(save_path, "SemanticHead.pth"))

    def load_checkpoint(self, load_path: str) -> int:
        assert os.path.exists(os.path.join(load_path, "Projector.pth")), f"Projector.pth not found in {load_path}"
        assert os.path.exists(os.path.join(load_path, "RenderNet.pth")), f"RenderNet.pth not found in {load_path}"
        overwatch.warning(f"loading checkpoints from {load_path}")

        def _log_missing_unexpected(title, missing_keys, unexpected_keys):
            def extract_top_level(keys):
                return sorted({k.split(".")[0] for k in keys})

            top_missing = extract_top_level(missing_keys)
            top_unexpected = extract_top_level(unexpected_keys)

            overwatch.warning(f"{title} - Missing top-level keys: {top_missing}")
            overwatch.warning(f"{title} - Unexpected top-level keys: {top_unexpected}")

        rendernet_ckpt = torch.load(os.path.join(load_path, "RenderNet.pth"), map_location="cpu")
        missing, unexpected = self.load_state_dict(rendernet_ckpt["model"], strict=False)
        _log_missing_unexpected("RenderNet", missing, unexpected)

        projector_ckpt = torch.load(os.path.join(load_path, "Projector.pth"), map_location="cpu")
        missing, unexpected = self.projector.load_state_dict(projector_ckpt, strict=False)
        _log_missing_unexpected("Projector", missing, unexpected)

        semantic_path = os.path.join(load_path, "SemanticHead.pth")
        if self.use_semantic_head and os.path.exists(semantic_path):
            semantic_ckpt = torch.load(semantic_path, map_location="cpu")
            missing, unexpected = self.semantic_head.load_state_dict(semantic_ckpt, strict=False)
            _log_missing_unexpected("SemanticHead", missing, unexpected)

        return rendernet_ckpt["global_step"]

    def encode(self, images: torch.Tensor, do_classifier_free_guidance: bool = False, return_dino: bool = False):
        """Encode image using VisionBackbone and project to transformer input space"""
        assert isinstance(images, torch.Tensor), f"Image must be a torch.Tensor, but: {images.__class__}"

        dtype = next(self.projector.parameters()).dtype

        images = images.to(device=self.device, dtype=dtype)

        # Get compressed image features
        dino_embeds = self.vision_backbone(images)
        image_embeds = self.projector(dino_embeds)

        pooled_embeds = self.dit_condition_head(image_embeds)

        if self.token_dropout:
            dropout_range = torch.randint(1, self.num_token + 1, ())
            image_embeds = image_embeds[:, :dropout_range]

        if do_classifier_free_guidance:
            negative_prompt_embeds = torch.zeros_like(image_embeds)
            negative_pooled_embeds = torch.zeros_like(pooled_embeds)
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeds = torch.cat([negative_prompt_embeds, image_embeds])
            pooled_embeds = torch.cat([negative_pooled_embeds, pooled_embeds])

        image_embeds = image_embeds.to(dtype=self.dtype)
        pooled_embeds = pooled_embeds.to(dtype=self.dtype)

        if return_dino:
            return image_embeds, pooled_embeds, dino_embeds
        return image_embeds, pooled_embeds

    def compute_time_ids(self, original_size, crops_coords_top_left, do_classifier_free_guidance=False):
        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        target_size = torch.tensor((self.width, self.height), device=self.device)
        add_time_ids = torch.concat([original_size, crops_coords_top_left, target_size], dim=-1)
        add_time_ids = add_time_ids.to(self.device, self.dtype).unsqueeze(0)

        if do_classifier_free_guidance:
            add_time_ids = torch.concat([add_time_ids, add_time_ids], dim=0)

        return add_time_ids

    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        sigmas = self.scheduler_copy.sigmas.to(device=self.device, dtype=dtype)
        schedule_timesteps = self.scheduler_copy.timesteps.to(self.device)
        timesteps = timesteps.to(self.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        return latents

    def vae_encode(self, images):
        latents = self.vae.encode(images).latent_dist.sample()
        shift_factor = getattr(self.vae.config, "shift_factor", 0.0) or 0.0
        scaling_factor = getattr(self.vae.config, "scaling_factor", 1.0) or 1.0
        latents = (latents - shift_factor) * scaling_factor
        latents = latents.to(dtype=self.dtype)
        return latents

    def vae_decode(self, latents):
        shift_factor = getattr(self.vae.config, "shift_factor", 0.0) or 0.0
        scaling_factor = getattr(self.vae.config, "scaling_factor", 1.0) or 1.0
        latents = latents / scaling_factor + shift_factor
        latents = latents.to(dtype=self.dtype)
        images = self.vae.decode(latents, return_dict=False)[0]
        return images

    def train(self, *args):
        super().train(*args)
        self.set_trainable_params()

    def train_step(self, inputs: Dict[str, Any], outputs: Dict[str, Any], criterion: nn.Module) -> Dict[str, Any]:
        # Pair mode: condition on image_t, reconstruct image_tp
        if "images_t" in inputs:
            images_t = inputs["images_t"]
            images_tp = inputs["images_tp"]
            bsz = images_t.shape[0]

            projector_images_t = self.projector_feature_extractor(images_t)
            dit_images_tp = self.dit_feature_extractor(images_tp)

            image_embeddings, pooled_projections = self.encode(projector_images_t)

            # Semantic head (only when enabled and labels provided)
            semantic_loss = torch.tensor(0.0, device=images_t.device)
            if self.use_semantic_head and "labels" in inputs:
                projector_images_tp = self.projector_feature_extractor(images_tp)
                _, pooled_tp = self.encode(projector_images_tp)

                pos = getattr(self, "semantic_position", "pooled")
                if pos == "pooled":
                    delta_sem = pooled_tp - pooled_projections
                elif pos == "proj":
                    embeds_tp, _ = self.encode(projector_images_tp)
                    delta_sem = embeds_tp.mean(dim=1) - image_embeddings.mean(dim=1)
                else:
                    delta_sem = pooled_tp - pooled_projections

                semantic_logits = self.semantic_head(delta_sem.float())
                ce_fn = nn.CrossEntropyLoss()
                semantic_loss = sum(
                    ce_fn(logits, inputs["labels"][field].to(logits.device))
                    for field, logits in semantic_logits.items()
                ) / len(semantic_logits)

            # Diffusion loss
            latents = self.vae_encode(dit_images_tp)
            latents = latents.to(dtype=self.dtype)
            noise = torch.randn_like(latents)

            u = compute_density_for_timestep_sampling(
                weighting_scheme=self.weighting_scheme,
                batch_size=bsz,
                logit_mean=self.logit_mean,
                logit_std=self.logit_std,
                mode_scale=self.mode_scale,
            )
            indices = (u * self.scheduler_copy.config.num_train_timesteps).long()
            timesteps = self.scheduler_copy.timesteps[indices].to(device=latents.device)
            sigmas = self.get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
            noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

            model_pred = self.DiT(
                hidden_states=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=image_embeddings,
                pooled_projections=pooled_projections,
                return_dict=False,
            )[0]

            weighting = compute_loss_weighting_for_sd3(weighting_scheme="logit_normal", sigmas=sigmas)
            target = noise - latents
            diffusion_loss = criterion(weighting, model_pred, target)

            outputs["loss"] = diffusion_loss + self.semantic_lambda * semantic_loss
            outputs["loss_diffusion"] = diffusion_loss.detach()
            outputs["loss_semantic"] = semantic_loss.detach()
            return outputs

        # Legacy single-image mode
        images = inputs["images"]
        bsz = images.shape[0]

        projector_images = self.projector_feature_extractor(images)
        dit_images = self.dit_feature_extractor(images)

        image_embeddings, pooled_projections = self.encode(projector_images)

        latents = self.vae_encode(dit_images)
        latents = latents.to(dtype=self.dtype)

        noise = torch.randn_like(latents)

        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.weighting_scheme,
            batch_size=bsz,
            logit_mean=self.logit_mean,
            logit_std=self.logit_std,
            mode_scale=self.mode_scale,
        )

        indices = (u * self.scheduler_copy.config.num_train_timesteps).long()
        timesteps = self.scheduler_copy.timesteps[indices].to(device=latents.device)
        sigmas = self.get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
        noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
        model_pred = self.DiT(
            hidden_states=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=image_embeddings,
            pooled_projections=pooled_projections,
            return_dict=False,
        )[0]

        weighting = compute_loss_weighting_for_sd3(weighting_scheme="logit_normal", sigmas=sigmas)

        target = noise - latents

        loss = criterion(weighting, model_pred, target)

        outputs["loss"] = loss
        return outputs

    @torch.no_grad()
    def eval_step(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
        images = inputs["images"]

        projector_images = self.projector_feature_extractor(images)
        do_classifier_free_guidance = self.guidance_scale > 1.0

        # Encode input image condition
        image_embeddings, pooled_projections = self.encode(
            projector_images,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, self.num_inference_steps, self.device, timesteps=None
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        num_channels_latents = self.DiT.config.in_channels
        generator = inputs["generator"]

        latents = self.prepare_latents(
            projector_images.shape[0],
            num_channels_latents,
            self.height,
            self.width,
            self.dtype,
            self.device,
            generator,
            latents=None,
        )

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                timestep = t.expand(latent_model_input.shape[0])
                # predict the noise residual
                noise_pred = self.DiT(
                    hidden_states=latent_model_input,  # [bs, 16, 96, 96]
                    timestep=timestep,  # [bs]
                    encoder_hidden_states=image_embeddings,  # [bs, 2, 4096]
                    pooled_projections=pooled_projections,  # [bs, 2048]
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # Post-processing
        latents = latents.to(dtype=self.dtype)
        gen_img = self.vae_decode(latents)

        outputs["images"] = gen_img
        return outputs

    @torch.no_grad()
    def interpolation_eval(self, image1, image2, generator, tokens=None, num_interpolation=5):
        do_classifier_free_guidance = self.guidance_scale > 1.0

        # Encode input images
        emb1, pooled1 = self.encode(image1, do_classifier_free_guidance)
        emb2, pooled2 = self.encode(image2, do_classifier_free_guidance)

        # Create interpolated embeddings
        interpolated_images = []
        for alpha in torch.linspace(0, 1, steps=num_interpolation):
            image_embeddings = emb1 * (1 - alpha) + emb2 * alpha
            pooled_projections = pooled1 * (1 - alpha) + pooled2 * alpha
            # pooled_projections = pooled1

            if tokens:
                image_embeddings[:, tokens, :] = emb1[:, tokens, :]

            # DDIM-like inference for each interpolation
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler,
                self.num_inference_steps,
                self.device,
                timesteps=None,
            )
            num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

            num_channels_latents = self.DiT.config.in_channels

            latents = self.prepare_latents(
                1,
                num_channels_latents,
                self.height,
                self.width,
                self.dtype,
                self.device,
                generator,
                latents=None,
            )

            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    timestep = t.expand(latent_model_input.shape[0])

                    noise_pred = self.DiT(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=image_embeddings,
                        pooled_projections=pooled_projections,
                        return_dict=False,
                    )[0]

                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()

            # Decode and store image
            latents = latents.to(dtype=self.dtype)
            gen_img = self.vae_decode(latents)
            interpolated_images.append(gen_img)

        # Concatenate all results along batch dimension
        return torch.cat(interpolated_images, dim=0)

    @torch.no_grad()
    def get_delta_action(self, start, end):
        do_classifier_free_guidance = self.guidance_scale > 1.0

        emb_start, pooled_start = self.encode(start, do_classifier_free_guidance)
        emb_end, pooled_end = self.encode(end, do_classifier_free_guidance)

        delta_emb = emb_end - emb_start
        delta_pooled = pooled_end - pooled_start

        return delta_emb, delta_pooled

    @torch.no_grad()
    def delta_interpolation(self, image, start, end, generator):
        do_classifier_free_guidance = self.guidance_scale > 1.0

        emb, pooled = self.encode(image, do_classifier_free_guidance)

        delta_emb, delta_pooled = self.get_delta_action(start, end)

        image_embeddings = emb + delta_emb
        pooled_projections = pooled + delta_pooled

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            self.num_inference_steps,
            self.device,
            timesteps=None,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        num_channels_latents = self.DiT.config.in_channels

        latents = self.prepare_latents(
            1,
            num_channels_latents,
            self.height,
            self.width,
            self.dtype,
            self.device,
            generator,
            latents=None,
        )

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.DiT(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=image_embeddings,
                    pooled_projections=pooled_projections,
                    return_dict=False,
                )[0]

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        latents = latents.to(dtype=self.dtype)
        gen_img = self.vae_decode(latents)
        return gen_img

    def forward(self, inputs, **kwargs):
        outputs = {}

        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed)
        inputs["generator"] = generator

        if self.training:
            outputs = self.train_step(inputs, outputs, **kwargs)
        else:
            outputs = self.eval_step(inputs, outputs, **kwargs)

        inputs.pop("generator", None)
        return outputs


if __name__ == "__main__":
    from omegaconf import OmegaConf

    args = OmegaConf.load("./configs/debug.yaml")

    model = RenderNet(args).to("cuda")

    images = torch.randn((4, 3, args.data.img_size, args.data.img_size))
    images = images.to("cuda")

    inputs = {"images": images}
    outputs = model(inputs)
    print(outputs)
