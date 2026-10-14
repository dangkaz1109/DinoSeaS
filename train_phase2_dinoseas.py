"""
DinoSeaS: Phase 2 - Joint Multi-Class Synthesis Training Script
Executes the unified optimization loop:
  L_DinoSeaS = (L_df + L_ob) + eta_1 L_M + eta_2 L_GDCR
Using:
  - Continuous CAR Class Anchor Injection
  - Hybrid Dual-Attention U-Net
  - Defect Noisy Bottleneck B_df
  - One-Step Tweedie Latent Decoding
  - CAR-Guided Refined Mask Prediction (CAR-MRM)
"""

import argparse
import logging
import os
import sys
import yaml
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.dinoseas_model import DinoSeaSModel
from datasets.multiclass_dataset import MultiClassAnomalyDataset, MVTEC_CATEGORIES
from train_phase1_evaluator import load_dinov2_backbone


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase2_DinoSeaS")


class SimpleNoiseScheduler:
    """
    DDPM / DDIM noise scheduler maintaining alpha, beta, and alphas_cumprod
    for diffusion and Tweedie one-step denoising.
    """
    def __init__(self, num_timesteps: int = 1000, beta_start: float = 0.00085, beta_end: float = 0.0120):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps) ** 2
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(
        self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1.0 - alphas_cumprod[timesteps]) ** 0.5

        sqrt_alpha_prod = sqrt_alpha_prod.view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.view(-1, 1, 1, 1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples


def load_sd_backbones(config: dict, device: str) -> Tuple[nn.Module, nn.Module]:
    """
    Loads UNet2DConditionModel and AutoencoderKL from runwayml/stable-diffusion-v1-5
    or constructs architecture-compatible standalone modules.
    """
    try:
        from diffusers import AutoencoderKL, UNet2DConditionModel
        model_id = config["backbones"].get("diffusion_model_id", "runwayml/stable-diffusion-v1-5")
        logger.info(f"Loading Stable Diffusion components from {model_id}...")
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", local_files_only=True)
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", local_files_only=True)
        return unet.to(device), vae.to(device)
    except Exception as e:
        logger.warning(f"Could not load official diffusers weights locally ({e}). Initializing standalone surrogate backbones.")

    # Standalone Lightweight Surrogate UNet & VAE for unit testing and offline training
    class SurrogateUNet(nn.Module):
        def __init__(self, in_ch: int = 4, out_ch: int = 4, embed_dim: int = 768):
            super().__init__()
            self.conv_in = nn.Conv2d(in_ch, 64, kernel_size=3, padding=1)
            self.time_proj = nn.Linear(1, 64)
            self.text_proj = nn.Linear(embed_dim, 64)
            self.mid_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.conv_out = nn.Conv2d(64, out_ch, kernel_size=3, padding=1)

            # Up features for RMP
            self.up2_proj = nn.Conv2d(64, 1280, kernel_size=1)
            self.up3_proj = nn.Conv2d(64, 640, kernel_size=1)

            self.last_up2_feature = None
            self.last_up3_feature = None

        def forward(self, sample, timestep, encoder_hidden_states=None, **kwargs):
            B = sample.shape[0]
            t_emb = self.time_proj(timestep.float().view(-1, 1)).view(B, 64, 1, 1)
            x = self.conv_in(sample) + t_emb
            x = F.silu(x)
            x = self.mid_conv(x)

            # Store surrogate up-2 and up-3 features for RMP
            self.last_up2_feature = self.up2_proj(F.interpolate(x, size=(32, 32), mode="bilinear", align_corners=False))
            self.last_up3_feature = self.up3_proj(F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False))

            out = self.conv_out(x)
            return out

    class SurrogateVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder_conv = nn.Conv2d(3, 4, kernel_size=8, stride=8)
            self.decoder_conv = nn.Conv2d(4, 3, kernel_size=3, padding=1)

        def encode(self, x):
            class Dist:
                def __init__(self, mean):
                    self.mean = mean
                def sample(self):
                    return self.mean
            class VAEOut:
                def __init__(self, mean):
                    self.latent_dist = Dist(mean)
            return VAEOut(self.encoder_conv(x))

        def decode(self, z):
            # Upsample 64x64 to 512x512
            x = F.interpolate(z, scale_factor=8, mode="bilinear", align_corners=False)
            img = self.decoder_conv(x)
            return (img,)

    unet = SurrogateUNet().to(device)
    vae = SurrogateVAE().to(device)
    return unet, vae


def parse_args():
    parser = argparse.ArgumentParser(description="DinoSeaS Phase 2 Joint Multi-Class Training")
    parser.add_argument(
        "--config",
        type=str,
        default="config/dinoseas_multiclass.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument("--steps", type=int, default=None, help="Override training steps")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--evaluator_ckpt", type=str, default=None, help="Path to Phase 1 evaluator weights")
    parser.add_argument("--fp16", action="store_true", default=True, help="Enable FP16 mixed precision training")
    parser.add_argument("--no_fp16", action="store_false", dest="fp16", help="Disable FP16 mixed precision")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def train_dinoseas():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device)
    total_steps = args.steps or config["training"].get("max_steps", 40000)
    batch_size = args.batch_size or config["training"].get("batch_size", 2)
    output_dir = config["experiment"].get("output_dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    seed = config["experiment"].get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    use_fp16 = getattr(args, "fp16", True) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    if use_fp16:
        logger.info("FP16 Mixed Precision & Tensor Core acceleration ENABLED.")

    # 1. Load Backbones
    logger.info("Initializing backbones (DINOv2, UNet, VAE)...")
    vit_encoder = load_dinov2_backbone(config, str(device))
    unet, vae = load_sd_backbones(config, str(device))

    # 2. Instantiate Unified DinoSeaS Model
    logger.info("Instantiating DinoSeaS Unified Architecture...")
    model = DinoSeaSModel(config=config, unet=unet, vae=vae, dinov2=vit_encoder).to(device)

    # Load Phase 1 evaluator weights if specified
    eval_ckpt = args.evaluator_ckpt or os.path.join(output_dir, "evaluator_phase1_final.pth")
    if os.path.exists(eval_ckpt):
        logger.info(f"Loading Phase 1 Evaluator weights from {eval_ckpt}...")
        state_dict = torch.load(eval_ckpt, map_location=device)
        model.evaluator.load_state_dict(state_dict, strict=False)

    # Freeze DINOv2 and VAE
    if config["backbones"].get("freeze_dinov2", True) and model.dinov2 is not None:
        for p in model.dinov2.parameters():
            p.requires_grad = False
    if config["backbones"].get("freeze_vae", True):
        for p in model.vae_wrapper.parameters():
            p.requires_grad = False

    # 3. Setup Noise Scheduler
    noise_scheduler = SimpleNoiseScheduler()

    # 4. Optimizer: Optimize Hybrid Attention, ob_proj, df_bottleneck, and rmp_branch
    opt_cfg = config["training"]["optimizer"]
    trainable_params = [
        {"params": model.ob_proj.parameters(), "lr": opt_cfg.get("lr_prompt_tokens", 5e-4)},
        {"params": model.df_bottleneck.parameters(), "lr": opt_cfg.get("lr_prompt_tokens", 5e-4)},
        {"params": model.rmp_branch.parameters(), "lr": opt_cfg.get("lr_rmp_branch", 1e-4)},
    ]

    # Add hybrid attention parameters if registered
    if hasattr(model, "registered_hybrid_attns"):
        trainable_params.append({
            "params": model.registered_hybrid_attns.parameters(),
            "lr": opt_cfg.get("lr_unet_attention", 1e-5),
        })

    optimizer = torch.optim.AdamW(
        trainable_params,
        weight_decay=opt_cfg.get("weight_decay", 1e-2),
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # 5. Dataset & Dataloader: Multi-Class Paired Dataset
    data_path = config["experiment"].get("data_path", "/home/lablee/AD/mvtec_data")
    dataset = MultiClassAnomalyDataset(
        root_dir=data_path,
        categories=MVTEC_CATEGORIES,
        split="train",
        image_size=config["experiment"].get("image_size", 512),
        embed_dim=config["dinomaly2"]["embed_dim"],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False,
        drop_last=True,
    )

    logger.info(f"Phase 2: Starting Joint Multi-Class Training on {len(dataset)} items for {total_steps} steps...")
    model.train()

    step = 0
    progress_bar = tqdm(total=total_steps, desc="Phase 2 DinoSeaS")

    while step < total_steps:
        for batch in dataloader:
            # Transfer batch to device
            batch_dev = {
                "x_df": batch["x_df"].to(device, non_blocking=True),
                "x_ob": batch["x_ob"].to(device, non_blocking=True),
                "mask_df": batch["mask_df"].to(device, non_blocking=True),
                "e_df": batch["e_df"].to(device, non_blocking=True),
                "e_ob": batch["e_ob"].to(device, non_blocking=True),
            }

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_fp16):
                loss_dict = model.train_step(batch_dev, noise_scheduler)
                loss_total = loss_dict["loss_total"]

            if torch.isnan(loss_total) or torch.isinf(loss_total):
                raise FloatingPointError(f"NaN/Inf loss encountered in step {step}")

            scaler.scale(loss_total).backward()
            scaler.unscale_(optimizer)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                [p for group in trainable_params for p in group["params"]],
                max_norm=config["training"].get("clip_grad_norm", 1.0),
            )

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            step += 1
            progress_bar.update(1)
            progress_bar.set_postfix({
                "loss": f"{loss_total.item():.4f}",
                "sd_df": f"{loss_dict['loss_sd_df'].item():.3f}",
                "da": f"{loss_dict['loss_da'].item():.3f}",
                "mask": f"{loss_dict['loss_mask'].item():.3f}",
                "gdcr": f"{loss_dict['loss_gdcr'].item():.3f}",
            })

            if step % config["training"].get("save_every", 2000) == 0 or step == total_steps:
                ckpt_path = os.path.join(output_dir, f"dinoseas_model_step{step}.pth")
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                }, ckpt_path)
                logger.info(f"Saved checkpoint to {ckpt_path}")

            if step % 500 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

            if step >= total_steps:
                break

    progress_bar.close()
    final_ckpt = os.path.join(output_dir, "dinoseas_model_final.pth")
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "config": config,
    }, final_ckpt)
    logger.info(f"Phase 2 Complete! Master model saved to {final_ckpt}")


if __name__ == "__main__":
    train_dinoseas()
