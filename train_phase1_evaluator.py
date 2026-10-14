"""
DinoSeaS: Phase 1 - Evaluator Warm-Up Script
Trains the Dinomaly2 Loose Group Evaluator on multi-class normal images
using L_loose with sg_0.1 hard-mining gradient modulation.
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

from models.dinomaly2_core import Dinomaly2LooseEvaluator
from models.losses import DinoSeaSLossEngine
from datasets.multiclass_dataset import MultiClassAnomalyDataset, MVTEC_CATEGORIES


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase1_Evaluator")


def parse_args():
    parser = argparse.ArgumentParser(description="DinoSeaS Phase 1 Evaluator Warmup")
    parser.add_argument(
        "--config",
        type=str,
        default="config/dinoseas_multiclass.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument("--steps", type=int, default=None, help="Override training steps")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_dinov2_backbone(config: dict, device: str) -> nn.Module:
    """
    Loads official DINOv2 ViT backbone from local checkpoint path using internal package.
    """
    local_pth = config["backbones"].get("local_dinov2_checkpoint", "")
    model_id = config["backbones"].get("dinov2_model_id", "facebook/dinov2-base")
    is_base = "base" in model_id or "vitb" in model_id
    arch_func = "vit_base" if is_base else "vit_small"
    patch_size = 14

    logger.info(f"Loading local DINOv2 backbone ({arch_func}14)...")
    try:
        from dinov2.models import vision_transformer as vision_transformer_dinov2
        model = getattr(vision_transformer_dinov2, arch_func)(
            patch_size=patch_size,
            img_size=518,
            block_chunks=0,
            init_values=1e-8,
            interpolate_antialias=False,
            interpolate_offset=0.1,
        )
        if local_pth and os.path.exists(local_pth):
            logger.info(f"Loading local DINOv2 weights from {local_pth}")
            state_dict = torch.load(local_pth, map_location="cpu")
            msg = model.load_state_dict(state_dict, strict=True)
            logger.info(f"DINOv2 weights successfully loaded: {msg}")
        else:
            raise FileNotFoundError(f"Local DINOv2 weights not found at {local_pth}")
    except Exception as e:
        logger.warning(f"Internal dinov2 loading failed ({e}). Falling back to torch.hub...")
        arch = "dinov2_vitb14" if is_base else "dinov2_vits14"
        model = torch.hub.load("facebookresearch/dinov2", arch, pretrained=False)
        if local_pth and os.path.exists(local_pth):
            state_dict = torch.load(local_pth, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)

    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def train_evaluator():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device)
    total_steps = args.steps or config["training"].get("phase1_evaluator_steps", 5000)
    batch_size = args.batch_size or config["training"].get("batch_size", 4)
    output_dir = config["experiment"].get("output_dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Set random seed
    seed = config["experiment"].get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Initialize ViT Backbone
    vit_encoder = load_dinov2_backbone(config, str(device))

    # Initialize Dinomaly2 Loose Evaluator
    dinom_cfg = config["dinomaly2"]
    evaluator = Dinomaly2LooseEvaluator(
        embed_dim=dinom_cfg["embed_dim"],
        group1_idx=dinom_cfg["loose_groups"]["group_1_shallow"],
        group2_idx=dinom_cfg["loose_groups"]["group_2_deep"],
        bottleneck_layers=dinom_cfg["noisy_bottleneck"]["num_layers"],
        dropout_p=dinom_cfg["noisy_bottleneck"]["dropout_prob"],
        decoder_depth=dinom_cfg["decoder"]["num_layers_per_group"],
        num_heads=dinom_cfg["decoder"]["num_heads"],
    ).to(device)

    # Initialize Loss Engine
    loss_engine = DinoSeaSLossEngine(config).to(device)

    # Optimizer for Evaluator: bottleneck + decoders
    lr = config["training"]["optimizer"].get("lr_dinomaly2_decoder", 2.0e-4)
    weight_decay = config["training"]["optimizer"].get("weight_decay", 1.0e-2)
    optimizer = torch.optim.AdamW(evaluator.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # Dataset & Dataloader: Normal multi-class samples
    data_path = config["experiment"].get("data_path", "/home/lablee/AD/mvtec_data")
    dataset = MultiClassAnomalyDataset(
        root_dir=data_path,
        categories=MVTEC_CATEGORIES,
        split="train",
        image_size=config["experiment"].get("image_size", 512),
        embed_dim=dinom_cfg["embed_dim"],
    )

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
    )

    logger.info(f"Phase 1: Starting Evaluator Warm-Up on {len(dataset)} items for {total_steps} steps...")
    evaluator.train()

    step = 0
    progress_bar = tqdm(total=total_steps, desc="Phase 1 Warm-up")

    while step < total_steps:
        for batch in dataloader:
            x_ob = batch["x_ob"].to(device)  # Normal images only

            # Resize to ViT input size (448 x 448 -> 32 x 32 patches)
            x_ob_448 = F.interpolate(x_ob, size=(448, 448), mode="bilinear", align_corners=False)

            with torch.no_grad():
                vit_outs = vit_encoder.get_intermediate_layers(x_ob_448, n=12, return_class_token=True)

            dino_out = evaluator(vit_outs)
            loss = loss_engine.compute_evaluator_loose_loss(dino_out)

            if torch.isnan(loss) or torch.isinf(loss):
                raise FloatingPointError(f"NaN/Inf loss encountered at step {step}")

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(evaluator.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            step += 1
            progress_bar.update(1)
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "d_cos_g1": f"{dino_out['d_cos_g1'].mean().item():.4f}",
                "d_cos_g2": f"{dino_out['d_cos_g2'].mean().item():.4f}",
            })

            if step % 1000 == 0 or step == total_steps:
                save_path = os.path.join(output_dir, f"evaluator_phase1_step{step}.pth")
                torch.save(evaluator.state_dict(), save_path)
                logger.info(f"Saved checkpoint to {save_path}")

            if step >= total_steps:
                break

    progress_bar.close()
    final_path = os.path.join(output_dir, "evaluator_phase1_final.pth")
    torch.save(evaluator.state_dict(), final_path)
    logger.info(f"Phase 1 Complete! Final weights saved to {final_path}")


if __name__ == "__main__":
    train_evaluator()
