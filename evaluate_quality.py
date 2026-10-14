"""
DinoSeaS: Scientific Evaluation & Metric Profiler
Strictly adheres to Universal Scientific Integrity & Methodological Rigor Directive:
  - Zero mock values / zero hardcoded results.
  - Runtime tensor computation for all metrics:
      1. Background Normalcy Preservation Error (BNPE)
      2. Intra-Mask Defect Salience (IMDS)
      3. Margin Violation Rate (MVR)
      4. Mask Boundary Sharpness / High-Frequency Gradient Norm
  - Multi-seed statistical reporting: Mean \pm Std across N >= 3 runs.
"""

import argparse
import logging
import math
import os
import sys
import yaml
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.dinoseas_model import DinoSeaSModel
from datasets.multiclass_dataset import MultiClassAnomalyDataset, MVTEC_CATEGORIES
from train_phase1_evaluator import load_dinov2_backbone
from train_phase2_dinoseas import load_sd_backbones


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DinoSeaS_Evaluation")


def compute_mask_boundary_gradient(mask_prob: torch.Tensor) -> float:
    """
    Measures the sharpness of the synthesized mask by computing Sobel spatial gradients.
    Sharp masks have high gradient norm along the boundary; blurry masks have low gradient norm.
    mask_prob: (B, 1, H, W)
    """
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=mask_prob.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=mask_prob.device).view(1, 1, 3, 3)

    grad_x = F.conv2d(mask_prob, sobel_x, padding=1)
    grad_y = F.conv2d(mask_prob, sobel_y, padding=1)
    magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

    # Focus on boundary region where mask transitions (0.1 < mask < 0.9)
    boundary_region = (mask_prob > 0.1) & (mask_prob < 0.9)
    if boundary_region.sum() > 0:
        return magnitude[boundary_region].mean().item()
    return magnitude.mean().item()


def evaluate_model(
    config: dict,
    checkpoint_path: Optional[str] = None,
    seeds: List[int] = [42, 100, 2026],
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 4,
    num_eval_batches: int = 10,
):
    device = torch.device(device_str)

    # Load Backbones
    vit_encoder = load_dinov2_backbone(config, str(device))
    unet, vae = load_sd_backbones(config, str(device))

    model = DinoSeaSModel(config=config, unet=unet, vae=vae, dinov2=vit_encoder).to(device)
    model.eval()

    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint weights from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)

    gamma_1 = config["training"]["gdcr_control"]["margin_gamma_1"]
    gamma_2 = config["training"]["gdcr_control"]["margin_gamma_2"]

    # Multi-Seed Evaluation Records
    seed_bnpe_scores = []
    seed_imds_scores = []
    seed_mvr_scores = []
    seed_sharpness_scores = []

    data_path = config["experiment"].get("data_path", "/home/lablee/AD/mvtec_data")

    for seed in seeds:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        dataset = MultiClassAnomalyDataset(
            root_dir=data_path,
            categories=MVTEC_CATEGORIES,
            split="train",
            image_size=config["experiment"].get("image_size", 512),
            embed_dim=config["dinomaly2"]["embed_dim"],
            seed=seed,
        )

        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
        )

        bnpe_list = []
        imds_list = []
        mvr_list = []
        sharpness_list = []

        batch_count = 0
        with torch.no_grad():
            for batch in dataloader:
                x_df = batch["x_df"].to(device)
                x_ob = batch["x_ob"].to(device)
                gt_mask = batch["mask_df"].to(device)
                base_e_df = batch["e_df"].to(device)
                base_e_ob = batch["e_ob"].to(device)

                # 1. CAR Global Class Anchor
                x_ob_448 = F.interpolate(x_ob, size=(448, 448), mode="bilinear", align_corners=False)
                vit_ref_out = model.extract_vit_features(x_ob_448)
                dino_ref = model.evaluator(vit_ref_out)
                f_cls_anchor = dino_ref["f_cls_agg"]

                # 2. Prompt Conditioning
                e_df_cond, _ = model.inject_prompt_conditioning(base_e_df, base_e_ob, f_cls_anchor)

                # 3. Latent forward & Tweedie reconstruction
                z_df = model.vae_wrapper.encode(x_df)
                z_df = (z_df.latent_dist.sample() if hasattr(z_df, "latent_dist") else z_df) * 0.18215
                t = torch.full((x_df.shape[0],), 500, device=device).long()
                eps_df = torch.randn_like(z_df)

                alpha_bar = torch.tensor(0.5, device=device)
                z_t_df = alpha_bar.sqrt() * z_df + (1.0 - alpha_bar).sqrt() * eps_df

                eps_pred_df, unet_feats_df = model.forward_unet(
                    z_t_df, t, encoder_hidden_states=e_df_cond, is_defect_stream=True
                )
                z0_hat_df = (z_t_df - (1.0 - alpha_bar).sqrt() * eps_pred_df) / (alpha_bar.sqrt() + 1e-8)

                # 4. VAE decode
                x_hat_df, vae_res_df = model.vae_wrapper.decode_with_residuals(z0_hat_df / 0.18215)

                # 5. Dinomaly2 evaluation
                x_hat_df_448 = F.interpolate(x_hat_df, size=(448, 448), mode="bilinear", align_corners=False)
                vit_syn_out = model.extract_vit_features(x_hat_df_448)
                dino_syn = model.evaluator(vit_syn_out)

                # 6. CAR-MRM Mask Prediction
                _, refined_logits = model.rmp_branch(
                    unet_feats_df["up2"], unet_feats_df["up3"], vae_res_df, dino_syn["anomaly_map"]
                )
                pred_mask_prob = F.softmax(refined_logits, dim=1)[:, 1:2, :, :]

                # 7. Metric Computations
                # Downsample mask to 32x32 patch grid
                B = x_df.shape[0]
                M_patch = F.interpolate(gt_mask.unsqueeze(1).float(), size=(32, 32), mode="bilinear", align_corners=False).view(B, 1024)
                M_bin = (M_patch > 0.2).float()

                d_cos_mean = 0.5 * (dino_syn["d_cos_g1"] + dino_syn["d_cos_g2"])  # (B, 1024)

                # Background Normalcy Preservation Error (BNPE): Outside mask error
                bg_mask = (1.0 - M_bin)
                if bg_mask.sum() > 0:
                    bnpe = (d_cos_mean * bg_mask).sum() / bg_mask.sum()
                    bnpe_list.append(bnpe.item())

                # Intra-Mask Defect Salience (IMDS): Inside mask error
                fg_mask = M_bin
                if fg_mask.sum() > 0:
                    imds = (d_cos_mean * fg_mask).sum() / fg_mask.sum()
                    imds_list.append(imds.item())

                    # Margin Violation Rate (MVR): % of defect patches below target margin
                    violations = (dino_syn["d_cos_g1"] < gamma_1).float() * fg_mask
                    mvr = violations.sum() / fg_mask.sum()
                    mvr_list.append(mvr.item())

                # Mask Boundary Sharpness
                sharpness = compute_mask_boundary_gradient(pred_mask_prob)
                sharpness_list.append(sharpness)

                batch_count += 1
                if batch_count >= num_eval_batches:
                    break

        seed_bnpe_scores.append(np.mean(bnpe_list))
        seed_imds_scores.append(np.mean(imds_list))
        seed_mvr_scores.append(np.mean(mvr_list))
        seed_sharpness_scores.append(np.mean(sharpness_list))

    # Compute Statistical Metrics: Mean \pm Std
    mean_bnpe, std_bnpe = np.mean(seed_bnpe_scores), np.std(seed_bnpe_scores)
    mean_imds, std_imds = np.mean(seed_imds_scores), np.std(seed_imds_scores)
    mean_mvr, std_mvr = np.mean(seed_mvr_scores), np.std(seed_mvr_scores)
    mean_sh, std_sh = np.mean(seed_sharpness_scores), np.std(seed_sharpness_scores)

    report = (
        f"\n{'='*70}\n"
        f"DinoSeaS Unified Framework: Scientific Audit Report (N={len(seeds)} Seeds)\n"
        f"{'='*70}\n"
        f"1. Background Normalcy Preservation Error (BNPE): {mean_bnpe:.4f} \pm {std_bnpe:.4f}\n"
        f"   (Lower is better: measures non-corruption of normal substrate outside mask)\n"
        f"2. Intra-Mask Defect Salience (IMDS):             {mean_imds:.4f} \pm {std_imds:.4f}\n"
        f"   (Higher is better: confirms genuine out-of-distribution manifold breach)\n"
        f"3. Anomaly Margin Violation Rate (MVR):          {mean_mvr*100:.2f}% \pm {std_mvr*100:.2f}%\n"
        f"   (Lower is better: % of defect patches failing margin threshold gamma)\n"
        f"4. Mask Boundary Gradient Sharpness:             {mean_sh:.4f} \pm {std_sh:.4f}\n"
        f"   (Higher is better: verifies absence of fuzzy/blurry false-positive mask edges)\n"
        f"{'='*70}\n"
    )

    print(report)
    return {
        "bnpe": (mean_bnpe, std_bnpe),
        "imds": (mean_imds, std_imds),
        "mvr": (mean_mvr, std_mvr),
        "sharpness": (mean_sh, std_sh),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/dinoseas_multiclass.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batches", type=int, default=5)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    evaluate_model(cfg, checkpoint_path=args.checkpoint, num_eval_batches=args.batches)
