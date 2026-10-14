"""
DinoSeaS: Module D - Unified Loss Engine
Implements:
  1. Decoupled Anomaly Alignment Loss (L_DA)
  2. Multi-Scale Mask Focal Loss (L_M)
  3. Dinomaly2 Loose Reconstruction Loss with sg_0.1 hard-mining (L_loose)
  4. Group-Decomposed Contrastive Reconstruction Loss (L_GDCR)
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dinomaly2_core import stop_gradient_scale
from .hybrid_attention import HybridDualCrossAttention


class MultiScaleFocalLoss(nn.Module):
    r"""
    Focal Loss for binary segmentation masks:
      FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)
    Supervises both coarse (64x64) and fine (512x512) mask prediction heads.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, eps: float = 1e-7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, 2, H, W) raw class logits
            targets: (B, H, W) binary ground truth labels in {0, 1}
        Returns:
            Scalar focal loss
        """
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            raise FloatingPointError("NaN or Inf detected in mask logits input to MultiScaleFocalLoss")

        targets = targets.long()
        probs = F.softmax(logits, dim=1)  # (B, 2, H, W)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B, H, W)
        p_t = p_t.clamp(min=self.eps, max=1.0 - self.eps)

        ce_loss = F.cross_entropy(logits, targets, reduction="none")  # (B, H, W)
        focal_weight = (1.0 - p_t) ** self.gamma
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)

        loss = (alpha_t * focal_weight * ce_loss).mean()
        return loss


class DinoSeaSLossEngine(nn.Module):
    """
    Unified Loss Engine for DinoSeaS.
    Couples generator (SeaS) and evaluator (Dinomaly2) in closed-loop optimization.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.cfg = config
        gdcr_cfg = config.get("training", {}).get("gdcr_control", {})
        dinom_cfg = config.get("dinomaly2", {}).get("loss_modulation", {})

        self.focal = MultiScaleFocalLoss(
            alpha=gdcr_cfg.get("focal_alpha", 0.25),
            gamma=gdcr_cfg.get("focal_gamma", 2.0),
        )

        self.quantile_q = dinom_cfg.get("hard_mining_quantile", 0.90)
        self.sg_scale = dinom_cfg.get("stop_grad_scale", 0.10)

        self.lambda_1 = gdcr_cfg.get("lambda_1_geometric", 1.5)
        self.lambda_2 = gdcr_cfg.get("lambda_2_semantic", 1.0)
        self.gamma_1 = gdcr_cfg.get("margin_gamma_1", 0.35)
        self.gamma_2 = gdcr_cfg.get("margin_gamma_2", 0.45)

    def set_steering_weights(self, lambda_1: float, lambda_2: float):
        """Allows dynamic taxonomy steering during training or inference."""
        self.lambda_1 = float(lambda_1)
        self.lambda_2 = float(lambda_2)

    def compute_da_loss(
        self,
        hybrid_attn_layers: List[HybridDualCrossAttention],
        gt_mask_512: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Decoupled Anomaly Alignment Loss (L_DA):
          L_DA = sum_l ( || (1/N) sum_n A_{n,l}^df - M^l ||_2^2 + || A_{ob,l}^LA \odot M^l ||_2^2 )
        Term 1: Attracts defect tokens to the spatial ground-truth mask M^l.
        Term 2: Suppresses normal object tokens inside the defect region M^l.
        """
        if not hybrid_attn_layers:
            return torch.tensor(0.0, device=gt_mask_512.device, requires_grad=True)

        B = gt_mask_512.shape[0]
        loss_da = torch.tensor(0.0, device=gt_mask_512.device)
        valid_layers = 0

        for layer in hybrid_attn_layers:
            A_df = layer.recorded_A_df  # (B, HW, N_df)
            A_ob = layer.recorded_A_ob  # (B, HW)

            if A_df is None or A_ob is None:
                continue

            HW = A_ob.shape[1]
            r = int(math.isqrt(HW))
            if r * r != HW:
                # If non-square sequence length, skip or interpolate along 1D
                continue

            # Downsample ground-truth mask to layer resolution (r x r)
            mask_4d = gt_mask_512.unsqueeze(1).float()
            M_l = F.interpolate(mask_4d, size=(r, r), mode="nearest").view(B, HW)

            # Term 1: Defect attraction
            A_df_mean = A_df.mean(dim=-1)  # (B, HW)
            loss_attract = F.mse_loss(A_df_mean, M_l)

            # Term 2: Normal object token suppression inside M^l
            loss_suppress = torch.mean((A_ob * M_l) ** 2)

            loss_da = loss_da + (loss_attract + loss_suppress)
            valid_layers += 1

        if valid_layers > 0:
            loss_da = loss_da / valid_layers
        return loss_da

    def compute_dinomaly2_modulated_distance(
        self, g_k: torch.Tensor, g_hat_k: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes modulated cosine distance with sg_0.1 hard-mining:
        When d_cos > tau_k (90th percentile), applies stop_gradient_scale(0.10) to g_hat_k.
        g_k, g_hat_k: (B, N, D)
        """
        g_k = g_k.float()
        g_hat_k = g_hat_k.float()
        with torch.no_grad():
            d_raw = 1.0 - F.cosine_similarity(g_k, g_hat_k, dim=-1, eps=1e-6)  # (B, N)
            tau_k = torch.quantile(d_raw.flatten(), self.quantile_q)
            hard_mask = (d_raw > tau_k).unsqueeze(-1)                 # (B, N, 1)

        g_hat_mod = torch.where(
            hard_mask,
            stop_gradient_scale(g_hat_k, self.sg_scale),
            g_hat_k,
        )
        return 1.0 - F.cosine_similarity(g_k, g_hat_mod, dim=-1, eps=1e-6)     # (B, N)

    def compute_evaluator_loose_loss(
        self, dino_out: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Phase 1 Evaluator pre-training loss on clean normal data:
        Minimizes cosine distance on Group 1 and Group 2 with sg_0.1 hard-mining.
        """
        d_mod_g1 = self.compute_dinomaly2_modulated_distance(dino_out["g1"], dino_out["g1_hat"])
        d_mod_g2 = self.compute_dinomaly2_modulated_distance(dino_out["g2"], dino_out["g2_hat"])
        return 0.5 * (d_mod_g1.mean() + d_mod_g2.mean())

    def compute_gdcr_loss(
        self,
        dino_out: Dict[str, torch.Tensor],
        gt_mask_512: torch.Tensor,
    ) -> torch.Tensor:
        """
        Group-Decomposed Contrastive Reconstruction Loss (L_GDCR):
          - Background Normalcy Preservation (outside M):
              (1 - M) * 0.5 * [d_mod_g1 + d_mod_g2]
          - Targeted Anomaly Margin Enforcement (inside M):
              M * [lambda_1 * relu(gamma_1 - d_cos_g1) + lambda_2 * relu(gamma_2 - d_cos_g2)]
        """
        B = gt_mask_512.shape[0]

        # Downsample ground-truth mask to ViT patch grid (e.g. 32x32 = 1024 patches)
        num_patches = dino_out["g1"].shape[1]
        grid_size = int(math.isqrt(num_patches))

        mask_4d = gt_mask_512.unsqueeze(1).float()
        M_patch = F.interpolate(
            mask_4d, size=(grid_size, grid_size), mode="bilinear", align_corners=False
        ).view(B, num_patches)
        M_bin = (M_patch > 0.2).float()

        # 1. Background Normalcy Preservation (sg_0.1 modulated)
        d_mod_g1 = self.compute_dinomaly2_modulated_distance(dino_out["g1"], dino_out["g1_hat"])
        d_mod_g2 = self.compute_dinomaly2_modulated_distance(dino_out["g2"], dino_out["g2_hat"])
        bg_loss = ((1.0 - M_bin) * 0.5 * (d_mod_g1 + d_mod_g2)).sum() / torch.clamp(
            (1.0 - M_bin).sum(), min=1.0
        )

        # 2. Targeted Anomaly Margin Enforcement
        margin_g1 = F.relu(self.gamma_1 - dino_out["d_cos_g1"])
        margin_g2 = F.relu(self.gamma_2 - dino_out["d_cos_g2"])

        fg_loss = (
            M_bin * (self.lambda_1 * margin_g1 + self.lambda_2 * margin_g2)
        ).sum() / torch.clamp(M_bin.sum(), min=1.0)

        total_gdcr = bg_loss + fg_loss
        if torch.isnan(total_gdcr) or torch.isinf(total_gdcr):
            raise FloatingPointError("NaN or Inf detected in L_GDCR loss calculation")
        return total_gdcr

    def compute_mask_loss(
        self,
        coarse_df: torch.Tensor,
        refined_df: torch.Tensor,
        coarse_ob: torch.Tensor,
        refined_ob: torch.Tensor,
        gt_mask_512: torch.Tensor,
    ) -> torch.Tensor:
        """
        Multi-Scale Mask Supervision Loss (L_M):
          L_M = Focal(M^_df, M_df_64) + Focal(M'^_df, M_df_512)
              + Focal(M^_ob, 0_64)    + Focal(M'^_ob, 0_512)
        """
        gt_mask_64 = F.interpolate(
            gt_mask_512.unsqueeze(1).float(), size=(64, 64), mode="nearest"
        ).squeeze(1)

        zeros_64 = torch.zeros_like(gt_mask_64)
        zeros_512 = torch.zeros_like(gt_mask_512)

        loss_df_64 = self.focal(coarse_df, gt_mask_64)
        loss_df_512 = self.focal(refined_df, gt_mask_512)
        loss_ob_64 = self.focal(coarse_ob, zeros_64)
        loss_ob_512 = self.focal(refined_ob, zeros_512)

        return loss_df_64 + loss_df_512 + loss_ob_64 + loss_ob_512
