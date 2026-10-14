r"""
DinoSeaS: Module C - CAR-Guided Mask Refinement Module (CAR-MRM)
Refined Mask Prediction (RMP) Branch with VAE decoder skip residuals
and Dinomaly2 CAR cosine anomaly map semantic gating:
  F_{out}^{(k)} = Conv3x3( Upsample(F_{out}^{(k-1)}) \odot \phi_v(F_k^{res}) \odot (1 + \sigma(\Psi_k(A^{\uparrow k}))) )
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskRefinementModuleBlock(nn.Module):
    r"""
    Cascaded MRM block at resolution k:
      1. Upsamples previous semantic features F_{out}^{(k-1)} to the target VAE residual resolution.
      2. Projects high-resolution VAE residual \phi_v(F_k^{res}).
      3. Upsamples and projects CAR Anomaly Map: Gate = 1 + \sigma(\Psi_k(A^{\uparrow k})).
      4. Fuses: F_{fused} = F_{up} \odot \phi_v(F_k^{res}) \odot Gate.
      5. Refines with residual double-conv block.
    """
    def __init__(self, in_ch: int, vae_ch: int, out_ch: int):
        super().__init__()
        self.up_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )
        self.vae_proj = nn.Sequential(
            nn.Conv2d(vae_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )
        # Spatial anomaly semantic gate: maps 1-channel anomaly map to out_ch
        self.anomaly_gate_proj = nn.Sequential(
            nn.Conv2d(1, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=1),
        )
        # Refinement residual block
        self.refine_conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )
        self.residual_conv = nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False)

    def forward(
        self,
        f_prev: torch.Tensor,
        f_vae_res: torch.Tensor,
        car_anomaly_map: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            f_prev: Features from previous stage (B, in_ch, H_prev, W_prev)
            f_vae_res: High-resolution VAE decoder residual (B, vae_ch, H_target, W_target)
            car_anomaly_map: CAR anomaly map (B, 1, H_dino, W_dino) or (B, N)
        Returns:
            Refined features at target resolution: (B, out_ch, H_target, W_target)
        """
        target_size = f_vae_res.shape[-2:]

        # Upsample coarse features to target resolution
        f_up = F.interpolate(f_prev, size=target_size, mode="bilinear", align_corners=False)
        f_up = self.up_conv(f_up)

        # Project high-res VAE residual (with adaptive channel matching)
        expected_ch = self.vae_proj[0].in_channels
        if f_vae_res.shape[1] != expected_ch:
            if f_vae_res.shape[1] > expected_ch:
                f_vae_res = f_vae_res[:, :expected_ch]
            else:
                repeats = (expected_ch // f_vae_res.shape[1]) + 1
                f_vae_res = f_vae_res.repeat(1, repeats, 1, 1)[:, :expected_ch]
        f_vae = self.vae_proj(f_vae_res)

        # Upsample Dinomaly2 CAR Anomaly Map A(n) to current resolution
        if car_anomaly_map.ndim == 2:
            side = int(car_anomaly_map.shape[1] ** 0.5)
            car_anomaly_map = car_anomaly_map.view(-1, 1, side, side)
        a_up = F.interpolate(car_anomaly_map, size=target_size, mode="bilinear", align_corners=False)

        # Semantic gate: 1 + sigmoid(Psi(A))
        semantic_gate = 1.0 + torch.sigmoid(self.anomaly_gate_proj(a_up))

        # Snapping: normal edges in f_vae are suppressed where semantic_gate is baseline (A ~ 0)
        fused = f_up * f_vae * semantic_gate
        out = self.refine_conv(fused) + self.residual_conv(fused)
        return out


class CARGuidedRMP(nn.Module):
    """
    Full CAR-Guided Refined Mask Prediction (CAR-MRM) Branch:
      Stage 1: U-Net up-2 (32x32x1280) + up-3 (64x64x640) -> 1x1 Conv -> Concat -> 4-layer Transformer -> F_hat (64x64x192)
      Stage 2: 3x Cascaded MRM blocks fusing VAE residuals (128 -> 256 -> 512) gated by CAR anomaly map
      Stage 3: Multi-resolution outputs: Coarse (64x64x2) and Refined (512x512x2)
    """
    def __init__(
        self,
        up2_ch: int = 1280,
        up3_ch: int = 640,
        comp_ch: int = 96,
        trans_dim: int = 192,
        trans_layers: int = 4,
        trans_heads: int = 6,
        vae_channels: Optional[List[int]] = None,
        mrm_channels: Optional[List[int]] = None,
        num_classes: int = 2,
    ):
        super().__init__()
        self.vae_channels = vae_channels or [512, 256, 128]
        self.mrm_channels = mrm_channels or [128, 64, 32]
        self.trans_dim = trans_dim

        # 1x1 Convolutions for compressing U-Net decoder features
        self.conv_up2 = nn.Sequential(
            nn.Conv2d(up2_ch, comp_ch, kernel_size=1, bias=False),
            nn.GroupNorm(8, comp_ch),
            nn.SiLU(),
        )
        self.conv_up3 = nn.Sequential(
            nn.Conv2d(up3_ch, comp_ch, kernel_size=1, bias=False),
            nn.GroupNorm(8, comp_ch),
            nn.SiLU(),
        )

        # 4-layer Transformer Encoder at 64x64 resolution (4096 tokens)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=trans_dim,
            nhead=trans_heads,
            dim_feedforward=trans_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=trans_layers)

        # Coarse Mask Prediction Head at 64x64 resolution
        self.coarse_mask_head = nn.Conv2d(trans_dim, num_classes, kernel_size=3, padding=1)

        # Cascaded MRM Blocks:
        # mrm1: 64x64 -> 128x128 (fuses vae_channels[0])
        self.mrm1 = MaskRefinementModuleBlock(trans_dim, self.vae_channels[0], self.mrm_channels[0])
        # mrm2: 128x128 -> 256x256 (fuses vae_channels[1])
        self.mrm2 = MaskRefinementModuleBlock(self.mrm_channels[0], self.vae_channels[1], self.mrm_channels[1])
        # mrm3: 256x256 -> 512x512 (fuses vae_channels[2])
        self.mrm3 = MaskRefinementModuleBlock(self.mrm_channels[1], self.vae_channels[2], self.mrm_channels[2])

        # Final Refined Mask Head at 512x512 resolution
        self.final_mask_head = nn.Sequential(
            nn.Conv2d(self.mrm_channels[2], self.mrm_channels[2], kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, self.mrm_channels[2]),
            nn.SiLU(),
            nn.Conv2d(self.mrm_channels[2], num_classes, kernel_size=3, padding=1),
        )

    def forward(
        self,
        f_up2: torch.Tensor,
        f_up3: torch.Tensor,
        vae_residuals: List[torch.Tensor],
        car_anomaly_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            f_up2: (B, 1280, 32, 32) deep semantic features from U-Net up-2 block
            f_up3: (B, 640, 64, 64) mid-level structural features from U-Net up-3 block
            vae_residuals: [F_1^res (128x128), F_2^res (256x256), F_3^res (512x512)]
            car_anomaly_map: (B, 1024) or (B, 1, 32, 32) Dinomaly2 CAR cosine anomaly map
        Returns:
            coarse_logits: (B, 2, 64, 64)
            refined_logits: (B, 2, 512, 512)
        """
        B = f_up2.shape[0]

        # Stage 1: Coarse Fusion at 64x64
        p1 = F.interpolate(self.conv_up2(f_up2), size=(64, 64), mode="bilinear", align_corners=False)
        p2 = self.conv_up3(f_up3)
        f_cat = torch.cat([p1, p2], dim=1)  # (B, 192, 64, 64)

        # Flatten spatial dimensions for Transformer: (B, 4096, 192)
        tokens = f_cat.flatten(2).transpose(1, 2)
        f_hat = self.transformer(tokens).transpose(1, 2).view(B, self.trans_dim, 64, 64)

        coarse_logits = self.coarse_mask_head(f_hat)  # (B, 2, 64, 64)

        # Stage 2: Cascaded Mask Refinement with VAE Residuals & CAR Anomaly Gating
        if car_anomaly_map.ndim == 2:
            side = int(car_anomaly_map.shape[1] ** 0.5)
            car_map_2d = car_anomaly_map.view(B, 1, side, side)
        else:
            car_map_2d = car_anomaly_map

        x = self.mrm1(f_hat, vae_residuals[0], car_map_2d)      # -> (B, 128, 128, 128)
        x = self.mrm2(x, vae_residuals[1], car_map_2d)          # -> (B, 64, 256, 256)
        x = self.mrm3(x, vae_residuals[2], car_map_2d)          # -> (B, 32, 512, 512)

        refined_logits = self.final_mask_head(x)                # -> (B, 2, 512, 512)
        return coarse_logits, refined_logits
