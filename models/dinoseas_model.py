"""
DinoSeaS: Master Unified Architecture
Combines SeaS (Diffusion U-Net + UA Prompting + CAR-MRM) and
Dinomaly2 (Frozen DINOv2 + CAR + Noisy Bottleneck + Loose Group Evaluator)
into a unified multi-class industrial anomaly synthesis and detection framework.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dinomaly2_core import (
    ContextAwareRecentering,
    Dinomaly2LooseEvaluator,
    NoisyBottleneckMLP,
    UnfocusedLinearAttention,
)
from .hybrid_attention import HybridCrossAttnProcessor, HybridDualCrossAttention
from .car_rmp import CARGuidedRMP
from .losses import DinoSeaSLossEngine
from .vae_hook import VAEDecoderWithResiduals


class ContinuousObjectProjection(nn.Module):
    r"""
    Pillar I: Continuous Object Projection \psi_{ob}(f_cls).
    Projects aggregated DINOv2 class token f_cls directly into the text embedding space
    to serve as the dynamic, instance-level continuous object token e_{ob}.
    Carries illumination, pose, and category identity, preventing cross-class texture bleed.
    """
    def __init__(self, in_dim: int = 768, out_dim: int = 768, hidden_dim: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-6),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02, a=-0.04, b=0.04)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, f_cls: torch.Tensor) -> torch.Tensor:
        # f_cls: (B, 1, in_dim) -> (B, 1, out_dim)
        return self.net(f_cls)


class DinoSeaSModel(nn.Module):
    """
    Unified DinoSeaS Model Architecture.
    Handles:
      - Continuous CAR-Conditioned Prompting (Pillar I)
      - Hybrid Dual-Attention Cross Attention (Pillar II)
      - Defect Token Noisy Bottleneck B_df (Pillar III)
      - Dinomaly2 Loose Group Evaluator (Pillar IV)
      - CAR-Guided Refined Mask Prediction (Pillar V)
      - One-Step Tweedie Latent Decoding
    """
    def __init__(
        self,
        config: dict,
        unet: nn.Module,
        vae: nn.Module,
        dinov2: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.cfg = config
        self.unet = unet
        self.vae_wrapper = VAEDecoderWithResiduals(vae)
        self.dinov2 = dinov2

        embed_dim = config["dinomaly2"]["embed_dim"]
        g1_idx = config["dinomaly2"]["loose_groups"]["group_1_shallow"]
        g2_idx = config["dinomaly2"]["loose_groups"]["group_2_deep"]
        bn_cfg = config["dinomaly2"]["noisy_bottleneck"]
        dec_cfg = config["dinomaly2"]["decoder"]

        # 1. Dinomaly2 Loose Group Evaluator
        self.evaluator = Dinomaly2LooseEvaluator(
            embed_dim=embed_dim,
            group1_idx=g1_idx,
            group2_idx=g2_idx,
            bottleneck_layers=bn_cfg["num_layers"],
            dropout_p=bn_cfg["dropout_prob"],
            decoder_depth=dec_cfg["num_layers_per_group"],
            num_heads=dec_cfg["num_heads"],
        )

        # 2. Continuous Object Projection \psi_{ob}
        self.ob_proj = ContinuousObjectProjection(
            in_dim=embed_dim, out_dim=embed_dim, hidden_dim=embed_dim
        )

        # 3. Defect Token Noisy Bottleneck B_df
        df_dropout = config["seas"]["prompt_design"]["df_bottleneck_dropout"]
        self.df_bottleneck = NoisyBottleneckMLP(
            dim=embed_dim, num_layers=3, dropout_p=df_dropout
        )

        # 4. CAR-Guided RMP Branch
        rmp_cfg = config["seas"]["rmp_branch"]
        self.rmp_branch = CARGuidedRMP(
            up2_ch=rmp_cfg["unet_up2_channels"],
            up3_ch=rmp_cfg["unet_up3_channels"],
            comp_ch=rmp_cfg["compressed_channels"],
            trans_dim=rmp_cfg["transformer_dim"],
            trans_layers=rmp_cfg["transformer_layers"],
            trans_heads=rmp_cfg["transformer_heads"],
            vae_channels=rmp_cfg["vae_res_channels"],
            mrm_channels=rmp_cfg["mrm_hidden_channels"],
            num_classes=rmp_cfg["num_mask_classes"],
        )

        # 5. Hybrid Dual Attention Layers collection
        self.hybrid_attn_layers: List[HybridDualCrossAttention] = []
        self._configure_hybrid_attention()

        # 6. Loss Engine
        self.loss_engine = DinoSeaSLossEngine(config)

    def _configure_hybrid_attention(self):
        """
        Inspects or patches U-Net cross-attention layers with HybridDualCrossAttention.
        """
        self.hybrid_attn_layers.clear()
        # Search for cross-attention blocks in UNet
        for name, module in self.unet.named_modules():
            if isinstance(module, HybridDualCrossAttention):
                self.hybrid_attn_layers.append(module)

        # If UNet doesn't already have HybridDualCrossAttention, create reference instances
        # for recorded resolutions [16, 32, 64]
        if not self.hybrid_attn_layers:
            resolutions = self.cfg["seas"]["hybrid_attention"]["record_resolutions"]
            for r in resolutions:
                dim = 320 if r == 64 else (640 if r == 32 else 1280)
                attn = HybridDualCrossAttention(
                    query_dim=dim,
                    context_dim=self.cfg["dinomaly2"]["embed_dim"],
                    heads=self.cfg["seas"]["hybrid_attention"]["heads"],
                    dim_head=self.cfg["seas"]["hybrid_attention"]["dim_head"],
                    ob_token_idx=1,
                    df_token_indices=[3, 4, 5, 6],
                )
                self.hybrid_attn_layers.append(attn)
            # Register as submodule list
            self.registered_hybrid_attns = nn.ModuleList(self.hybrid_attn_layers)

    def extract_vit_features(
        self, img_448: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Extracts multi-layer (f_patch^i, f_cls^i) representations from frozen DINOv2.
        Input img_448: (B, 3, 448, 448).
        With patch_size=14, grid is 32 x 32 = 1024 patches.
        """
        if self.dinov2 is not None:
            if hasattr(self.dinov2, "get_intermediate_layers"):
                # DINOv2 official API
                layers = self.dinov2.get_intermediate_layers(
                    img_448, n=12, return_class_token=True
                )
                return layers
            elif hasattr(self.dinov2, "blocks"):
                # Manual ViT block loop
                x = self.dinov2.prepare_tokens(img_448)
                outputs = []
                for blk in self.dinov2.blocks:
                    x = blk(x)
                    f_cls = x[:, :1, :]
                    f_patch = x[:, 1:, :]
                    outputs.append((f_patch, f_cls))
                return outputs

        # Standalone / Fallback ViT simulation (if external ViT backbone not provided in unit tests)
        B = img_448.shape[0]
        D = self.cfg["dinomaly2"]["embed_dim"]
        N = 1024
        outputs = []
        for _ in range(12):
            f_patch = torch.randn(B, N, D, device=img_448.device)
            f_cls = torch.randn(B, 1, D, device=img_448.device)
            outputs.append((f_patch, f_cls))
        return outputs

    def inject_prompt_conditioning(
        self,
        base_e_df: torch.Tensor,
        base_e_ob: torch.Tensor,
        f_cls_anchor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Injects continuous CAR class anchor f_cls into <ob> slot (index 1)
        and applies Noisy Bottleneck B_df to <df> slots (indices 3, 4, 5, 6).
        """
        e_ob_cond = base_e_ob.clone()
        e_df_cond = base_e_df.clone()

        # Dynamic continuous object token injection
        ob_anchor = self.ob_proj(f_cls_anchor)  # (B, 1, 768)
        e_ob_cond[:, 1:2, :] = ob_anchor
        e_df_cond[:, 1:2, :] = ob_anchor

        # Noisy Bottleneck stochastic perturbation on defect tokens
        # Prevents identity copying of few-shot defects
        df_tokens = base_e_df[:, 3:7, :]  # (B, 4, 768)
        perturbed_df_tokens = self.df_bottleneck(df_tokens)
        e_df_cond[:, 3:7, :] = perturbed_df_tokens

        return e_df_cond, e_ob_cond

    def tweedie_denoise_latent(
        self,
        z_t: torch.Tensor,
        eps_pred: torch.Tensor,
        t: torch.Tensor,
        alphas_cumprod: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        One-Step Tweedie Denoising formula:
          z~_0 = (z_t - \beta_t * \epsilon_\theta(z_t, t, e)) / \alpha_t
        where \alpha_t = sqrt(\bar{\alpha}_t), \beta_t = sqrt(1 - \bar{\alpha}_t).
        """
        alpha_bar = alphas_cumprod[t].view(-1, 1, 1, 1)
        alpha_t = alpha_bar.sqrt()
        beta_t = (1.0 - alpha_bar).sqrt()
        z0_hat = (z_t - beta_t * eps_pred) / (alpha_t + 1e-8)
        return z0_hat

    def forward_unet(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        is_defect_stream: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through U-Net, returning noise prediction and decoder features.
        """
        if hasattr(self.unet, "forward_with_features"):
            return self.unet.forward_with_features(
                z_t, t, encoder_hidden_states, is_defect_stream=is_defect_stream
            )

        # Standard U-Net call
        out = self.unet(z_t, t, encoder_hidden_states=encoder_hidden_states)
        eps_pred = out.sample if hasattr(out, "sample") else (out[0] if isinstance(out, tuple) else out)

        # Extract or synthesize up-2 and up-3 features if hooked
        B = z_t.shape[0]
        up2 = getattr(self.unet, "last_up2_feature", None)
        up3 = getattr(self.unet, "last_up3_feature", None)

        if up2 is None or up3 is None:
            # Fallback projection from z_t for decoupled module unit-testing
            up2 = F.interpolate(z_t, size=(32, 32), mode="bilinear", align_corners=False)
            up2 = up2.repeat(1, 1280 // 4, 1, 1)
            up3 = F.interpolate(z_t, size=(64, 64), mode="bilinear", align_corners=False)
            up3 = up3.repeat(1, 640 // 4, 1, 1)

        unet_feats = {"up2": up2, "up3": up3}
        return eps_pred, unet_feats

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        noise_scheduler,
    ) -> Dict[str, torch.Tensor]:
        """
        Single unified end-to-end training iteration across multi-class normal
        and defective pairs.
        """
        x_df = batch["x_df"]          # (B, 3, 512, 512)
        x_ob = batch["x_ob"]          # (B, 3, 512, 512)
        gt_mask = batch["mask_df"]    # (B, 512, 512)
        base_e_df = batch["e_df"]     # (B, T, 768)
        base_e_ob = batch["e_ob"]     # (B, T, 768)

        # Step 1: Extract CAR Global Class Anchor from clean reference x_ob
        with torch.no_grad():
            x_ob_448 = F.interpolate(x_ob, size=(448, 448), mode="bilinear", align_corners=False)
            vit_ref_out = self.extract_vit_features(x_ob_448)
            dino_ref = self.evaluator(vit_ref_out)
            f_cls_anchor = dino_ref["f_cls_agg"]  # (B, 1, 768)

        # Step 2: Inject CAR continuous anchor & perturb defect tokens
        e_df_cond, e_ob_cond = self.inject_prompt_conditioning(
            base_e_df, base_e_ob, f_cls_anchor
        )

        # Step 3: Forward Diffusion on Latent space
        with torch.no_grad():
            z_df = self.vae_wrapper.encode(x_df)
            z_df = (z_df.latent_dist.sample() if hasattr(z_df, "latent_dist") else z_df) * 0.18215
            z_ob = self.vae_wrapper.encode(x_ob)
            z_ob = (z_ob.latent_dist.sample() if hasattr(z_ob, "latent_dist") else z_ob) * 0.18215

        max_t = self.cfg["training"]["tweedie_max_timestep"]
        t = torch.randint(0, max_t, (x_df.shape[0],), device=x_df.device).long()
        eps_df = torch.randn_like(z_df)
        eps_ob = torch.randn_like(z_ob)

        z_t_df = noise_scheduler.add_noise(z_df, eps_df, t)
        z_t_ob = noise_scheduler.add_noise(z_ob, eps_ob, t)

        # Step 4: U-Net predictions
        eps_pred_df, unet_feats_df = self.forward_unet(
            z_t_df, t, encoder_hidden_states=e_df_cond, is_defect_stream=True
        )
        eps_pred_ob, unet_feats_ob = self.forward_unet(
            z_t_ob, t, encoder_hidden_states=e_ob_cond, is_defect_stream=False
        )

        loss_sd_df = F.mse_loss(eps_pred_df, eps_df)
        loss_sd_ob = F.mse_loss(eps_pred_ob, eps_ob)
        loss_da = self.loss_engine.compute_da_loss(self.hybrid_attn_layers, gt_mask)

        # Step 5: One-Step Tweedie Latent Reconstruction & VAE Decoding
        alphas = noise_scheduler.alphas_cumprod.to(z_df.device)
        z0_hat_df = self.tweedie_denoise_latent(z_t_df, eps_pred_df, t, alphas)
        z0_hat_ob = self.tweedie_denoise_latent(z_t_ob, eps_pred_ob, t, alphas)

        with torch.no_grad():
            x_hat_df, vae_res_df = self.vae_wrapper.decode_with_residuals(z0_hat_df / 0.18215)
            x_hat_ob, vae_res_ob = self.vae_wrapper.decode_with_residuals(z0_hat_ob / 0.18215)
            x_hat_df_448 = F.interpolate(x_hat_df, size=(448, 448), mode="bilinear", align_corners=False)
            vit_syn_out = self.extract_vit_features(x_hat_df_448)

        dino_syn = self.evaluator(vit_syn_out)
        loss_gdcr = self.loss_engine.compute_gdcr_loss(dino_syn, gt_mask)

        # Step 7: CAR-Guided Refined Mask Prediction (RMP)
        coarse_df, refined_df = self.rmp_branch(
            unet_feats_df["up2"], unet_feats_df["up3"], vae_res_df, dino_syn["anomaly_map"].detach()
        )
        coarse_ob, refined_ob = self.rmp_branch(
            unet_feats_ob["up2"], unet_feats_ob["up3"], vae_res_ob, dino_ref["anomaly_map"].detach()
        )

        loss_mask = self.loss_engine.compute_mask_loss(
            coarse_df, refined_df, coarse_ob, refined_ob, gt_mask
        )

        # Step 8: Total Loss Formulation
        w = self.cfg["training"]["loss_weights"]
        total_loss = (
            w["lambda_sd_df"] * loss_sd_df
            + w["lambda_da"] * loss_da
            + w["lambda_sd_ob"] * loss_sd_ob
            + w["eta_1_mask_focal"] * loss_mask
            + w["eta_2_gdcr"] * loss_gdcr
        )

        return {
            "loss_total": total_loss,
            "loss_sd_df": loss_sd_df.detach(),
            "loss_sd_ob": loss_sd_ob.detach(),
            "loss_da": loss_da.detach(),
            "loss_mask": loss_mask.detach(),
            "loss_gdcr": loss_gdcr.detach(),
            "x_hat_df": x_hat_df.detach(),
            "refined_mask_logits": refined_df.detach(),
        }
