"""
DinoSeaS: Unified End-to-End Inference Pipeline
High-level interface for multi-class category-consistent defect synthesis,
pixel-sharp mask generation, and anomaly localization.
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from ..models.dinoseas_model import DinoSeaSModel


class DinoSeaSOutput:
    """Encapsulates synthesis and anomaly detection results."""
    def __init__(
        self,
        defect_image: Image.Image,
        refined_mask: Image.Image,
        overlay: Image.Image,
        anomaly_score: float,
        d_cos_g1: torch.Tensor,
        d_cos_g2: torch.Tensor,
    ):
        self.defect_image = defect_image
        self.refined_mask = refined_mask
        self.overlay = overlay
        self.anomaly_score = anomaly_score
        self.d_cos_g1 = d_cos_g1
        self.d_cos_g2 = d_cos_g2

    def save(self, output_dir: str, prefix: str = "dinoseas"):
        os.makedirs(output_dir, exist_ok=True)
        img_p = os.path.join(output_dir, f"{prefix}_defect.png")
        mask_p = os.path.join(output_dir, f"{prefix}_mask.png")
        overlay_p = os.path.join(output_dir, f"{prefix}_overlay.png")

        self.defect_image.save(img_p)
        self.refined_mask.save(mask_p)
        self.overlay.save(overlay_p)
        return img_p, mask_p, overlay_p


class DinoSeaSPipeline:
    """
    Unified Pipeline for DinoSeaS.
    Enables single-weight multi-class defect generation and anomaly evaluation.
    """
    def __init__(self, model: DinoSeaSModel, config: dict, device: str = "cuda"):
        self.model = model
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

        self.img_size = config["experiment"].get("image_size", 512)
        self.img_transform = T.Compose([
            T.Resize((self.img_size, self.img_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, config_path: str, device: str = "cuda"):
        import yaml
        from ..train_phase1_evaluator import load_dinov2_backbone
        from ..train_phase2_dinoseas import load_sd_backbones

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        dev = torch.device(device if torch.cuda.is_available() else "cpu")
        vit_encoder = load_dinov2_backbone(config, str(dev))
        unet, vae = load_sd_backbones(config, str(dev))

        model = DinoSeaSModel(config=config, unet=unet, vae=vae, dinov2=vit_encoder)
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=dev)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict, strict=False)

        return cls(model, config, device=str(dev))

    def synthesize(
        self,
        ref_image: Union[str, Image.Image, torch.Tensor],
        defect_type: str = "micro_surface",
        lambda_1: Optional[float] = None,
        lambda_2: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> DinoSeaSOutput:
        """
        Synthesizes a category-consistent defect on top of ref_image.
        """
        if seed is not None:
            torch.manual_seed(seed)

        # 1. Prepare reference image
        if isinstance(ref_image, str):
            pil_ref = Image.open(ref_image).convert("RGB")
            x_ob = self.img_transform(pil_ref).unsqueeze(0).to(self.device)
        elif isinstance(ref_image, Image.Image):
            x_ob = self.img_transform(ref_image).unsqueeze(0).to(self.device)
        else:
            x_ob = ref_image.to(self.device)
            if x_ob.ndim == 3:
                x_ob = x_ob.unsqueeze(0)

        # 2. Defect Taxonomy Steering
        if lambda_1 is not None and lambda_2 is not None:
            l1, l2 = lambda_1, lambda_2
        elif defect_type == "micro_surface":
            l1, l2 = 2.0, 0.2
        elif defect_type == "structural":
            l1, l2 = 0.5, 2.5
        else:
            l1, l2 = 1.0, 1.0
        self.model.loss_engine.set_steering_weights(l1, l2)

        with torch.no_grad():
            # Extract CAR Global Class Anchor f_cls
            x_ob_448 = F.interpolate(x_ob, size=(448, 448), mode="bilinear", align_corners=False)
            vit_ref_out = self.model.extract_vit_features(x_ob_448)
            dino_ref = self.model.evaluator(vit_ref_out)
            f_cls_anchor = dino_ref["f_cls_agg"]

            # Prompt Embeddings
            embed_dim = self.config["dinomaly2"]["embed_dim"]
            base_e_df = torch.randn(1, 7, embed_dim, device=self.device) * 0.02
            base_e_ob = torch.randn(1, 7, embed_dim, device=self.device) * 0.02
            e_df_cond, _ = self.model.inject_prompt_conditioning(base_e_df, base_e_ob, f_cls_anchor)

            # Sample Latent Noise
            z_t = torch.randn(1, 4, 64, 64, device=self.device)
            t = torch.tensor([500], device=self.device).long()

            eps_pred_df, unet_feats_df = self.model.forward_unet(
                z_t, t, encoder_hidden_states=e_df_cond, is_defect_stream=True
            )

            # Tweedie Reconstruction
            alpha_bar = torch.tensor(0.5, device=self.device)
            z0_hat_df = (z_t - (1.0 - alpha_bar).sqrt() * eps_pred_df) / (alpha_bar.sqrt() + 1e-8)

            # VAE decode with residuals
            x_hat_df, vae_res_df = self.model.vae_wrapper.decode_with_residuals(z0_hat_df / 0.18215)

            # Dinomaly2 evaluation on synthesized defect
            x_hat_df_448 = F.interpolate(x_hat_df, size=(448, 448), mode="bilinear", align_corners=False)
            vit_syn_out = self.model.extract_vit_features(x_hat_df_448)
            dino_syn = self.model.evaluator(vit_syn_out)

            # CAR-Guided Mask Refinement
            _, refined_logits = self.model.rmp_branch(
                unet_feats_df["up2"], unet_feats_df["up3"], vae_res_df, dino_syn["anomaly_map"]
            )
            mask_prob = F.softmax(refined_logits, dim=1)[:, 1, :, :]  # (1, 512, 512)

        # Convert to PIL images
        img_np = ((x_hat_df[0].clamp(-1.0, 1.0) + 1.0) * 127.5).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        mask_np = (mask_prob[0].clamp(0.0, 1.0) * 255.0).cpu().numpy().astype(np.uint8)

        overlay_np = img_np.copy()
        red_mask = (mask_np > 128)
        overlay_np[red_mask, 0] = np.clip(overlay_np[red_mask, 0] * 0.5 + 128, 0, 255)
        overlay_np[red_mask, 1] = np.clip(overlay_np[red_mask, 1] * 0.5, 0, 255)
        overlay_np[red_mask, 2] = np.clip(overlay_np[red_mask, 2] * 0.5, 0, 255)

        anomaly_score = dino_syn["anomaly_map"].max().item()

        return DinoSeaSOutput(
            defect_image=Image.fromarray(img_np),
            refined_mask=Image.fromarray(mask_np),
            overlay=Image.fromarray(overlay_np),
            anomaly_score=anomaly_score,
            d_cos_g1=dino_syn["d_cos_g1"],
            d_cos_g2=dino_syn["d_cos_g2"],
        )
