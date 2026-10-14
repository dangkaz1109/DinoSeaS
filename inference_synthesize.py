"""
DinoSeaS: Inference & Defect Synthesis Script
Generates category-consistent, pixel-annotated industrial defects
given a clean reference image of any category, with controllable defect taxonomy:
  - Micro-Surface Defects (Textural / Scratch / Stain): lambda_1 = 2.0, lambda_2 = 0.2
  - Structural / Logical Defects (Missing / Bent / Displaced): lambda_1 = 0.5, lambda_2 = 2.5
"""

import argparse
import logging
import os
import sys
import yaml

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.dinoseas_model import DinoSeaSModel
from train_phase1_evaluator import load_dinov2_backbone
from train_phase2_dinoseas import load_sd_backbones, SimpleNoiseScheduler


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DinoSeaS_Inference")


def parse_args():
    parser = argparse.ArgumentParser(description="DinoSeaS Defect Synthesis & Steering")
    parser.add_argument("--checkpoint", type=str, required=False, default=None, help="Path to DinoSeaS checkpoint")
    parser.add_argument("--config", type=str, default="config/dinoseas_multiclass.yaml", help="Path to YAML config")
    parser.add_argument("--category", type=str, default=None, help="MVTec category name (e.g. bottle, cable)")
    parser.add_argument("--ref_image", type=str, required=False, default=None, help="Path to clean normal input image")
    parser.add_argument("--output_dir", type=str, default="outputs/synthesized", help="Directory to save generated samples")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of defect variations to generate")
    parser.add_argument(
        "--defect_type",
        type=str,
        default="micro_surface",
        choices=["micro_surface", "surface", "structural", "balanced"],
        help="Type of defect steering taxonomy",
    )
    parser.add_argument("--lambda_1", type=float, default=None, help="Override Group 1 Geometric weight")
    parser.add_argument("--lambda_2", type=float, default=None, help="Override Group 2 Semantic weight")
    parser.add_argument("--timestep", type=int, default=400, help="Diffusion timestep for defect synthesis (100-600)")
    parser.add_argument(
        "--no_ref",
        action="store_true",
        help="Reference-Free Mode: synthesize defects directly from precomputed category prototypes without any input reference image",
    )
    parser.add_argument(
        "--prototype_path",
        type=str,
        default="weights/category_prototypes.pth",
        help="Path to precomputed category prototypes dictionary",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def synthesize_defects():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Steering weights according to Defect Taxonomy:
    if args.lambda_1 is not None and args.lambda_2 is not None:
        lambda_1, lambda_2 = args.lambda_1, args.lambda_2
    elif args.defect_type in ["micro_surface", "surface"]:
        lambda_1, lambda_2 = 2.0, 0.2  # Hairline scratches, cracks, surface stains
    elif args.defect_type == "structural":
        lambda_1, lambda_2 = 0.5, 2.5  # Missing parts, bent pins, structural deformations
    else:
        lambda_1, lambda_2 = 1.0, 1.0

    logger.info(f"Taxonomy Steering: {args.defect_type} (lambda_1={lambda_1}, lambda_2={lambda_2})")

    # Load Backbones
    vit_encoder = load_dinov2_backbone(config, str(device))
    unet, vae = load_sd_backbones(config, str(device))

    model = DinoSeaSModel(config=config, unet=unet, vae=vae, dinov2=vit_encoder).to(device)
    model.eval()

    if args.checkpoint and os.path.exists(args.checkpoint):
        logger.info(f"Loading checkpoint from {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)

    model.loss_engine.set_steering_weights(lambda_1, lambda_2)

    # Prepare Substrate & Class Anchor
    img_size = config["experiment"].get("image_size", 512)
    img_transform = T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if args.no_ref:
        cat = args.category or "bottle"
        logger.info(f"[Reference-Free Mode] Generating defects for '{cat}' using prototype anchor (no input image required)...")
        if not os.path.exists(args.prototype_path):
            raise FileNotFoundError(f"Prototypes file {args.prototype_path} not found. Please precompute category prototypes.")
        prototypes = torch.load(args.prototype_path, map_location=device)
        if cat not in prototypes:
            raise KeyError(f"Category '{cat}' not found in prototypes. Available: {list(prototypes.keys())}")
        
        proto_data = prototypes[cat]
        z_proto_base = proto_data["z_proto"].to(device)
        f_cls_anchor = proto_data["cls_proto"].to(device)
        
        # Decode clean prototype image for DINOv2 baseline & visualization
        with torch.no_grad():
            x_ob, _ = model.vae_wrapper.decode_with_residuals(z_proto_base / 0.18215)
            x_ob_448 = F.interpolate(x_ob, size=(448, 448), mode="bilinear", align_corners=False)
            vit_ref_out = model.extract_vit_features(x_ob_448)
            dino_ref = model.evaluator(vit_ref_out)
    else:
        # Reference-Guided Mode (Original Method - 100% Preserved)
        if args.ref_image and os.path.exists(args.ref_image):
            pil_img = Image.open(args.ref_image).convert("RGB")
            x_ob = img_transform(pil_img).unsqueeze(0).to(device)
        else:
            # Search for any valid normal image from dataset
            data_path = config["experiment"].get("data_path", "/home/lablee/AD/mvtec_data")
            candidate_images = []
            search_cats = [args.category] if args.category else ["bottle", "cable", "carpet", "transistor"]
            for cat in search_cats:
                good_p = os.path.join(data_path, cat, "train", "good")
                if os.path.isdir(good_p):
                    for f in sorted(os.listdir(good_p)):
                        if f.endswith(".png"):
                            candidate_images.append(os.path.join(good_p, f))
                            break
            if candidate_images:
                ref_path = candidate_images[0]
                logger.info(f"Using reference image: {ref_path}")
                pil_img = Image.open(ref_path).convert("RGB")
                x_ob = img_transform(pil_img).unsqueeze(0).to(device)
            else:
                logger.info("Generating random normal reference substrate for synthesis test.")
                x_ob = torch.randn(1, 3, img_size, img_size, device=device)

        # 1. Extract CAR Global Class Anchor f_cls from clean reference x_ob
        with torch.no_grad():
            x_ob_448 = F.interpolate(x_ob, size=(448, 448), mode="bilinear", align_corners=False)
            vit_ref_out = model.extract_vit_features(x_ob_448)
            dino_ref = model.evaluator(vit_ref_out)
            f_cls_anchor = dino_ref["f_cls_agg"]  # (1, 1, 768)

    seq_len = 7
    embed_dim = config["dinomaly2"]["embed_dim"]

    logger.info(f"Generating {args.num_samples} synthesized defect samples...")
    for s_idx in range(args.num_samples):
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16, enabled=device.type == "cuda"):
            # Prompt initialization
            base_e_df = torch.randn(1, seq_len, embed_dim, device=device) * 0.02
            base_e_ob = torch.randn(1, seq_len, embed_dim, device=device) * 0.02

            # Inject CAR anchor & Noisy Bottleneck perturbation
            e_df_cond, _ = model.inject_prompt_conditioning(base_e_df, base_e_ob, f_cls_anchor)

            # 1. Obtain clean latent z_0
            if args.no_ref:
                # Add subtle sample-level stochasticity to prototype latent for diversity
                z_0 = z_proto_base + torch.randn_like(z_proto_base) * 0.03
            else:
                z_ob = model.vae_wrapper.encode(x_ob)
                z_0 = (z_ob.latent_dist.sample() if hasattr(z_ob, "latent_dist") else z_ob) * 0.18215

            # 2. Add controlled diffusion noise to z_0 using noise scheduler
            scheduler = SimpleNoiseScheduler()
            timestep = torch.tensor([args.timestep], device=device).long()
            eps = torch.randn_like(z_0)
            z_t = scheduler.add_noise(z_0, eps, timestep)

            # 3. Denoise step conditioned on defect prompt
            eps_pred_df, unet_feats_df = model.forward_unet(
                z_t, timestep, encoder_hidden_states=e_df_cond, is_defect_stream=True
            )

            # 4. Tweedie One-Step Reconstruction
            alphas = scheduler.alphas_cumprod.to(device)
            z0_hat_df = model.tweedie_denoise_latent(z_t, eps_pred_df, timestep, alphas)

            # 5. VAE decode with multi-scale skip residuals
            x_hat_df, vae_res_df = model.vae_wrapper.decode_with_residuals(z0_hat_df / 0.18215)

            # Dinomaly2 evaluation on synthesized defect
            x_hat_df_448 = F.interpolate(x_hat_df, size=(448, 448), mode="bilinear", align_corners=False)
            vit_syn_out = model.extract_vit_features(x_hat_df_448)
            dino_syn = model.evaluator(vit_syn_out)

            # Refined Mask Prediction with CAR Anomaly Gate
            _, refined_logits = model.rmp_branch(
                unet_feats_df["up2"], unet_feats_df["up3"], vae_res_df, dino_syn["anomaly_map"]
            )
            mask_prob = F.softmax(refined_logits, dim=1)[:, 1, :, :]  # (1, 512, 512)

            # Normalize output image to [0, 255]
            img_out = ((x_hat_df[0].clamp(-1.0, 1.0) + 1.0) * 127.5).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            mask_out = (mask_prob[0].clamp(0.0, 1.0) * 255.0).cpu().numpy().astype(np.uint8)

            # Create Overlay (adaptive threshold to clearly reveal synthesized defect locus)
            overlay = img_out.copy()
            thresh = 128 if mask_out.max() >= 128 else max(30, int(mask_out.max() * 0.70))
            red_mask = (mask_out >= thresh)
            overlay[red_mask, 0] = np.clip(overlay[red_mask, 0] * 0.4 + 150, 0, 255)
            overlay[red_mask, 1] = np.clip(overlay[red_mask, 1] * 0.4, 0, 255)
            overlay[red_mask, 2] = np.clip(overlay[red_mask, 2] * 0.4, 0, 255)

            # Save Files
            prefix = f"sample_{s_idx:03d}_{args.defect_type}"
            img_p = os.path.join(args.output_dir, f"{prefix}_defect.png")
            mask_p = os.path.join(args.output_dir, f"{prefix}_mask.png")
            overlay_p = os.path.join(args.output_dir, f"{prefix}_overlay.png")

            Image.fromarray(img_out).save(img_p)
            Image.fromarray(mask_out).save(mask_p)
            Image.fromarray(overlay).save(overlay_p)

            logger.info(f"Saved: {img_p}, {mask_p}, {overlay_p}")

    logger.info("Defect synthesis completed successfully!")


if __name__ == "__main__":
    synthesize_defects()
