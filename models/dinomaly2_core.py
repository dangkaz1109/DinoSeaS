"""
DinoSeaS: Module A - Dinomaly2 Core
Context-Aware Recentering (CAR), Noisy Bottleneck, Unfocused Linear Attention,
and Loose Group Evaluator (S_1 Geometric vs S_2 Semantic).
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def stop_gradient_scale(x: torch.Tensor, scale: float = 0.1) -> torch.Tensor:
    """
    Implements sg_{scale}(x): Forward pass returns x unchanged;
    Backward pass scales the gradient by `scale` (e.g. 0.10 = 10% gradient).
    Prevents outlier/hard normal patches from destabilizing model updates.
    """
    return scale * x + (1.0 - scale) * x.detach()


class ContextAwareRecentering(nn.Module):
    r"""
    Context-Aware Recentering (CAR):
    Subtracts the global class token f_cls^i from spatial patch tokens f_patch^i:
        f~_patch^i = f_patch^i - f_cls^i \otimes 1_N
    This maps diverse object categories into a zero-centered, class-agnostic
    structural coordinate system, eliminating multi-class ambiguity and illumination shifts.
    """
    def __init__(self, normalize: bool = True, eps: float = 1e-8):
        super().__init__()
        self.normalize = normalize
        self.eps = eps

    def forward(self, f_patch: torch.Tensor, f_cls: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_patch: (B, N, D) spatial patch representations
            f_cls:   (B, 1, D) global class token
        Returns:
            f_tilde: (B, N, D) recentered zero-mean structural features
        """
        if f_patch.ndim != 3 or f_cls.ndim != 3:
            raise ValueError(
                f"Expected 3D tensors: f_patch {f_patch.shape}, f_cls {f_cls.shape}"
            )
        f_tilde = f_patch - f_cls.expand_as(f_patch)
        if self.normalize:
            f_tilde = F.layer_norm(f_tilde, (f_tilde.shape[-1],), eps=self.eps)
        return f_tilde


class NoisyBottleneckMLP(nn.Module):
    """
    3-Layer Information Bottleneck MLP with Dropout:
    During defect synthesis, B_df acts as a stochastic subspace sub-sampler
    m_l ~ Bernoulli(1 - p), preventing identity copying and few-shot defect memorization.
    """
    def __init__(
        self,
        dim: int = 768,
        num_layers: int = 3,
        dropout_p: float = 0.3,
        hidden_dim: Optional[int] = None,
        activation: str = "gelu",
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        h_dim = hidden_dim or dim

        act_cls = nn.GELU if activation.lower() == "gelu" else nn.ReLU

        layers: List[nn.Module] = []
        for i in range(num_layers):
            in_d = dim if i == 0 else h_dim
            out_d = dim if i == num_layers - 1 else h_dim
            layers.append(nn.Linear(in_d, out_d))
            if i < num_layers - 1:
                layers.append(act_cls())
            layers.append(nn.Dropout(p=dropout_p))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02, a=-0.04, b=0.04)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UnfocusedLinearAttention(nn.Module):
    """
    Unfocused Linear Attention (LA):
        LA(Q, K, V) = phi(Q) (phi(K)^T V) / Normalizer
    Using non-negative activation phi(x) = ELU(x) + 1.0.
    With maximum rank d << N, it acts as a spatial low-pass filter, mathematically
    preventing normal substrate tokens <ob> from concentrating energy into sharp local defects.
    """
    def __init__(self, dim: int = 768, num_heads: int = 8, eps: float = 1e-6):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.qkv.weight, std=0.02, a=-0.04, b=0.04)
        nn.init.trunc_normal_(self.proj.weight, std=0.02, a=-0.04, b=0.04)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0.0)

    @staticmethod
    def kernel_phi(x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        # Disable autocast and cast to float32 to prevent FP16 denominator overflow in linear attention
        with torch.cuda.amp.autocast(enabled=False):
            x_f = x.float()
            B, N, D = x_f.shape
            qkv = self.qkv(x_f).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, d)

            q_phi = self.kernel_phi(q)
            k_phi = self.kernel_phi(k)

            # O(N * d^2) linear attention computation
            # kv: (B, H, d, d)
            kv = torch.einsum("bhnd,bhne->bhde", k_phi, v)

            # Normalizer z: (B, H, N)
            k_sum = k_phi.sum(dim=2)  # (B, H, d)
            denom = torch.einsum("bhnd,bhd->bhn", q_phi, k_sum) + self.eps
            z = 1.0 / denom

            # Output projection
            out = torch.einsum("bhnd,bhde,bhn->bhne", q_phi, kv, z)
            out = out.transpose(1, 2).reshape(B, N, D)
            out = self.proj(out)
        return out.to(orig_dtype)


class LinearTransformerBlock(nn.Module):
    """
    Linear Attention Transformer Block with pre-LayerNorm and MLP.
    """
    def __init__(self, dim: int = 768, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = UnfocusedLinearAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Dinomaly2LooseEvaluator(nn.Module):
    """
    Dinomaly2 Loose Group Reconstruction Module.
    Separates ViT representations into:
      - Group 1 (Geometric/Shallow): Layers S_1 = {3, 4, 5, 6} (0-indexed [2, 3, 4, 5])
      - Group 2 (Semantic/Deep):     Layers S_2 = {7, 8, 9, 10} (0-indexed [6, 7, 8, 9])
    Computes group-wise cosine distances d_cos(g_k, g^_k) and recentered anomaly maps.
    """
    def __init__(
        self,
        embed_dim: int = 768,
        group1_idx: List[int] = [2, 3, 4, 5],
        group2_idx: List[int] = [6, 7, 8, 9],
        bottleneck_layers: int = 3,
        dropout_p: float = 0.3,
        decoder_depth: int = 4,
        num_heads: int = 8,
    ):
        super().__init__()
        self.group1_idx = group1_idx
        self.group2_idx = group2_idx
        self.embed_dim = embed_dim
        self.car = ContextAwareRecentering(normalize=True)
        self.bottleneck = NoisyBottleneckMLP(
            dim=embed_dim, num_layers=bottleneck_layers, dropout_p=dropout_p
        )

        # Decoders for Group 1 and Group 2 using Unfocused Linear Attention
        self.decoder_g1 = nn.Sequential(
            *[LinearTransformerBlock(embed_dim, num_heads) for _ in range(decoder_depth)]
        )
        self.decoder_g2 = nn.Sequential(
            *[LinearTransformerBlock(embed_dim, num_heads) for _ in range(decoder_depth)]
        )

    def forward(
        self, vit_layer_outputs: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            vit_layer_outputs: List of tuples (f_patch^i, f_cls^i) from frozen ViT encoder.
                               f_patch^i shape: (B, N, D), f_cls^i shape: (B, 1, D)
        Returns:
            Dictionary containing:
              - g1, g2: Ground-truth target group features (B, N, D)
              - g1_hat, g2_hat: Reconstructed group features (B, N, D)
              - d_cos_g1, d_cos_g2: Spatial cosine distance per patch (B, N)
              - anomaly_map: Combined anomaly score per patch (B, N)
              - f_cls_agg: Aggregated class token anchor (B, 1, D)
        """
        if not vit_layer_outputs:
            raise ValueError("vit_layer_outputs must not be empty")

        recentered_patches = []
        cls_tokens = []

        for f_patch, f_cls in vit_layer_outputs:
            f_tilde = self.car(f_patch, f_cls)
            recentered_patches.append(f_tilde)
            cls_tokens.append(f_cls)

        # Target Group Aggregations
        g1 = torch.stack([recentered_patches[i] for i in self.group1_idx], dim=0).sum(dim=0)
        g2 = torch.stack([recentered_patches[i] for i in self.group2_idx], dim=0).sum(dim=0)

        # Bottleneck Input: fusion of all target layers
        all_target_idx = self.group1_idx + self.group2_idx
        z_0 = torch.stack([recentered_patches[i] for i in all_target_idx], dim=0).sum(dim=0)
        z = self.bottleneck(z_0)

        # Loose Group Reconstructions
        g1_hat = self.decoder_g1(z)
        g2_hat = self.decoder_g2(g1_hat + z)

        # Spatial Cosine Distance per patch n: d_cos(g_k(n), g^_k(n)) = 1 - cos_sim (FP32 precision)
        d_cos_g1 = 1.0 - F.cosine_similarity(g1.float(), g1_hat.float(), dim=-1, eps=1e-6)  # (B, N)
        d_cos_g2 = 1.0 - F.cosine_similarity(g2.float(), g2_hat.float(), dim=-1, eps=1e-6)  # (B, N)
        anomaly_map = 0.5 * (d_cos_g1 + d_cos_g2)                                           # (B, N)

        # Aggregated class anchor for continuous prompting
        f_cls_agg = torch.stack(cls_tokens, dim=0).mean(dim=0)    # (B, 1, D)

        return {
            "g1": g1,
            "g2": g2,
            "g1_hat": g1_hat,
            "g2_hat": g2_hat,
            "d_cos_g1": d_cos_g1,
            "d_cos_g2": d_cos_g2,
            "anomaly_map": anomaly_map,
            "f_cls_agg": f_cls_agg,
        }
