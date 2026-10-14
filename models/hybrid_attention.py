"""
DinoSeaS: Module B - Hybrid Dual-Attention U-Net Cross-Attention Layer
Splits Cross-Attention into two regimes:
  1. Unfocused Linear Attention on <ob> (low-pass global substrate, prevents local defect bleed)
  2. Focused Softmax Attention on <df> tokens (sharp localized defect generation)
Captures attention maps A_df and A_ob for Decoupled Anomaly Alignment Loss (L_DA).
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridDualCrossAttention(nn.Module):
    """
    Split-Regime Hybrid Cross-Attention Layer.
    Replaces standard U-Net Cross-Attention:
      Z_{cross}^l = phi(Q) (phi(K_{ob})^T V_{ob}) + Softmax(Q K_{df}^T / sqrt(d)) V_{df}
    """
    def __init__(
        self,
        query_dim: int,
        context_dim: int = 768,
        heads: int = 8,
        dim_head: int = 64,
        ob_token_idx: int = 1,
        df_token_indices: Optional[List[int]] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.ob_token_idx = ob_token_idx
        self.df_token_indices = df_token_indices if df_token_indices is not None else [3, 4, 5, 6]

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )

        # Captured attention maps for L_DA loss computation
        self.recorded_A_df: Optional[torch.Tensor] = None
        self.recorded_A_ob: Optional[torch.Tensor] = None

        self._init_weights()

    def _init_weights(self):
        for m in [self.to_q, self.to_k, self.to_v]:
            nn.init.trunc_normal_(m.weight, std=0.02, a=-0.04, b=0.04)
        nn.init.trunc_normal_(self.to_out[0].weight, std=0.02, a=-0.04, b=0.04)
        if self.to_out[0].bias is not None:
            nn.init.constant_(self.to_out[0].bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        is_defect_stream: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            x: Visual latent features (B, HW, query_dim)
            context: Prompt embeddings (B, T, context_dim)
            is_defect_stream: If False (normal reference image), suppresses defect branch.
        Returns:
            Projected output features: (B, HW, query_dim)
        """
        B, HW, _ = x.shape
        T = context.shape[1]

        q = self.to_q(x).view(B, HW, self.heads, self.dim_head).transpose(1, 2)     # (B, H, HW, d)
        k = self.to_k(context).view(B, T, self.heads, self.dim_head).transpose(1, 2)  # (B, H, T, d)
        v = self.to_v(context).view(B, T, self.heads, self.dim_head).transpose(1, 2)  # (B, H, T, d)

        # ----------------------------------------------------------------------
        # 1. Softmax Attention Map for Alignment Supervision (L_DA)
        # ----------------------------------------------------------------------
        sim = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale  # (B, H, HW, T)
        attn_softmax = F.softmax(sim, dim=-1)

        # Average across heads for L_DA supervision
        attn_mean = attn_softmax.mean(dim=1)  # (B, HW, T)
        safe_ob_idx = min(self.ob_token_idx, T - 1)
        valid_df_indices = [idx for idx in self.df_token_indices if idx < T]

        self.recorded_A_ob = attn_mean[:, :, safe_ob_idx]  # (B, HW)
        if valid_df_indices:
            self.recorded_A_df = attn_mean[:, :, valid_df_indices]  # (B, HW, N_df)
        else:
            self.recorded_A_df = torch.zeros(
                B, HW, len(self.df_token_indices), device=x.device, dtype=x.dtype
            )

        # ----------------------------------------------------------------------
        # 2. Split-Regime Feature Aggregation
        # ----------------------------------------------------------------------
        df_mask = torch.zeros(T, dtype=torch.bool, device=x.device)
        for idx in valid_df_indices:
            df_mask[idx] = True
        ob_mask = ~df_mask

        # Stream A: Unfocused Linear Attention on <ob> and non-defect tokens
        q_phi = F.elu(q) + 1.0  # (B, H, HW, d)
        k_ob_phi = F.elu(k[:, :, ob_mask, :]) + 1.0  # (B, H, T_ob, d)
        v_ob = v[:, :, ob_mask, :]                    # (B, H, T_ob, d)

        # kv_ob: (B, H, d, d)
        kv_ob = torch.einsum("bhnd,bhne->bhde", k_ob_phi, v_ob)
        k_ob_sum = k_ob_phi.sum(dim=2)  # (B, H, d)
        norm_ob = 1.0 / (torch.einsum("bhid,bhd->bhi", q_phi, k_ob_sum) + 1e-6)  # (B, H, HW)
        out_ob = torch.einsum("bhid,bhde,bhi->bhie", q_phi, kv_ob, norm_ob)

        if not is_defect_stream or not valid_df_indices:
            out = out_ob.transpose(1, 2).reshape(B, HW, -1)
            return self.to_out(out)

        # Stream B: Focused Softmax Attention exclusively on <df> tokens
        attn_df = attn_softmax[:, :, :, df_mask]  # (B, H, HW, N_df)
        v_df = v[:, :, df_mask, :]                # (B, H, N_df, d)
        out_df = torch.einsum("bhij,bhjd->bhid", attn_df, v_df)

        # Fused output: Low-pass substrate + sharp defect
        out = (out_ob + out_df).transpose(1, 2).reshape(B, HW, -1)
        return self.to_out(out)


class HybridCrossAttnProcessor:
    """
    Drop-in Attention Processor adapter for Hugging Face Diffusers UNet2DConditionModel.
    Delegates cross-attention to HybridDualCrossAttention while maintaining compatibility.
    """
    def __init__(self, hybrid_attn: HybridDualCrossAttention):
        self.hybrid_attn = hybrid_attn

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        is_defect_stream: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        if encoder_hidden_states is None:
            # Self-attention: standard scaled dot product attention
            batch_size, sequence_length, _ = hidden_states.shape
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)

            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            return hidden_states
        else:
            # Cross-attention routed through Hybrid Dual Attention
            return self.hybrid_attn(
                hidden_states, encoder_hidden_states, is_defect_stream=is_defect_stream
            )
