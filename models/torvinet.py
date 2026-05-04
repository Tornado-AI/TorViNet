# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from typing import Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


SelectionMode = Literal["hard", "soft"]


class DFSM(nn.Module):
    """
    Dynamic Frame-level Selection Module.

    Input:
        x: Tensor of shape (B, 3, T, H, W)

    Output:
        selected: Tensor of shape (B, C, 2N, H, W)
        scores:   Tensor of shape (B, T)

    The hard mode follows the paper's Top-N/Bottom-N strategy:
        - compute frame saliency scores
        - select top-N and bottom-N frames
        - multiply selected frames by their saliency weights
        - concatenate positive and negative selected frames

    The soft mode is a differentiable relaxation that produces N soft-selected
    positive frames and N soft-selected negative frames.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_channels: int = 4,
        num_select: int = 4,
        selection_mode: SelectionMode = "hard",
        soft_temperature: float = 0.1,
    ) -> None:
        super().__init__()
        if num_select <= 0:
            raise ValueError("num_select must be positive.")
        if selection_mode not in ("hard", "soft"):
            raise ValueError("selection_mode must be 'hard' or 'soft'.")

        self.embed_channels = embed_channels
        self.num_select = num_select
        self.selection_mode = selection_mode
        self.soft_temperature = soft_temperature

        self.frame_embed = nn.Sequential(
            nn.Conv3d(in_channels, embed_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(embed_channels),
            nn.ReLU(inplace=True),
        )

        # Temporal attention scorer. The paper describes a TAM adapted from SE:
        # global pooling over channel and spatial dimensions yields one scalar per frame.
        self.score_mlp = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1),
        )

    def _frame_scores(self, x_embed: torch.Tensor) -> torch.Tensor:
        # x_embed: (B, C, T, H, W)
        pooled = x_embed.mean(dim=(1, 3, 4), keepdim=False)  # (B, T)
        scores = self.score_mlp(pooled.unsqueeze(-1)).squeeze(-1)  # (B, T)
        return scores

    def _hard_select(self, x_embed: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x_embed.shape
        if 2 * self.num_select > t:
            raise ValueError(f"2*num_select={2*self.num_select} cannot exceed T={t}.")

        weights = torch.sigmoid(scores)  # (B, T)

        _, top_idx = torch.topk(scores, self.num_select, dim=1, largest=True, sorted=True)
        _, bot_idx = torch.topk(scores, self.num_select, dim=1, largest=False, sorted=True)

        def gather_frames(indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            # indices: (B, N)
            idx = indices[:, None, :, None, None].expand(-1, c, -1, h, w)
            frames = x_embed.gather(dim=2, index=idx)  # (B, C, N, H, W)
            frame_weights = weights.gather(dim=1, index=indices)  # (B, N)
            frame_weights = frame_weights[:, None, :, None, None]
            return frames, frame_weights

        top_frames, top_weights = gather_frames(top_idx)
        bot_frames, bot_weights = gather_frames(bot_idx)

        # Paper equations: X_p^N = F_p^N ⊙ W_p, X_n^N = F_n^N ⊙ W_n
        top_frames = top_frames * top_weights
        bot_frames = bot_frames * bot_weights

        return torch.cat([top_frames, bot_frames], dim=2)

    def _soft_select(self, x_embed: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """
        Differentiable approximate top-k without replacement.

        For positive frames, use scores.
        For negative frames, use -scores.
        Each selected "frame" is a weighted mixture over all T frames.
        """
        b, c, t, h, w = x_embed.shape

        def soft_k(logits: torch.Tensor) -> torch.Tensor:
            selected = []
            remaining_logits = logits
            for _ in range(self.num_select):
                alpha = F.softmax(remaining_logits / self.soft_temperature, dim=1)  # (B, T)
                mixed = torch.einsum("bt,bcthw->bchw", alpha, x_embed)
                selected.append(mixed.unsqueeze(2))
                # Approximate without-replacement update.
                remaining_logits = remaining_logits + torch.log(torch.clamp(1.0 - alpha, min=1e-6))
            return torch.cat(selected, dim=2)  # (B, C, N, H, W)

        top_soft = soft_k(scores)
        bot_soft = soft_k(-scores)
        return torch.cat([top_soft, bot_soft], dim=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError("Expected x of shape (B, C, T, H, W).")
        x_embed = self.frame_embed(x)
        scores = self._frame_scores(x_embed)

        if self.selection_mode == "hard":
            selected = self._hard_select(x_embed, scores)
        else:
            selected = self._soft_select(x_embed, scores)

        return selected, scores


class HaarDWT2D(nn.Module):
    """
    2D Haar wavelet transform applied frame-wise.

    Input:
        x: (B, C, T, H, W)
    Output:
        concat([LL, LH, HL, HH]) with shape (B, 4C, T, H/2, W/2)

    H and W are cropped to even sizes if needed.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("Expected x of shape (B, C, T, H, W).")

        h_even = x.shape[-2] - (x.shape[-2] % 2)
        w_even = x.shape[-1] - (x.shape[-1] % 2)
        x = x[..., :h_even, :w_even]

        x00 = x[..., 0::2, 0::2]
        x01 = x[..., 0::2, 1::2]
        x10 = x[..., 1::2, 0::2]
        x11 = x[..., 1::2, 1::2]

        # Orthonormal Haar-like scaling.
        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (x00 - x01 + x10 - x11) * 0.5
        hl = (x00 + x01 - x10 - x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5

        return torch.cat([ll, lh, hl, hh], dim=1)


class SFMHA(nn.Module):
    """
    Spatial-Frequency Multi-Head Attention.

    Spatial path:
        selected frames -> 3D convolution -> spatial tokens as Q

    Frequency path:
        selected frames -> Haar DWT -> 1x1x1 projection -> frequency tokens as K,V

    Output:
        fused spatial-frequency tokens, shape (B, Lq, D)
    """

    def __init__(
        self,
        in_channels: int = 4,
        embed_dim: int = 128,
        num_heads: int = 8,
        patch_size: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.spatial_conv = nn.Sequential(
            nn.Conv3d(in_channels, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(embed_dim),
            nn.GELU(),
        )

        self.dwt = HaarDWT2D()
        self.freq_proj = nn.Sequential(
            nn.Conv3d(in_channels * 4, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm3d(embed_dim),
            nn.GELU(),
        )

        self.spatial_patch = nn.Conv3d(
            embed_dim, embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size),
            bias=False,
        )
        self.freq_patch = nn.Conv3d(
            embed_dim, embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size),
            bias=False,
        )

        self.q_norm = nn.LayerNorm(embed_dim)
        self.kv_norm = nn.LayerNorm(embed_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_norm = nn.LayerNorm(embed_dim)

    @staticmethod
    def _to_tokens(x: torch.Tensor) -> torch.Tensor:
        # (B, D, T, H, W) -> (B, T*H*W, D)
        return x.flatten(2).transpose(1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial tokens.
        fs = self.spatial_conv(x)
        fs = self.spatial_patch(fs)
        q = self._to_tokens(fs)

        # Frequency tokens.
        ff = self.dwt(x)
        ff = self.freq_proj(ff)
        ff = self.freq_patch(ff)
        kv = self._to_tokens(ff)

        q_norm = self.q_norm(q)
        kv_norm = self.kv_norm(kv)

        attn_out, _ = self.cross_attn(query=q_norm, key=kv_norm, value=kv_norm, need_weights=False)
        return self.out_norm(q + attn_out)


class LocalContrastMLP(nn.Module):
    """
    Localized Contrast-aware MLP.

    This module implements the paper's idea:
        Z = ReLU(F W1 + b1)
        A = Softmax(Z Z^T)
        Z_tilde = A Z
        F_out = ReLU(Flatten(Z_tilde) W2 + b2)

    To avoid the quadratic memory cost over all video tokens, the attention is
    computed inside local token windows. The default 392 tokens corresponds to
    the paper's 2*S^2 with S=14.
    """

    def __init__(
        self,
        dim: int = 128,
        hidden_dim: int = 64,
        window_tokens: int = 392,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.window_tokens = window_tokens
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        residual = x
        x = self.norm(x)

        b, l, d = x.shape
        pad_len = (self.window_tokens - l % self.window_tokens) % self.window_tokens
        if pad_len:
            x = F.pad(x, (0, 0, 0, pad_len))
        l_pad = x.shape[1]
        num_windows = l_pad // self.window_tokens

        xw = x.view(b * num_windows, self.window_tokens, d)
        z = F.relu(self.fc1(xw), inplace=True)  # (B*nw, W, hidden)

        scale = z.shape[-1] ** -0.5
        attn = torch.matmul(z, z.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)

        z_refined = torch.matmul(attn, z)
        out = F.relu(self.fc2(z_refined), inplace=True)
        out = self.dropout(out)

        out = out.view(b, l_pad, d)
        if pad_len:
            out = out[:, :l, :]

        return residual + out


class ContrastAwareSpatiotemporalTransformer(nn.Module):
    """
    One paper-aligned transformer encoder block:
        SFMHA -> LC-MLP
    """

    def __init__(
        self,
        in_channels: int = 4,
        embed_dim: int = 128,
        num_heads: int = 8,
        patch_size: int = 16,
        lc_hidden_dim: int = 64,
        lc_window_tokens: int = 392,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.sfmha = SFMHA(
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_heads=num_heads,
            patch_size=patch_size,
            dropout=dropout,
        )
        self.lc_mlp = LocalContrastMLP(
            dim=embed_dim,
            hidden_dim=lc_hidden_dim,
            window_tokens=lc_window_tokens,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.sfmha(x)
        tokens = self.lc_mlp(tokens)
        return tokens


class TorViNet(nn.Module):
    """
    Paper-aligned TorViNet.

    Expected input:
        (B, 3, T, H, W), e.g. (B, 3, 64, 224, 224)

    Output:
        logits of shape (B, num_classes)

    For binary classification:
        - use num_classes=1 with BCEWithLogitsLoss, or
        - use num_classes=2 with CrossEntropyLoss.
    """

    def __init__(
        self,
        num_classes: int = 1,
        in_channels: int = 3,
        dfsm_channels: int = 4,
        num_select: int = 4,
        selection_mode: SelectionMode = "hard",
        embed_dim: int = 128,
        num_heads: int = 8,
        patch_size: int = 16,
        lc_hidden_dim: int = 64,
        lc_window_tokens: int = 392,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.dfsm = DFSM(
            in_channels=in_channels,
            embed_channels=dfsm_channels,
            num_select=num_select,
            selection_mode=selection_mode,
        )

        self.encoder = ContrastAwareSpatiotemporalTransformer(
            in_channels=dfsm_channels,
            embed_dim=embed_dim,
            num_heads=num_heads,
            patch_size=patch_size,
            lc_hidden_dim=lc_hidden_dim,
            lc_window_tokens=lc_window_tokens,
            dropout=dropout,
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, return_scores: bool = False):
        selected, scores = self.dfsm(x)
        tokens = self.encoder(selected)
        pooled = self.norm(tokens).mean(dim=1)
        logits = self.classifier(pooled)

        if return_scores:
            return logits, scores
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


