"""
models/ViT.py
标准 Vision Transformer，适配 HSI patch 输入。
输入格式: (batch, spectral, patch_size, patch_size)

思路：
  1. 先用 1x1 卷积将光谱维度降维到 embed_dim
  2. 把空间位置展开成 token 序列
  3. 加上可学习的 [CLS] token 和位置编码
  4. 过 Transformer Encoder
  5. 用 [CLS] token 做分类
"""

import torch
import torch.nn as nn
import math
from einops import rearrange


class PatchEmbed(nn.Module):
    """将 HSI patch 转成 token 序列。"""
    def __init__(self, spectral_size, patch_size, embed_dim):
        super().__init__()
        self.num_tokens = patch_size * patch_size  # 每个空间位置一个 token
        # 1x1 卷积：光谱降维
        self.proj = nn.Sequential(
            nn.Conv2d(spectral_size, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)                          # (B, embed_dim, H, W)
        x = rearrange(x, 'b d h w -> b (h w) d')  # (B, num_tokens, embed_dim)
        return x


class TransformerEncoderBlock(nn.Module):
    """单层 Transformer Encoder（Pre-LN 风格）。"""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class ViTNet(nn.Module):
    def __init__(self, params):
        super(ViTNet, self).__init__()
        data_params = params['data']
        net_params  = params['net']

        num_classes   = data_params.get('num_classes', 9)
        spectral_size = data_params.get('spectral_size', 103)
        patch_size    = data_params.get('patch_size', 13)

        embed_dim  = net_params.get('embed_dim', 64)
        num_heads  = net_params.get('num_heads', 4)
        depth      = net_params.get('depth', 4)       # Transformer 层数
        mlp_ratio  = net_params.get('mlp_ratio', 4.0)
        dropout    = net_params.get('dropout', 0.1)

        num_tokens = patch_size * patch_size

        # ── Patch Embedding ───────────────────────────────────────
        self.patch_embed = PatchEmbed(spectral_size, patch_size, embed_dim)

        # ── [CLS] token + 位置编码 ────────────────────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(dropout)

        # ── Transformer Encoder ───────────────────────────────────
        self.blocks = nn.Sequential(*[
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # ── 分类头 ────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, num_classes),
        )

    def forward(self, x):
        """
        x: (batch, spectral, H, W)
        返回 (feat, logits)，与 BaseTrainer 兼容
        """
        B = x.size(0)

        # Patch embedding
        tokens = self.patch_embed(x)                       # (B, N, D)

        # 拼接 [CLS] token
        cls = self.cls_token.expand(B, -1, -1)            # (B, 1, D)
        tokens = torch.cat([cls, tokens], dim=1)           # (B, N+1, D)

        # 位置编码
        tokens = self.pos_drop(tokens + self.pos_embed)    # (B, N+1, D)

        # Transformer
        tokens = self.blocks(tokens)                       # (B, N+1, D)
        tokens = self.norm(tokens)

        # 取 [CLS] token 作为特征
        feat   = tokens[:, 0]                             # (B, D)
        logits = self.head(feat)                          # (B, num_classes)

        return feat, logits
