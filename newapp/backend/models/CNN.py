"""
models/SimpleCNN.py
纯净 3层卷积 CNN，用于 HSI 分类。
输入格式: (batch, spectral, patch_size, patch_size)
"""

import torch
import torch.nn as nn
from einops import rearrange


class SimpleCNN(nn.Module):
    def __init__(self, params):
        super(SimpleCNN, self).__init__()
        data_params = params['data']

        num_classes   = data_params.get('num_classes', 9)
        spectral_size = data_params.get('spectral_size', 103)
        patch_size    = data_params.get('patch_size', 13)

        # ── 空间-光谱卷积主干 ──────────────────────────────────────
        self.features = nn.Sequential(
            # Layer 1：跨光谱特征提取
            nn.Conv2d(spectral_size, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Layer 2：空间特征聚合
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Layer 3：进一步压缩
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # 全局平均池化，去掉空间维度
        self.gap = nn.AdaptiveAvgPool2d(1)

        # ── 分类头 ────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        """
        x: (batch, spectral, H, W)
        返回 (feat, logits) 格式，与 BaseTrainer 兼容
        """
        feat = self.features(x)       # (B, 32, H, W)
        feat = self.gap(feat)         # (B, 32, 1, 1)
        feat = feat.view(feat.size(0), -1)  # (B, 32)
        logits = self.classifier(feat)      # (B, num_classes)
        return feat, logits
