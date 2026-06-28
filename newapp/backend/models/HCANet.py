import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv3d(dim, hidden_features * 3, kernel_size=(1, 1, 1), bias=bias)

        self.dwconv1 = nn.Conv3d(hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, dilation=2, padding=2, groups=hidden_features, bias=bias)
        self.dwconv3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, dilation=3, padding=3, groups=hidden_features, bias=bias)

        self.project_out = nn.Conv3d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = x.unsqueeze(2)
        x = self.project_in(x)
        x1, x2, x3 = x.chunk(3, dim=1)
        x1 = self.dwconv1(x1).squeeze(2)
        x2 = self.dwconv2(x2.squeeze(2))
        x3 = self.dwconv3(x3.squeeze(2))
        x = F.gelu(x1) * x2 * x3
        x = x.unsqueeze(2)
        x = self.project_out(x)
        return x.squeeze(2)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=1, bias=bias)

        inter_dim = dim * 3
        self.fc = nn.Conv3d(3 * num_heads, 9, kernel_size=1, bias=True)
        self.dep_conv = nn.Conv3d(9 * dim // num_heads, dim, kernel_size=3, padding=1, groups=dim // num_heads, bias=True)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.unsqueeze(2)
        qkv = self.qkv_dwconv(self.qkv(x)).squeeze(2)

        f_conv = qkv.permute(0, 2, 3, 1)
        f_all = qkv.reshape(B, H * W, 3 * self.num_heads, -1).permute(0, 2, 1, 3)
        f_all = self.fc(f_all.unsqueeze(2)).squeeze(2)

        f_conv = f_all.permute(0, 3, 1, 2).reshape(B, 9 * C // self.num_heads, H, W)
        out_conv = self.dep_conv(f_conv.unsqueeze(2)).squeeze(2)

        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (h c) H W -> b h c (H W)', h=self.num_heads)
        k = rearrange(k, 'b (h c) H W -> b h c (H W)', h=self.num_heads)
        v = rearrange(v, 'b (h c) H W -> b h c (H W)', h=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, 'b h c (H W) -> b (h c) H W', h=self.num_heads, H=H, W=W)

        return self.project_out(out.unsqueeze(2)).squeeze(2) + out_conv


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c, embed_dim, bias=False):
        super().__init__()
        self.proj = nn.Conv3d(in_c, embed_dim, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x.unsqueeze(2)).squeeze(2)


class Downsample(nn.Module):
    def __init__(self, in_feat):
        super().__init__()
        self.conv = nn.Conv2d(in_feat, in_feat // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.unshuffle = nn.PixelUnshuffle(3)

    def forward(self, x):
        x = self.conv(x)
        return self.unshuffle(x)


class HCANet(nn.Module):
    def __init__(self, params):
        super(HCANet,self).__init__()
        num_classes=params['data'].get('num_classes',9)
        patchsize = params['data'].get('patchsize',21)
        in_channels = params['data'].get('spectral_size',270)
        assert patchsize % 3 == 0, "Patch size must be divisible by 3 twice (e.g., 15, 21, etc.)"
        dim = params['data'].get('dim',3)
        dim1 = dim
        dim2 = (dim1 // 2) * 9
        dim3 = (dim2 // 2) * 9

        self.patch_embed = OverlapPatchEmbed(in_channels, dim1)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim1, 1, 2.66, False) for _ in range(2)])

        self.down1_2 = Downsample(dim1)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim2, 2, 2.66, False) for _ in range(3)])

        self.down2_3 = Downsample(dim2)
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim3, 3, 2.66, False) for _ in range(3)])

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(dim2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.encoder_level1(x)

        x = self.down1_2(x)
        x = self.encoder_level2(x)

        # x = self.down2_3(x)
        # x = self.encoder_level3(x)

        x = self.global_pool(x).flatten(1)
        return x,self.classifier(x)
