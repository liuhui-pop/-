import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class CNN(nn.Module):
    def __init__(self, out_channel, band=1):
        super().__init__()
        self.conv1 = nn.Conv2d(band, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(128, out_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, num=1):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc(x))
        return x

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(0.)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(0.)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CTMF(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(4)])
        self.drop_path = DropPath(0.1)
        self.mlp = Mlp(dim)
        self.attn = Attention(dim, num_heads)

    def forward(self, x, y):
        x_ = self.drop_path(self.attn(self.norm[1](torch.cat([y[:, 0:1], x[:, 1:]], dim=1))))
        y_ = self.drop_path(self.attn(self.norm[1](torch.cat([x[:, 0:1], y[:, 1:]], dim=1))))

        x = x + torch.cat([(x[:, 0:1] + y_[:, 0:1]), x_[:, 1:]], dim=1)
        y = y + torch.cat([(y[:, 0:1] + x_[:, 0:1]), y_[:, 1:]], dim=1)

        x = x + self.drop_path(self.mlp(self.norm[2](x)))
        y = y + self.drop_path(self.mlp(self.norm[2](y)))

        return x, y

class CASST(nn.Module):
    def __init__(self, params):
        super().__init__()
        num_classes = params['data'].get('num_classes', 16)
        embed_dim = 512
        self.conv_h = nn.Sequential(
            nn.Conv2d(params['data'].get('spectral_size', 30), embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU()
        )
        self.cnn = CNN(out_channel=embed_dim)

        self.spa_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.spe_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.register_parameter("spa_pos_embed", nn.Parameter(torch.zeros(1, 1, embed_dim)))
        self.register_parameter("spe_pos_embed", nn.Parameter(torch.zeros(1, 1, embed_dim)))

        self.CTMF = nn.Sequential(*[CTMF(embed_dim, num_heads=8) for _ in range(2)])
        self.pos_drop = nn.Dropout(p=0.4)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.cls_head = nn.Linear(embed_dim * 2, num_classes)
        self.pre_logits = nn.Identity()

    def forward_spa(self, x):
        B, C, H, W = x.shape
        x = self.conv_h(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        cls_token = self.spa_cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        spa_pos_embed = torch.zeros(1, x.shape[1], x.shape[2]).to(x.device)
        nn.init.trunc_normal_(spa_pos_embed, std=.02)
        x = self.pos_drop(x + spa_pos_embed)
        return x

    def forward_spe(self, y):
        B, C, H, W = y.shape
        cnn_output = []
        for index in range(C):
            band = y[:, index, :, :].unsqueeze(1)
            band_feat = self.cnn(band)
            cnn_output.append(band_feat.unsqueeze(1))
        y = torch.cat(cnn_output, dim=1)
        cls_token = self.spe_cls_token.expand(B, -1, -1)
        y = torch.cat((cls_token, y), dim=1)

        spe_pos_embed = torch.zeros(1, y.shape[1], y.shape[2]).to(y.device)
        nn.init.trunc_normal_(spe_pos_embed, std=.02)
        y = self.pos_drop(y + spe_pos_embed)
        return y

    def forward(self, x):
        x = x.squeeze(1)
        x_spa = self.forward_spa(x)
        x_spe = self.forward_spe(x)

        for blk in self.CTMF:
            x_spa, x_spe = blk(x_spa, x_spe)

        spa_cls = self.norm1(self.pre_logits(x_spa[:, 0]))
        spe_cls = self.norm2(self.pre_logits(x_spe[:, 0]))
        cls = torch.cat((spa_cls, spe_cls), dim=1)
        return self.cls_head(cls)


