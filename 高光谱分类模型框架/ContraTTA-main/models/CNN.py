import PIL
import time, json
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.nn.init as init
from einops import rearrange, repeat
import collections
import torch.nn as nn
from utils import device
import random
# class Downsample(nn.Module):
#     def __init__(self, in_feat):
#         super().__init__()
#         self.conv = nn.Conv2d(in_feat, in_feat // 2, kernel_size=3, stride=1, padding=1, bias=False)
#         self.unshuffle = nn.PixelUnshuffle(3)

#     def forward(self, x):
#         x = self.conv(x)
#         return self.unshuffle(x)
class SelfAttention(nn.Module):
    def __init__(self, dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** 0.5

    def forward(self, x):
        # x: (batch, seq_len, dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (B, seq, seq)
        attn_probs = torch.softmax(attn_scores, dim=-1)             # (B, seq, seq)

        output = torch.bmm(attn_probs, V)                            # (B, seq, dim)
        return output

class CNNNet(nn.Module):
    def __init__(self, params):
        super(CNNNet, self).__init__()
        self.params = params
        net_params = params['net']
        data_params = params['data']
        dim = params['data'].get('dim',3)

        num_classes = data_params.get("num_classes", 16)
        self.patch_size = patch_size = data_params.get("patch_size", 13)
        self.spectral_size = data_params.get("spectral_size", 200)

        self.wh = self.patch_size * self.patch_size
        self.ln_attn = nn.LayerNorm(dim*2)
        conv2d_out = 64
        # dim2 = (conv2d_out // 2) * 9
        kernal = 3
        padding = 1
        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=self.spectral_size, out_channels=conv2d_out, kernel_size=(kernal, kernal), stride=1, padding=(padding,padding)),
            nn.BatchNorm2d(conv2d_out),
            nn.ReLU(),
            # featuremap 
            nn.Conv2d(in_channels=conv2d_out,out_channels=conv2d_out,kernel_size=3,padding=1),
            nn.BatchNorm2d(conv2d_out),
            nn.ReLU()
        )
        # self.downsample = Downsample(in_feat=conv2d_out)
        self.projector = nn.Linear(self.wh, 1) 
        
        # dim = dim2
        dim2 = dim*2
        self.attn = SelfAttention(dim2)
        linear_dim = dim2 * 2
        self.classifier_mlp = nn.Sequential(
            nn.Linear(dim2, linear_dim),
            nn.BatchNorm1d(linear_dim), 
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(linear_dim, num_classes),
        )

    def encoder_block(self, x):
        '''
        x: (batch, s, w, h), s=spectral, w=weigth, h=height
        '''
        x_pixel = x 

        x_pixel = self.conv2d_features(x_pixel)
   
        # x_pixel = self.downsample(x_pixel)

        b, s, w, h = x_pixel.shape
        img = w * h

        x_pixel = rearrange(x_pixel, 'b s w h-> b (w h) s') # (batch, spe, w*h)
  
        # x_pixel = self.projector(x_pixel) # (batch, spe, 1)
        # print(x_pixel.shape)
        x_pixel = self.attn(x_pixel)                          # (B, seq_len, dim)
        # x_pixel = self.ln_attn(x_pixel)
        x_pixel = x_pixel.mean(dim=1)
        # x_pixel = rearrange(x_pixel, 'b s 1 -> b s')
        # print(x_pixel.shape)

        return x_pixel,self.classifier_mlp(x_pixel)

    def forward(self, x):
        '''
        x: (batch, s, w, h), s=spectral, w=weigth, h=height

        '''
        logit_x = self.encoder_block(x)
        return  logit_x