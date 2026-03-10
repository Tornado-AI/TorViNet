# -*- coding: utf-8 -*-
# -------------------------------

# @软件：PyCharm
# @PyCharm：
# @Python：python=3.11
# @项目：Tornado_video_detection

# -------------------------------

# @文件：TSTVM.py
# @时间：2025/3/8 14:46
# @作者：chenhj
# @邮箱：2426742974@qq.com

# -------------------------------

import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward, DWTInverse
import torch.nn.functional as F
from functools import partial
import numpy as np

class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1 + F.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

class TorViNet(nn.Module):
    def __init__(self):
        super(TorViNet, self).__init__()
        self.kfs = KFS()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(8, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(16, 8, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Linear(8*56*56*8, 1)


    def forward(self, x):
        x = self.kfs(x)
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class KFS(nn.Module):
    def __init__(self, out_channels=1):
        super(KFS, self).__init__()
        self.out_channels = out_channels
        self.conv1 = nn.Sequential(
            nn.Conv3d(3, out_channels, kernel_size=(1, 1, 1)),
            nn.ReLU(inplace=True)
        )
        self.se_block = SE_Block(64, 32)


    def forward(self, x):

        x= self.conv1(x)
        x_weight = self.se_block(x.permute(0, 2, 3, 4, 1))
        # extract 4 biggest and 4 smallest weights index
        _, max_index = torch.topk(x_weight, 4, dim=1)
        _, min_index = torch.topk(x_weight, 4, dim=1, largest=False)
        max_index = max_index.unsqueeze(1).unsqueeze(3).unsqueeze(4).expand(-1, self.out_channels, -1, 224, 224)
        min_index = min_index.unsqueeze(1).unsqueeze(3).unsqueeze(4).expand(-1, self.out_channels, -1, 224, 224)

        x1 = x.gather(2, max_index)
        x2 = x.gather(2, min_index)
        x = torch.cat((x1, x2), dim=2)


        return x

class SE_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, in_channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, t, h, w, c = x.size()
        y = self.avg_pool(x).view(b, t)
        y = self.fc(y)
        return y
    

class FEM(nn.Module):
    def __init__(self, patch_size=16, in_chans=32, num_classes=1, num_features=64,
            depth=12, num_heads=8, mlp_ratio=4., qkv_bias=True, drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=GELU):
        super(FEM, self).__init__()
        self.dwt = DWTForward(J=1, wave='haar', mode='periodization')
        self.patch_embed1 = PatchEmbed([224, 224], 16, 1, 64)
        self.patch_embed2 = PatchEmbed([112, 112], 8, 4, 64)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=num_features,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer
                ) for i in range(depth)
            ]
        )

        self.norm = norm_layer(num_features)

        self.head1 = nn.Linear(num_features, 32)

        self.head2 = nn.Linear(32, num_classes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):   # x: [b, c, t, h, w]

        # wavelet transform
        b, c, t, h, w = x.size()
        x_v = x.permute(0, 2, 1, 3, 4)
        x_v = x_v.contiguous().view(-1, x_v.size(2), x_v.size(3), x_v.size(4))
        xl, (xh, ) = self.dwt(x_v)
        xhh , xhv, xhd = xh[:, :, 0], xh[:, :,1], xh[:, :,2]
        x_v = torch.cat((xl, xhh, xhv, xhd), dim=1)
        x_v = x_v.view(b, t, -1, h//2, w//2).permute(0, 2, 1, 3, 4)     # [b, c * 4, t, h, w]

        # patch embedding
        x1 = self.patch_embed1(x)
        x2 = self.patch_embed2(x_v)

        x = torch.cat((x1, x2), dim=1)
        x = self.blocks(x)
        x = self.norm(x)

        x = self.head1(x[:, 0])
        x = self.relu(x)
        x = self.head2(x)

        return x

class PatchEmbed(nn.Module):
    def __init__(self, input_shape=[224, 224], patch_size=16, in_chans=3, num_features=64, key_frames_num=4,
                 norm_layer=None, flatten=True):
        super().__init__()
        self.num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size) * 2
        self.flatten = flatten

        self.proj = nn.Conv3d(in_chans, num_features, kernel_size=[key_frames_num, patch_size, patch_size],
                              stride=[key_frames_num, patch_size, patch_size])
        self.norm = norm_layer(num_features) if norm_layer else nn.Identity()

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, num_features))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, num_features))

        self.pos_drop = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.proj(x)  # B C T H W
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        cls_token_pe = self.pos_embed[:, 0:1, :]
        img_token_pe = self.pos_embed[:, 1:, :]

        img_token_pe = img_token_pe.view(1, 14, 14, 2, -1).permute(0, 4, 1, 2, 3)
        img_token_pe = F.interpolate(img_token_pe, [14, 14, 2], mode='trilinear', align_corners=False)
        img_token_pe = img_token_pe.permute(0, 2, 3, 4, 1).flatten(1, 3)
        pos_embed = torch.cat([cls_token_pe, img_token_pe], dim=1)

        x = self.pos_drop(x + pos_embed)

        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = (drop, drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


if __name__ == '__main__':

    x = torch.randn(6, 3, 64, 224, 224)
    model = TorViNet()
    out = model(x)
    print(out.shape)

    # compute the number of parameters
    num_parameters = 0
    for parameter in model.parameters():
        num_parameters += parameter.numel()
    print('Number of parameters: %d' % num_parameters)
