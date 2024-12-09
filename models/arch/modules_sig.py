# --------------------------------------------------------
# Reversible Column Networks
# Copyright (c) 2022 Megvii Inc.
# Licensed under The Apache License 2.0 [see LICENSE for details]
# Written by Yuxuan Cai
# --------------------------------------------------------

import imp
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath




class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, dim, expand_dim, out_dim, kernel_size=3, layer_scale_init_value=1e-6, drop_path=0.):
        super().__init__()
        drop_out_rate = 0. 
        dw_channel = expand_dim
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=kernel_size, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = expand_dim
        self.conv4 = nn.Conv2d(in_channels=dim, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=out_dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(dim)
        self.norm2 = LayerNorm2d(dim)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.ones((1, dim, 1, 1)) * layer_scale_init_value, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones((1, dim, 1, 1)) * layer_scale_init_value, requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class UpSampleConvnext(nn.Module):
    def __init__(self, ratio, inchannel, outchannel):
        super().__init__()
        self.ratio = ratio
        self.channel_reschedule = nn.Sequential(  
                                        # LayerNorm(inchannel, eps=1e-6, data_format="channels_last"),
                                        nn.Linear(inchannel, outchannel),
                                        LayerNorm(outchannel, eps=1e-6, data_format="channels_last"))
        self.upsample  = nn.Upsample(scale_factor=2**ratio, mode='bilinear')
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.channel_reschedule(x)
        x = x = x.permute(0, 3, 1, 2)
        
        return self.upsample(x)

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first", elementwise_affine = True):
        super().__init__()
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            if self.elementwise_affine:
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNextBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, in_channel, hidden_dim, out_channel, kernel_size=3, layer_scale_init_value=1e-6, drop_path= 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, groups=in_channel) # depthwise conv
        self.norm = nn.LayerNorm(in_channel, eps=1e-6)
        self.pwconv1 = nn.Linear(in_channel, hidden_dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, out_channel)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((out_channel)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class Decoder(nn.Module):
    def __init__(self, depth=[2,2,2,2], dim=[112, 72, 40, 24], block_type = None, kernel_size = 3) -> None:
        super().__init__()
        self.depth = depth
        self.dim = dim
        self.block_type = block_type
        self._build_decode_layer(dim, depth, kernel_size)
        self.pixelshuffle=nn.PixelShuffle(2)
        # self.star_relu=StarReLU()
        self.projback_ = nn.Sequential(
            nn.Conv2d(
                in_channels=dim[-1],
                out_channels=2 ** 2 * 3 , kernel_size=1),
            nn.PixelShuffle(2)
        )
        self.projback_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=dim[-1],
                out_channels=2 ** 2 * 3, kernel_size=1),
            nn.PixelShuffle(2)
        )
        
    def _build_decode_layer(self, dim, depth, kernel_size):
        normal_layers = nn.ModuleList()
        upsample_layers = nn.ModuleList()
        proj_layers = nn.ModuleList()

        norm_layer = LayerNorm

        for i in range(1, len(dim)):
            module = [self.block_type(dim[i], dim[i], dim[i], kernel_size) for _ in range(depth[i])]
            normal_layers.append(nn.Sequential(*module))
            upsample_layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
            proj_layers.append(nn.Sequential(
                nn.Conv2d(dim[i-1], dim[i], 1, 1), 
                norm_layer(dim[i]),
                # StarReLU() #self.star_relu()
                nn.GELU()
                ))
        for i in range(1, len(dim)):
            module = [self.block_type(dim[i], dim[i], dim[i], kernel_size) for _ in range(depth[i])]
            normal_layers.append(nn.Sequential(*module))
            upsample_layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
            proj_layers.append(nn.Sequential(
                               nn.Conv2d(dim[i-1], dim[i], 1, 1),
                               norm_layer(dim[i]),
            ))
        self.normal_layers = normal_layers
        self.upsample_layers = upsample_layers
        self.proj_layers = proj_layers

    def _forward_stage(self, stage, x):
        x = self.proj_layers[stage](x)
        x = self.upsample_layers[stage](x)
        return self.normal_layers[stage](x)

    def forward(self, c3, c2, c1, c0):
        c0_clean, c0_ref = c0, c0 
        c1_clean, c1_ref = c1, c1 
        c2_clean, c2_ref = c2, c2 
        c3_clean, c3_ref = c3, c3 
        x_clean = self._forward_stage(0, c3_clean) * c2_clean
        x_clean = self._forward_stage(1, x_clean) * c1_clean
        x_clean = self._forward_stage(2, x_clean) * c0_clean
        x_clean = self.projback_(x_clean)
        
        x_ref = self._forward_stage(3, c3_ref) * c2_ref
        x_ref = self._forward_stage(4, x_ref) * c1_ref
        x_ref = self._forward_stage(5, x_ref) * c0_ref
        x_ref = self.projback_2(x_ref)

        x=torch.cat((x_clean,x_ref),dim=1)
        return x

class SimDecoder(nn.Module):
    def __init__(self, in_channel, encoder_stride) -> None:
        super().__init__()
        self.projback = nn.Sequential(
            LayerNorm(in_channel),
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=encoder_stride ** 2 * 3, kernel_size=1),
            nn.PixelShuffle(encoder_stride),
        )

    def forward(self, c3):
        return self.projback(c3)
    

class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """
    def __init__(self, scale_value=1.0, bias_value=0.0,
        scale_learnable=True, bias_learnable=True, 
        mode=None, inplace=True):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
            requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
            requires_grad=bias_learnable)
    def forward(self, x):
        return self.scale * self.relu(x)**2 + self.bias
