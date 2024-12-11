import numpy as np
from models.arch.focalnet import build_focalnet
import torch
import torch.nn as nn
from models.arch.modules_sig import ConvNextBlock, Decoder, LayerNorm, NAFBlock, SimDecoder, UpSampleConvnext
from models.arch.reverse_function import ReverseFunction
from timm.models.layers import trunc_normal_

class Fusion(nn.Module):
    def __init__(self, level, channels, first_col) -> None:
        super().__init__()

        self.level = level
        self.first_col = first_col
        self.down = nn.Sequential(
            nn.Conv2d(channels[level - 1], channels[level], kernel_size=2, stride=2),
            LayerNorm(channels[level], eps=1e-6, data_format="channels_first"),
        ) if level in [1, 2, 3] else nn.Identity()
        if not first_col:
            self.up = UpSampleConvnext(1, channels[level + 1], channels[level]) if level in [0, 1, 2] else nn.Identity()
    
    def forward(self, *args):

        c_down, c_up = args
        channels_dowm=c_down.size(1)
        if self.first_col:
            x_clean = self.down(c_down)
            return x_clean
        if c_up is not None:
            channels_up=c_up.size(1)
        if self.level == 3:
            x_clean = self.down(c_down)
        else:
            x_clean = self.up(c_up) + self.down(c_down)
            
        return x_clean 

class Level(nn.Module):
    def __init__(self, level, channels, layers, kernel_size, first_col, dp_rate=0.0, block_type=ConvNextBlock) -> None:
        super().__init__()
        countlayer = sum(layers[:level])
        expansion = 4
        self.fusion = Fusion(level, channels, first_col)
        modules = [block_type(channels[level], expansion * channels[level], channels[level], kernel_size=kernel_size,
                                 layer_scale_init_value=1e-6, drop_path=dp_rate[countlayer + i]) for i in
                   range(layers[level])]
        self.blocks = nn.Sequential(*modules)

    def forward(self, *args):
        x = self.fusion(*args)
        x_clean = self.blocks(x)
        return x_clean


class SubNet(nn.Module):
    def __init__(self, channels, layers, kernel_size, first_col, dp_rates, save_memory, block_type=ConvNextBlock) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        self.save_memory = save_memory
        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[0], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None

        self.level0 = Level(0, channels, layers, kernel_size, first_col, dp_rates, block_type=block_type)

        self.level1 = Level(1, channels, layers, kernel_size, first_col, dp_rates, block_type=block_type)

        self.level2 = Level(2, channels, layers, kernel_size, first_col, dp_rates, block_type=block_type)

        self.level3 = Level(3, channels, layers, kernel_size, first_col, dp_rates, block_type=block_type)

    def _forward_nonreverse(self, *args):
        x, c0, c1, c2, c3 = args
        c0 = self.alpha0 * c0 + self.level0(x, c1)
        c1 = self.alpha1 * c1 + self.level1(c0, c2)
        c2 = self.alpha2 * c2 + self.level2(c1, c3)
        c3 = self.alpha3 * c3 + self.level3(c2, None)
        return c0, c1, c2, c3

    def _forward_reverse(self, *args):
        x, c0, c1, c2, c3 = args
        local_funs = [self.level0, self.level1, self.level2, self.level3]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3]
        _, c0, c1, c2, c3 = ReverseFunction.apply(
            local_funs, alpha, *args)

        return c0, c1, c2, c3

    def forward(self, *args):

        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        self._clamp_abs(self.alpha3.data, 1e-3)
        if self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign

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

class FullNet_NLP(nn.Module):
    def __init__(self, channels=[32, 64, 96, 128], layers=[2, 3, 6, 3], num_subnet=5,loss_col=4, kernel_size=3, num_classes=1000,
                 drop_path=0.0, save_memory=True, inter_supv=True, head_init_scale=None, pretrained_cols=16) -> None:
        super().__init__()
        self.num_subnet = num_subnet
        self.Loss_col=(loss_col+1)
        self.inter_supv = inter_supv
        self.channels = channels
        self.layers = layers
        self.stem_comp = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=5, stride=2, padding=2),
            LayerNorm(channels[0], eps=1e-6, data_format="channels_first")
        )
        self.prompt=nn.Sequential(nn.Linear(in_features=6,out_features=512),
                                  StarReLU(),
                                  nn.Linear(in_features=512,out_features=channels[0]),
                                  StarReLU(),
                                  )
        dp_rate = [x.item() for x in torch.linspace(0, drop_path, sum(layers))]
        for i in range(num_subnet):
            first_col = True if i == 0 else False
            self.add_module(f'subnet{str(i)}', SubNet(
                channels, layers, kernel_size, first_col, 
                dp_rates=dp_rate, save_memory=save_memory,
                block_type=NAFBlock))

        channels.reverse()
        self.decoder_blocks = nn.ModuleList(
            [Decoder(depth=[1, 1, 1, 1], dim=channels, block_type=NAFBlock, kernel_size=3) for _ in
             range(3)])

        self.apply(self._init_weights)
        self.baseball = build_focalnet('focalnet_L_384_22k_fl4')
        self.baseball_adapter = nn.ModuleList()
        self.baseball_adapter.append(nn.Conv2d(192, 64, kernel_size=1))
        self.baseball_adapter.append(nn.Conv2d(192, 64, kernel_size=1))
        self.baseball_adapter.append(nn.Conv2d(192 * 2, 64 * 2, kernel_size=1))
        self.baseball_adapter.append(nn.Conv2d(192 * 4, 64 * 4, kernel_size=1))
        self.baseball_adapter.append(nn.Conv2d(192 * 8, 64 * 8, kernel_size=1))
    
    def forward(self, x_in,alpha,prompt=True):
        x_cls_out = []
        x_img_out = []
        c0, c1, c2, c3 = 0, 0, 0, 0
        interval = self.num_subnet // 4

        x_base, x_stem = self.baseball(x_in)
        c0, c1, c2, c3 = x_base
        x_stem = self.baseball_adapter[0](x_stem)
        c0, c1, c2, c3 = self.baseball_adapter[1](c0),\
                         self.baseball_adapter[2](c1),\
                         self.baseball_adapter[3](c2),\
                         self.baseball_adapter[4](c3)
        if prompt==True:
            prompt_alpha=self.prompt(alpha)
            prompt_alpha = prompt_alpha.unsqueeze(-1).unsqueeze(-1)
            x=prompt_alpha*x_stem
        else :
            x = x_stem
        for i in range(self.num_subnet):
            c0, c1, c2, c3 = getattr(self, f'subnet{str(i)}')(x, c0, c1, c2, c3)
            if i>(self.num_subnet-self.Loss_col):
                x_img_out.append(torch.cat([x_in, x_in], dim=-3) - self.decoder_blocks[-1](c3, c2, c1, c0) )
 
        return x_cls_out, x_img_out

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            trunc_normal_(module.weight, std=.02)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            nn.init.constant_(module.bias, 0)

   
