from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as tnf
from torchvision.models.mobilenetv3 import ConvNormActivation, InvertedResidual, InvertedResidualConfig

from timm.models.convnext import ConvNeXtBlock, LayerNorm2d
from compressai.models.google import CompressionModel
from models.registry import register_model
from models.irvine2022wacv import InputBottleneck, BottleneckResNet


def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )

class ResBlock(nn.Module):
    def __init__(self, in_out, hidden=None):
        super().__init__()
        hidden = hidden or (in_out // 2)
        self.conv_1 = nn.Conv2d(in_out, hidden, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(hidden, in_out, kernel_size=3, padding=1)

    def forward(self, input):
        x = self.conv_1(tnf.gelu(input))
        x = self.conv_2(tnf.gelu(x))
        out = input + x
        return out


class Bottleneck4(InputBottleneck):
    def __init__(self, hidden, zdim, num_target_channels=256, n_blocks=4):
        super().__init__(zdim)
        if n_blocks > 0:
            modules = [ResBlock(hidden) for _ in range(n_blocks)]
        else:
            modules = [nn.GELU()]

        self.encoder = nn.Sequential(
            nn.Conv2d(3, hidden, kernel_size=4, stride=4, padding=0, bias=True),
            *modules,
            nn.Conv2d(hidden, zdim, kernel_size=1, stride=1, padding=0),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(zdim, num_target_channels * 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(num_target_channels * 2, num_target_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(num_target_channels, num_target_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )

@register_model
def s4res(num_classes=1000, bpp_lmb=1.28, teacher=True):
    bottleneck = Bottleneck4(hidden=72, zdim=24, num_target_channels=256, n_blocks=3)
    model = BottleneckResNet(zdim=24, num_classes=num_classes, bpp_lmb=bpp_lmb, teacher=teacher,
                             bottleneck_layer=bottleneck)
    return model


class Bottleneck8(InputBottleneck):
    def __init__(self, hidden, zdim, num_target_channels=256, n_blocks=4):
        super().__init__(zdim)
        if n_blocks > 0:
            modules = [ResBlock(hidden) for _ in range(n_blocks)]
        else:
            modules = [nn.GELU()]

        self.encoder = nn.Sequential(
            nn.Conv2d(3, hidden, kernel_size=8, stride=8, padding=0, bias=True),
            *modules,
            nn.Conv2d(hidden, zdim, kernel_size=1, stride=1, padding=0),
        )
        self.decoder = nn.Sequential(
            deconv(zdim, num_target_channels),
            nn.GELU(),
            nn.Conv2d(num_target_channels, num_target_channels * 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(num_target_channels * 2, num_target_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(num_target_channels, num_target_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )

@register_model
def s8l_enc(num_classes=1000, bpp_lmb=1.28, teacher=True):
    bottleneck = Bottleneck8(hidden=128, zdim=64, num_target_channels=256, n_blocks=8)
    model = BottleneckResNet(zdim=64, num_classes=num_classes, bpp_lmb=bpp_lmb, teacher=teacher,
                             bottleneck_layer=bottleneck, mode='encoder')
    return model

@register_model
def s8m(num_classes=1000, bpp_lmb=1.28, teacher=True):
    bottleneck = Bottleneck8(hidden=128, zdim=64, num_target_channels=256, n_blocks=4)
    model = BottleneckResNet(zdim=64, num_classes=num_classes, bpp_lmb=bpp_lmb, teacher=teacher,
                             bottleneck_layer=bottleneck)
    return model

@register_model
def s8m_enc(num_classes=1000, bpp_lmb=1.28, teacher=True):
    bottleneck = Bottleneck8(hidden=128, zdim=64, num_target_channels=256, n_blocks=4)
    model = BottleneckResNet(zdim=64, num_classes=num_classes, bpp_lmb=bpp_lmb, teacher=teacher,
                             bottleneck_layer=bottleneck, mode='encoder')
    return model

@register_model
def s8s(num_classes=1000, bpp_lmb=1.28, teacher=True):
    bottleneck = Bottleneck8(hidden=128, zdim=64, num_target_channels=256, n_blocks=2)
    model = BottleneckResNet(zdim=64, num_classes=num_classes, bpp_lmb=bpp_lmb, teacher=teacher,
                             bottleneck_layer=bottleneck)
    return model

@register_model
def s8t(num_classes=1000, bpp_lmb=1.28, teacher=True):
    bottleneck = Bottleneck8(hidden=128, zdim=64, num_target_channels=256, n_blocks=0)
    model = BottleneckResNet(zdim=64, num_classes=num_classes, bpp_lmb=bpp_lmb, teacher=teacher,
         bottleneck_layer=bottleneck)
    return model

@register_model
def s8t_enc(num_classes=1000, bpp_lmb=1.28, teacher=True):
    bottleneck = Bottleneck8(hidden=128, zdim=64, num_target_channels=256, n_blocks=0)
    model = BottleneckResNet(zdim=64, num_classes=num_classes, bpp_lmb=bpp_lmb, teacher=teacher,
                             bottleneck_layer=bottleneck, mode='encoder')
    return model


# @register_model
# def s8_96z(num_classes=1000, bpp_lmb=1.28, teacher=True):
#     bottleneck = Bottleneck8(hidden=128, zdim=96, num_target_channels=256)
#     model = BottleneckResNet(zdim=96, num_classes=num_classes, bpp_lmb=bpp_lmb, teacher=teacher,
#                              bottleneck_layer=bottleneck)
#     return model


# class ResBlockv2(nn.Module):
#     def __init__(self, in_out, hidden=None):
#         super().__init__()
#         hidden = hidden or (in_out // 2)
#         self.conv_1 = nn.Conv2d(in_out, hidden, kernel_size=1)
#         self.conv_2 = nn.Conv2d(hidden, in_out, kernel_size=3, padding=1)

#     def forward(self, input):
#         x = self.conv_1(tnf.gelu(input))
#         x = self.conv_2(tnf.gelu(x))
#         out = input + x
#         return out

# class Bottleneck8v2(InputBottleneck):
#     def __init__(self, hidden, zdim, num_target_channels=256):
#         super().__init__(zdim)
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, hidden, kernel_size=8, stride=8, padding=0, bias=True),
#             ResBlockv2(hidden),
#             ResBlockv2(hidden),
#             ResBlockv2(hidden),
#             ResBlockv2(hidden),
#             # ResBlockv2(hidden),
#             # nn.GELU(),
#             nn.Conv2d(hidden, zdim, kernel_size=1, stride=1, padding=0),
#         )
#         self.decoder = nn.Sequential(
#             deconv(zdim, num_target_channels),
#             nn.GELU(),
#             nn.Conv2d(num_target_channels, num_target_channels * 2, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.GELU(),
#             nn.Conv2d(num_target_channels * 2, num_target_channels, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.GELU(),
#             nn.Conv2d(num_target_channels, num_target_channels, kernel_size=1, stride=1, padding=0, bias=True)
#         )

# @register_model
# def s8v2(num_classes=1000, bpp_lmb=1.28, teacher=True):
#     zdim = 64
#     bottleneck = Bottleneck8v2(hidden=160, zdim=zdim, num_target_channels=256)
#     model = BottleneckResNet(zdim=zdim, num_classes=num_classes, bpp_lmb=bpp_lmb, teacher=teacher,
#                              bottleneck_layer=bottleneck)
#     return model

# class Bottleneck8next(InputBottleneck):
#     def __init__(self, zdim, num_target_channels=256):
#         super().__init__(zdim)
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, zdim, kernel_size=8, stride=8, padding=0, bias=True),
#             ConvNeXtBlock(zdim, conv_mlp=False, mlp_ratio=4),
#             ConvNeXtBlock(zdim, conv_mlp=False, mlp_ratio=4),
#             ConvNeXtBlock(zdim, conv_mlp=False, mlp_ratio=4),
#             ConvNeXtBlock(zdim, conv_mlp=False, mlp_ratio=4),
#         )
#         self.decoder = nn.Sequential(
#             deconv(zdim, num_target_channels),
#             nn.GELU(),
#             nn.Conv2d(num_target_channels, num_target_channels * 2, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.GELU(),
#             nn.Conv2d(num_target_channels * 2, num_target_channels, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.GELU(),
#             nn.Conv2d(num_target_channels, num_target_channels, kernel_size=1, stride=1, padding=0, bias=True)
#         )

# @register_model
# def baseline_s8x(num_classes=1000, bpp_lmb=1.28, teacher=True):
#     model = BottleneckResNet(zdim=64, num_classes=num_classes, bpp_lmb=bpp_lmb, teacher=teacher,
#                              bottleneck_layer=Bottleneck8next(64, 256))
#     return model


# class Bottleneck16(InputBottleneck):
#     def __init__(self, zdim, num_target_channels=256):
#         super().__init__(zdim)
#         width = round(zdim*4/3)
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, width, kernel_size=16, stride=16, padding=0, bias=True),
#             ResBlock(width),
#             ResBlock(width),
#             ResBlock(width),
#             ResBlock(width),
#             nn.Conv2d(width, zdim, kernel_size=1, stride=1, padding=0),
#         )
#         self.decoder = nn.Sequential(
#             deconv(zdim, num_target_channels, stride=4),
#             nn.GELU(),
#             nn.Conv2d(num_target_channels, num_target_channels * 2, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.GELU(),
#             nn.Conv2d(num_target_channels * 2, num_target_channels, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.GELU(),
#             nn.Conv2d(num_target_channels, num_target_channels, kernel_size=1, stride=1, padding=0, bias=True)
#         )

# @register_model
# def baseline_s16(num_classes=1000, bpp_lmb=1.28, teacher=True):
#     model = BottleneckResNet(zdim=192, num_classes=num_classes, bpp_lmb=bpp_lmb, teacher=teacher,
#                              bottleneck_layer=Bottleneck16(192, 256))
#     return model


# class CustomConvBottleneck(CompressionModel):
#     def __init__(self, zdim=24, outdim=256, _flops_mode=False):
#         super().__init__(entropy_bottleneck_channels=zdim)

#         hidden = 64
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, hidden, kernel_size=4, stride=4),
#             LayerNorm2d(hidden),
#             ConvNeXtBlock(hidden, conv_mlp=True, mlp_ratio=2),
#             ConvNeXtBlock(hidden, conv_mlp=True, mlp_ratio=2),
#             nn.Conv2d(hidden, zdim, kernel_size=1, stride=1),
#         )
#         self.decoder = nn.Sequential(
#             nn.Conv2d(zdim, outdim * 2, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(outdim * 2, outdim, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(),
#             nn.Conv2d(outdim, outdim, kernel_size=3, stride=1, padding=1)
#         )

#         if _flops_mode:
#             self.decoder = None
#         self._flops_mode = _flops_mode

#     def flops_mode_(self):
#         self.decoder = None
#         self._flops_mode = True

#     @torch.autocast('cuda', enabled=False)
#     def encode(self, x):
#         z = self.encoder(x)
#         z_quantized, z_probs = self.entropy_bottleneck(z)
#         return z_quantized, z_probs

#     def forward(self, x):
#         z_quantized, z_probs = self.encode(x)
#         if self._flops_mode:
#             return z_quantized, z_probs
#         x_hat = self.decoder(z_quantized)
#         return x_hat, z_probs


# @register_model
# def plus_convnext(num_classes=1000, bpp_lmb=1.28, teacher=True):
#     model = BottleneckResNet(zdim=24, num_classes=num_classes, bpp_lmb=bpp_lmb, teacher=teacher)
#     model.bottleneck_layer = CustomConvBottleneck(24, 256)
#     return model


# class DiscretizedGaussianBottleneck(nn.Module):
#     def __init__(self):
#         super().__init__()
#         from compressai.entropy_models import GaussianConditional
#         self.gaussian_conditional = GaussianConditional(scale_table=None)

#     def update(self):
#         import math
#         # From Balle's tensorflow compression examples
#         lower = self.gaussian_conditional.lower_bound_scale.bound.item()
#         max_scale = 20
#         scale_table = torch.exp(torch.linspace(math.log(lower), math.log(max_scale), steps=32))
#         updated = self.gaussian_conditional.update_scale_table(scale_table)
#         self.gaussian_conditional.update()

#     def compress(self, x):
#         mean = torch.randn_like(x)
#         scales = torch.rand_like(x) * 4
#         indexes = self.gaussian_conditional.build_indexes(scales)
#         y_strings = self.gaussian_conditional.compress(x, indexes)
#         return y_strings

#     @torch.no_grad()
#     def decompress(self, compressed_obj):
#         raise NotImplementedError()
