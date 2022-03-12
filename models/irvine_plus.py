from functools import partial
import torch
import torch.nn as nn
from torchvision.models.mobilenetv3 import ConvNormActivation, InvertedResidual, InvertedResidualConfig

from compressai.models.google import CompressionModel
from models.registry import register_model
from models.irvine2022wacv import BottleneckResNet


class CustomBottleneck(CompressionModel):
    def __init__(self, enc_configs, dec_configs, outdim=256, _flops_mode=False):
        zdim = enc_configs[-1].out_channels
        super().__init__(entropy_bottleneck_channels=zdim)

        enc_layers = []
        firstconv_output_channels = enc_configs[0].input_channels
        enc_layers.append(ConvNormActivation(3, firstconv_output_channels, kernel_size=3, stride=2,
                                             norm_layer=None, activation_layer=nn.ReLU))
        for cfg in enc_configs:
            enc_layers.append(InvertedResidual(cfg, norm_layer=None))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        # for cfg in dec_configs:
        #     dec_layers.append(InvertedResidual(cfg, norm_layer=None))
        dec_layers.append(ConvNormActivation(zdim, outdim*2, 3, 1, norm_layer=None))
        dec_layers.append(ConvNormActivation(outdim*2, outdim, 3, 1, norm_layer=None))
        dec_layers.append(ConvNormActivation(outdim, outdim, 3, 1, norm_layer=None))
        self.decoder = nn.Sequential(*dec_layers)

        if _flops_mode:
            self.decoder = None
        self._flops_mode = _flops_mode

    def flops_mode_(self):
        self.decoder = None
        self._flops_mode = True

    @torch.autocast('cuda', enabled=False)
    def encode(self, x):
        z = self.encoder(x)
        z_quantized, z_probs = self.entropy_bottleneck(z)
        return z_quantized, z_probs

    def forward(self, x):
        z_quantized, z_probs = self.encode(x)
        if self._flops_mode:
            return z_quantized, z_probs
        x_hat = self.decoder(z_quantized)
        return x_hat, z_probs


@register_model
def plus_v1(num_classes=1000, bpp_lmb=1.28, teacher=True):
    model = BottleneckResNet(zdim=24, num_classes=num_classes, bpp_lmb=bpp_lmb, teacher=teacher)
    cfg = partial(InvertedResidualConfig, use_se=False, activation='RE', dilation=1, width_mult=1.0)
    enc_configs = [
        cfg(16, kernel=3, expanded_channels=24, out_channels=16, stride=1),
        cfg(16, kernel=3, expanded_channels=64, out_channels=24, stride=2),
        cfg(24, kernel=3, expanded_channels=72, out_channels=24, stride=1),
        cfg(24, kernel=3, expanded_channels=72, out_channels=24, stride=1),
    ]
    dec_configs = [
        cfg(24, kernel=3, expanded_channels=72, out_channels=64, stride=1),
    ]
    model.bottleneck_layer = CustomBottleneck(enc_configs, dec_configs, 256)
    return model


from timm.models.convnext import ConvNeXtBlock, LayerNorm2d
# class ConvNeXtBlock5x5(ConvNeXtBlock):
#     def __init__(self, dim, drop_path=0., ls_init_value=1e-6, conv_mlp=False, mlp_ratio=4, norm_layer=None):
#         super().__init__(dim, drop_path, ls_init_value, conv_mlp, mlp_ratio, norm_layer)
#         self.conv_dw = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)  # depthwise conv


class CustomConvBottleneck(CompressionModel):
    def __init__(self, zdim=24, outdim=256, _flops_mode=False):
        super().__init__(entropy_bottleneck_channels=zdim)

        hidden = 64
        self.encoder = nn.Sequential(
            nn.Conv2d(3, hidden, kernel_size=4, stride=4),
            LayerNorm2d(hidden),
            ConvNeXtBlock(hidden, conv_mlp=True, mlp_ratio=2),
            ConvNeXtBlock(hidden, conv_mlp=True, mlp_ratio=2),
            nn.Conv2d(hidden, zdim, kernel_size=1, stride=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(zdim, outdim * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(outdim * 2, outdim, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(outdim, outdim, kernel_size=3, stride=1, padding=1)
        )

        if _flops_mode:
            self.decoder = None
        self._flops_mode = _flops_mode

    def flops_mode_(self):
        self.decoder = None
        self._flops_mode = True

    @torch.autocast('cuda', enabled=False)
    def encode(self, x):
        z = self.encoder(x)
        z_quantized, z_probs = self.entropy_bottleneck(z)
        return z_quantized, z_probs

    def forward(self, x):
        z_quantized, z_probs = self.encode(x)
        if self._flops_mode:
            return z_quantized, z_probs
        x_hat = self.decoder(z_quantized)
        return x_hat, z_probs


@register_model
def plus_convnext(num_classes=1000, bpp_lmb=1.28, teacher=True):
    model = BottleneckResNet(zdim=24, num_classes=num_classes, bpp_lmb=bpp_lmb, teacher=teacher)
    model.bottleneck_layer = CustomConvBottleneck(24, 256)
    return model
