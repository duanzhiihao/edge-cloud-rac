from functools import partial
import torch
import torch.nn as nn
from torchvision.models.mobilenetv3 import ConvNormActivation, InvertedResidual, InvertedResidualConfig

from models.registry import register_model
from models.irvine2022wacv import BottleneckResNet


from compressai.models.google import CompressionModel
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
