import torch.nn as nn
import torch.nn.functional as tnf

from models.registry import register_model
from models.entropic_student import InputBottleneck, BottleneckResNet


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
def ours_n8(num_classes=1000, bpp_lmb=1.28, teacher=True):
    bottleneck = Bottleneck8(hidden=128, zdim=64, num_target_channels=256, n_blocks=8)
    model = BottleneckResNet(zdim=64, num_classes=num_classes, bpp_lmb=bpp_lmb, teacher=teacher,
                             bottleneck_layer=bottleneck)
    return model

@register_model
def ours_n8_enc(num_classes=1000, bpp_lmb=1.28, teacher=True):
    bottleneck = Bottleneck8(hidden=128, zdim=64, num_target_channels=256, n_blocks=8)
    model = BottleneckResNet(zdim=64, num_classes=num_classes, bpp_lmb=bpp_lmb, teacher=teacher,
                             bottleneck_layer=bottleneck, mode='encoder')
    return model

@register_model
def ours_n4(num_classes=1000, bpp_lmb=1.28, teacher=True):
    bottleneck = Bottleneck8(hidden=128, zdim=64, num_target_channels=256, n_blocks=4)
    model = BottleneckResNet(zdim=64, num_classes=num_classes, bpp_lmb=bpp_lmb, teacher=teacher,
                             bottleneck_layer=bottleneck)
    return model

@register_model
def ours_n4_enc(num_classes=1000, bpp_lmb=1.28, teacher=True):
    bottleneck = Bottleneck8(hidden=128, zdim=64, num_target_channels=256, n_blocks=4)
    model = BottleneckResNet(zdim=64, num_classes=num_classes, bpp_lmb=bpp_lmb, teacher=teacher,
                             bottleneck_layer=bottleneck, mode='encoder')
    return model

@register_model
def ours_n0(num_classes=1000, bpp_lmb=1.28, teacher=True):
    bottleneck = Bottleneck8(hidden=128, zdim=64, num_target_channels=256, n_blocks=0)
    model = BottleneckResNet(zdim=64, num_classes=num_classes, bpp_lmb=bpp_lmb, teacher=teacher,
         bottleneck_layer=bottleneck)
    return model

@register_model
def ours_n0_enc(num_classes=1000, bpp_lmb=1.28, teacher=True):
    bottleneck = Bottleneck8(hidden=128, zdim=64, num_target_channels=256, n_blocks=0)
    model = BottleneckResNet(zdim=64, num_classes=num_classes, bpp_lmb=bpp_lmb, teacher=teacher,
                             bottleneck_layer=bottleneck, mode='encoder')
    return model
