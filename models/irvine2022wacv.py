import math
import torch
import torch.nn as nn

from compressai.layers.gdn import GDN1
from compressai.models.google import FactorizedPrior
from models.mobile import MobileCloudBase


class BottleneckResNetLayerWithIGDN(FactorizedPrior):
    def __init__(self, num_enc_channels=16, num_target_channels=256, _flops_mode=False):
        super().__init__(N=num_enc_channels, M=num_enc_channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, num_enc_channels * 4, kernel_size=5, stride=2, padding=2, bias=False),
            GDN1(num_enc_channels * 4),
            nn.Conv2d(num_enc_channels * 4, num_enc_channels * 2, kernel_size=5, stride=2, padding=2, bias=False),
            GDN1(num_enc_channels * 2),
            nn.Conv2d(num_enc_channels * 2, num_enc_channels, kernel_size=2, stride=1, padding=0, bias=False)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(num_enc_channels, num_target_channels * 2, kernel_size=2, stride=1, padding=1, bias=False),
            GDN1(num_target_channels * 2, inverse=True),
            nn.Conv2d(num_target_channels * 2, num_target_channels, kernel_size=2, stride=1, padding=0, bias=False),
            GDN1(num_target_channels, inverse=True),
            nn.Conv2d(num_target_channels, num_target_channels, kernel_size=2, stride=1, padding=1, bias=False)
        )
        self.updated = False
        if _flops_mode:
            self.decoder = None
        self._flops_mode = _flops_mode

    @torch.autocast('cuda', enabled=False)
    def forward_entropy(self, z):
        z = z.float()
        z_hat, z_probs = self.entropy_bottleneck(z)
        return z_hat, z_probs

    def forward(self, x):
        x = x.float()
        z = self.encoder(x)
        z_hat, z_probs = self.forward_entropy(z)
        if self._flops_mode:
            dummy = z_hat.sum() + z_probs.sum()
            return dummy
        x_hat = self.decoder(z_hat)
        return x_hat, z_probs, None


class BaselinePlus(BottleneckResNetLayerWithIGDN):
    def __init__(self, num_enc_channels=16, num_target_channels=256, _flops_mode=False):
        super().__init__(num_enc_channels, num_target_channels, _flops_mode)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, num_enc_channels, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GELU(),
            nn.Conv2d(num_enc_channels, num_enc_channels, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GELU(),
            nn.Conv2d(num_enc_channels, num_enc_channels, kernel_size=2, stride=1, padding=0, bias=False)
        )


class BottleneckVQa(nn.Module):
    def __init__(self, num_enc_channels=64, num_target_channels=256, _flops_mode=False):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, num_enc_channels, kernel_size=5, stride=2, padding=2, bias=False),
            # GDN1(num_enc_channels),
            nn.GELU(),
            nn.Conv2d(num_enc_channels, num_enc_channels, kernel_size=5, stride=2, padding=2, bias=False),
            # GDN1(num_enc_channels),
            nn.GELU(),
            nn.Conv2d(num_enc_channels, num_enc_channels, kernel_size=2, stride=1, padding=0, bias=False)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(num_enc_channels, num_target_channels * 2, kernel_size=2, stride=1, padding=1, bias=False),
            nn.GELU(),
            # GDN1(num_target_channels * 2, inverse=True),
            nn.Conv2d(num_target_channels * 2, num_target_channels, kernel_size=2, stride=1, padding=0, bias=False),
            nn.GELU(),
            # GDN1(num_target_channels, inverse=True),
            nn.Conv2d(num_target_channels, num_target_channels, kernel_size=2, stride=1, padding=1, bias=False)
        )
        from mycv.models.vae.vqvae.myvqvae import MyCodebookEMA
        num_codes = 24
        self.codebook = MyCodebookEMA(num_codes, embedding_dim=num_enc_channels, commitment_cost=0.25)
        self.updated = False
        if _flops_mode:
            self.decoder = None
        self._flops_mode = _flops_mode

    @torch.autocast('cuda', enabled=False)
    def forward_entropy(self, z):
        z = z.float()
        vq_loss, z_quantized, code_indices = self.codebook.forward(z)
        nB, nC, nH, nW = z.shape
        z_probs = self.codebook.get_probs(code_indices)
        # z_hat, z_probs = self.entropy_bottleneck(z)
        return vq_loss, z_quantized, z_probs

    def forward(self, x):
        x = x.float()
        z = self.encoder(x)
        vq_loss, z_hat, z_probs = self.forward_entropy(z)
        if self._flops_mode:
            dummy = z_hat.sum() + z_probs.sum()
            return dummy
        x_hat = self.decoder(z_hat)
        return x_hat, z_probs, vq_loss


class BottleneckResNetBackbone(MobileCloudBase):
    def __init__(self, num_classes=1000, zdim=24, bottleneck='factorized'):
        super().__init__()
        if bottleneck == 'factorized':
            self.bottleneck_layer = BottleneckResNetLayerWithIGDN(zdim, 256)
        elif bottleneck == 'vqa':
            self.bottleneck_layer = BottleneckVQa(zdim, 256)
        else:
            raise ValueError()

        from torchvision.models.resnet import resnet50
        # from timm.models.resnet import resnet50
        resnet_model = resnet50(pretrained=True, num_classes=num_classes)

        self.layer2 = resnet_model.layer2
        self.layer3 = resnet_model.layer3
        self.layer4 = resnet_model.layer4
        self.avgpool = resnet_model.avgpool
        self.fc = resnet_model.fc
        # self.inplanes = resnet_model.inplanes
        # self.updated = False
        self.cache = [None, None, None, None]

    def forward(self, x):
        x, probs, vq_loss = self.bottleneck_layer(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = self.avgpool(x4)
        x = torch.flatten(x, 1)
        y = self.fc(x)

        self.cache = [None, x2, x3, x4]
        if not self.training:
            return y, probs
        return y, probs, vq_loss

    def forward_entropy(self, z):
        raise NotImplementedError()

    # def update(self):
    #     self.bottleneck_layer.update()
    #     self.updated = True
