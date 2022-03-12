from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as tnf
import torchvision.models.mobilenetv3 as mbnetv3

from models.registry import register_model
from models.irvine2022wacv import BottleneckResNet


from compressai.models.google import CompressionModel
class BottleneckResNetLayerWithIGDN(CompressionModel):
    def __init__(self, num_enc_channels=24, num_target_channels=256, _flops_mode=False):
        super().__init__(entropy_bottleneck_channels=num_enc_channels)
        from compressai.layers.gdn import GDN1
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
        if _flops_mode:
            self.decoder = None
        self._flops_mode = _flops_mode

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


class MyBottleneckResNet(BottleneckResNet):
    def __init__(self, zdim=24, num_classes=1000, bpp_lmb=0.02, teacher='torchvision'):
        super().__init__()
        self.bottleneck_layer = BottleneckResNetLayerWithIGDN(zdim, 256)

        from torchvision.models.resnet import resnet50
        resnet_model = resnet50(pretrained=True, num_classes=num_classes)
        self.layer1 = resnet_model.layer1
        self.layer2 = resnet_model.layer2
        self.layer3 = resnet_model.layer3
        self.layer4 = resnet_model.layer4
        self.avgpool = resnet_model.avgpool
        self.fc = resnet_model.fc

        if teacher is not None:
            from models.teachers import ResNetTeacher
            self._teacher = ResNetTeacher(source=teacher)
            for p in self._teacher.parameters():
                p.requires_grad_(False)
        else:
            self._teacher = None

        self.bpp_lmb = float(bpp_lmb)

    def forward_train(self, x, y):
        nB, _, imH, imW = x.shape
        x1, p_z = self.bottleneck_layer(x)
        x1 = self.layer1(x1)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        feature = self.avgpool(x4)
        feature = torch.flatten(feature, 1)
        logits_hat = self.fc(feature)

        # bit rate loss
        bppix = -1.0 * torch.log2(p_z).mean(0).sum() / float(imH * imW)
        # label prediction loss
        l_cls = tnf.cross_entropy(logits_hat, y, reduction='mean')
        # transfer loss
        logits_teach, features_teach = self.forward_teacher(x)
        l_trs = self.transfer_loss([x1, x2, x3, x4], features_teach)
        loss = l_cls + sum(l_trs) + self.bpp_lmb * bppix

        stats = OrderedDict()
        stats['loss'] = loss
        stats['bppix'] = bppix.item()
        stats['l_cls'] = l_cls.item()
        for i, lt in enumerate(l_trs):
            stats[f'trs_{i}'] = lt.item()
        stats['studt_acc'] = (torch.argmax(logits_hat, dim=1) == y).sum().item() / float(nB)
        stats['teach_acc'] = (torch.argmax(logits_teach, dim=1) == y).sum().item() / float(nB)
        return stats


@register_model
def irvine_with_layer1(num_classes=1000, bpp_lmb=1.28, teacher=True):
    model = BottleneckResNet(zdim=24, num_classes=num_classes, bpp_lmb=bpp_lmb, teacher='torchvision')
    return model
