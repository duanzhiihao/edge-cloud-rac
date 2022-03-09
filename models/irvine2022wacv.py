from collections import OrderedDict
import math
import torch
import torch.nn as nn
import torch.nn.functional as tnf

from compressai.layers.gdn import GDN1
from compressai.models.google import CompressionModel

from models.registry import register_model


class BottleneckResNetLayerWithIGDN(CompressionModel):
    def __init__(self, num_enc_channels=16, num_target_channels=256, _flops_mode=False):
        super().__init__(entropy_bottleneck_channels=num_enc_channels)
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
        return x_hat, z_probs


class BottleneckResNet(nn.Module):
    def __init__(self, zdim=24, num_classes=1000, bpp_lmb=0.02, teacher=True):
        super().__init__()
        self.bottleneck_layer = BottleneckResNetLayerWithIGDN(zdim, 256)

        from torchvision.models.resnet import resnet50
        resnet_model = resnet50(pretrained=True, num_classes=num_classes)
        self.layer2 = resnet_model.layer2
        self.layer3 = resnet_model.layer3
        self.layer4 = resnet_model.layer4
        self.avgpool = resnet_model.avgpool
        self.fc = resnet_model.fc

        if teacher:
            from models.teachers import ResNetTeacher
            self._teacher = ResNetTeacher(source='torchvision')
            for p in self._teacher.parameters():
                p.requires_grad_(False)
        else:
            self._teacher = None

        self.bpp_lmb = float(bpp_lmb)

    def forward_train(self, x, y):
        nB, _, imH, imW = x.shape
        x1, p_z = self.bottleneck_layer(x)
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

    @torch.no_grad()
    def forward_teacher(self, x):
        y_teach = self._teacher(x)
        t1, t2, t3, t4 = self._teacher.cache
        assert all([(not t.requires_grad) for t in (t1, t2, t3, t4)])
        return y_teach, (t1, t2, t3, t4)

    def transfer_loss(self, student_features, teacher_features):
        losses = []
        for fake, real in zip(student_features, teacher_features):
            if (fake is not None) and (real is not None):
                assert fake.shape == real.shape, f'fake{fake.shape}, real{real.shape}'
                l_trs = tnf.mse_loss(fake, real, reduction='mean')
                losses.append(l_trs)
            else:
                device = next(self.parameters()).device
                losses.append(torch.zeros(1, device=device))
        return losses

    @torch.no_grad()
    def self_evaluate(self, x, y):
        nB, _, imH, imW = x.shape
        x1, p_z = self.bottleneck_layer(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        feature = self.avgpool(x4)
        feature = torch.flatten(feature, 1)
        logits_hat = self.fc(feature)
        l_cls = tnf.cross_entropy(logits_hat, y, reduction='mean')

        _, top5_idx = torch.topk(logits_hat, k=5, dim=1, largest=True)
        # compute top1 and top5 matches (true positives)
        correct5 = torch.eq(y.cpu().view(nB, 1), top5_idx.cpu())
        stats = OrderedDict()
        stats['top1'] = correct5[:, 0].float().mean().item()
        stats['top5'] = correct5.any(dim=1).float().mean().item()
        bppix = -1.0 * torch.log2(p_z).mean(0).sum() / float(imH * imW)
        stats['bppix'] = bppix.item()
        stats['loss'] = float(l_cls + self.bpp_lmb * bppix)
        return stats


@register_model
def irvine2022wacv(num_classes=1000, bpp_lmb=1.28, teacher=True):
    model = BottleneckResNet(zdim=24, num_classes=num_classes, bpp_lmb=bpp_lmb, teacher=teacher)
    return model
