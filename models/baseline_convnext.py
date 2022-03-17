import itertools
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as tnf

from timm.models.convnext import ConvNeXtBlock, LayerNorm2d
from compressai.models.google import CompressionModel

from models.registry import register_model
from mycv.utils.lr_schedulers import get_cosine_lrf
from mycv.utils.coding import get_object_size


class CustomConvBottleneck(CompressionModel):
    def __init__(self, zdim=24, outdim=64):
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
            nn.Conv2d(zdim, outdim, kernel_size=1, stride=1, padding=0),
        )
        self._flops_mode = False

    def flops_mode_(self):
        self.decoder = None
        self._flops_mode = True

    def encode(self, x):
        z = self.encoder(x)
        with torch.autocast('cuda', enabled=False):
            z_quantized, z_probs = self.entropy_bottleneck(z)
        return z_quantized, z_probs

    def forward(self, x):
        z_quantized, z_probs = self.encode(x)
        if self._flops_mode:
            return z_quantized, z_probs
        x_hat = self.decoder(z_quantized)
        return x_hat, z_probs

    @torch.no_grad()
    def compress(self, x):
        z = self.encoder(x)
        compressed_z = self.entropy_bottleneck.compress(z)
        return compressed_z, z.shape[2:]

    @torch.no_grad()
    def decompress(self, compressed_obj):
        bitstreams, latent_shape = compressed_obj
        z_quantized = self.entropy_bottleneck.decompress(bitstreams, latent_shape)
        feature = self.decoder(z_quantized)
        return feature


class BottleneckResNet(nn.Module):
    def __init__(self, zdim=24, num_classes=1000, bpp_lmb=0.02, teacher=True, mode='joint'):
        super().__init__()
        self.bottleneck_layer = CustomConvBottleneck(zdim, 64)

        from torchvision.models.resnet import resnet50
        resnet_model = resnet50(pretrained=True, num_classes=num_classes)
        self.layer1 = resnet_model.layer1
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

        self.initial_bpp_lmb = float(bpp_lmb)
        self.lambdas = [1.0, 1.0, self.initial_bpp_lmb] # cls, trs, bpp

        self.compress_mode = False

    def compress_mode_(self):
        self.bottleneck_layer.update(force=True)
        self.compress_mode = True

    def forward(self, x, y):
        nB, _, imH, imW = x.shape
        x1, p_z = self.bottleneck_layer(x)
        x1 = self.layer1(x1)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        feature = self.avgpool(x4)
        feature = torch.flatten(feature, 1)
        logits_hat = self.fc(feature)

        probs_teach, features_teach = self.forward_teacher(x)

        # bit rate loss
        bppix = -1.0 * torch.log2(p_z).mean(0).sum() / float(imH * imW)
        # label prediction loss
        l_ce = tnf.cross_entropy(logits_hat, y, reduction='mean')

        l_kd = tnf.kl_div(input=torch.log_softmax(logits_hat, dim=1),
                          target=probs_teach, reduction='batchmean')
        # transfer loss
        lmb_cls, lmb_trs, lmb_bpp = self.lambdas
        if lmb_trs > 0:
            l_trs = self.transfer_loss([x1, x2, x3, x4], features_teach)
        else:
            l_trs = [torch.zeros(1, device=x.device) for _ in range(4)]
        loss = lmb_cls * (0.5*l_ce + 0.5*l_kd) + lmb_trs * sum(l_trs) + lmb_bpp * bppix

        stats = OrderedDict()
        stats['loss'] = loss
        stats['bppix'] = bppix.item()
        stats['CE'] = l_ce.item()
        stats['KD'] = l_kd.item()
        for i, lt in enumerate(l_trs):
            stats[f'trs_{i}'] = lt.item()
        stats['acc'] = (torch.argmax(logits_hat, dim=1) == y).sum().item() / float(nB)
        if lmb_trs > 0:
            stats['t_acc'] = (torch.argmax(probs_teach, dim=1) == y).sum().item() / float(nB)
        else:
            stats['t_acc'] = -1.0
        return stats

    @torch.no_grad()
    def forward_teacher(self, x):
        y_teach = self._teacher(x)
        t1, t2, t3, t4 = self._teacher.cache
        assert all([(not t.requires_grad) for t in (t1, t2, t3, t4)])
        assert y_teach.dim() == 2
        y_teach = torch.softmax(y_teach, dim=1)
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
        if self.compress_mode:
            compressed_obj = self.bottleneck_layer.compress(x)
            num_bits = get_object_size(compressed_obj)
            x1 = self.bottleneck_layer.decompress(compressed_obj)
        else:
            x1, p_z = self.bottleneck_layer(x)
        x1 = self.layer1(x1)
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
        if self.compress_mode:
            bppix = num_bits / float(nB * imH * imW)
        else:
            bppix = -1.0 * torch.log2(p_z).mean(0).sum() / float(imH * imW)
        stats['bppix'] = bppix
        stats['loss'] = float(l_cls + self.initial_bpp_lmb * bppix)
        return stats

    def state_dict(self):
        msd = super().state_dict()
        for k in list(msd.keys()):
            if '_teacher' in k:
                msd.pop(k)
        return msd


@register_model
def baseline_convnext(num_classes=1000, bpp_lmb=1.28, teacher=True):
    model = BottleneckResNet(24, num_classes=num_classes, bpp_lmb=bpp_lmb, teacher=teacher)
    return model
