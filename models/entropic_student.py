import itertools
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as tnf

from compressai.layers.gdn import GDN1
from compressai.models.google import CompressionModel

from models.registry import register_model


class InputBottleneck(CompressionModel):
    def __init__(self, zdim):
        super().__init__(entropy_bottleneck_channels=zdim)
        self.encoder: nn.Module
        self.decoder: nn.Module
        self._flops_mode = False

    def flops_mode_(self):
        self.decoder = None
        self._flops_mode = True

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

    def update(self, force=False):
        return self.entropy_bottleneck.update(force=force)

    @torch.no_grad()
    def compress(self, x):
        z = self.encoder(x)
        compressed_z = self.entropy_bottleneck.compress(z)
        compressed_obj = (compressed_z, tuple(z.shape[2:]))
        return compressed_obj

    @torch.no_grad()
    def decompress(self, compressed_obj):
        bitstreams, latent_shape = compressed_obj
        z_quantized = self.entropy_bottleneck.decompress(bitstreams, latent_shape)
        feature = self.decoder(z_quantized)
        return feature


class BottleneckResNetLayerWithIGDN(InputBottleneck):
    def __init__(self, num_enc_channels=24, num_target_channels=256):
        super().__init__(num_enc_channels)
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


class BottleneckResNet(nn.Module):
    def __init__(self, zdim=24, num_classes=1000, bpp_lmb=0.02, teacher=True, mode='joint',
                 bottleneck_layer=None):
        super().__init__()
        if bottleneck_layer is None:
            bottleneck_layer = BottleneckResNetLayerWithIGDN(zdim, 256)
        self.bottleneck_layer = bottleneck_layer

        from torchvision.models.resnet import resnet50, ResNet50_Weights
        resnet_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1, num_classes=num_classes)
        # if mode == 'encoder':
        #     for p in resnet_model.parameters():
        #         p.requires_grad_(False)
        #     for m in resnet_model.modules():
        #         if isinstance(m, nn.BatchNorm2d):
        #             m.track_running_stats = False
        #         debug = 1
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

        self.train_mode = mode
        if mode == 'encoder':
            for p in itertools.chain(self.layer2.parameters(), self.layer3.parameters(),
                                     self.layer4.parameters(), self.fc.parameters()):
                p.requires_grad_(False)
            self.lambdas = [1.0, 1.0, self.bpp_lmb] # cls, trs, bpp
        elif mode == 'classifier':
            raise DeprecationWarning()
            for p in self.bottleneck_layer.encoder.parameters():
                p.requires_grad_(False)
            for p in self.bottleneck_layer.entropy_bottleneck.parameters():
                p.requires_grad_(False)
            self.lambdas = [1.0, 0.0, 0.0] # cls, trs, bpp
        elif mode == 'joint':
            self.lambdas = [1.0, 1.0, self.bpp_lmb] # cls, trs, bpp
        else:
            raise ValueError()

        self.compress_mode = False

    def train(self, mode=True):
        super().train(mode)
        if self.train_mode == 'encoder': # make classifier and teacher always eval
            self.layer2.eval()
            self.layer3.eval()
            self.layer4.eval()
            self.fc.eval()
            if self._teacher is not None:
                self._teacher.eval()
        return self

    def compress_mode_(self):
        self.bottleneck_layer.update(force=True)
        self.compress_mode = True

    def forward(self, x, y):
        nB, _, imH, imW = x.shape
        x1, p_z = self.bottleneck_layer(x)
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
        raise NotImplementedError()
        nB, _, imH, imW = x.shape
        if self.compress_mode:
            compressed_obj = self.bottleneck_layer.compress(x)
            num_bits = get_object_size(compressed_obj)
            x1 = self.bottleneck_layer.decompress(compressed_obj)
        else:
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
        if self.compress_mode:
            bppix = num_bits / float(nB * imH * imW)
        else:
            bppix = -1.0 * torch.log2(p_z).mean(0).sum() / float(imH * imW)
        stats['bppix'] = bppix
        stats['loss'] = float(l_cls + self.bpp_lmb * bppix)
        return stats

    def state_dict(self):
        msd = super().state_dict()
        for k in list(msd.keys()):
            if '_teacher' in k:
                msd.pop(k)
        return msd

    def update(self):
        self.bottleneck_layer.update()

    @torch.no_grad()
    def send(self, x):
        compressed_obj = self.bottleneck_layer.compress(x)
        return compressed_obj

    @torch.no_grad()
    def receive(self, compressed_obj):
        feature = self.bottleneck_layer.decompress(compressed_obj)
        x2 = self.layer2(feature)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        feature = self.avgpool(x4)
        feature = torch.flatten(feature, 1)
        p_logits = self.fc(feature)
        return p_logits


@register_model
def matsubara2022wacv(num_classes=1000, bpp_lmb=1.28, teacher=True, mode='joint'):
    """ Supervised Compression for Resource-Constrained Edge Computing Systems

    - Paper: https://arxiv.org/abs/2108.11898
    - Github: https://github.com/yoshitomo-matsubara/supervised-compression

    Args:
        num_classes (int, optional): _description_. Defaults to 1000.
        bpp_lmb (float, optional): _description_. Defaults to 1.28.
        teacher (bool, optional): _description_. Defaults to True.
        mode (str, optional): _description_. Defaults to 'joint'.

    Returns:
        _type_: _description_
    """
    model = BottleneckResNet(24, num_classes=num_classes, bpp_lmb=bpp_lmb, teacher=teacher, mode=mode)
    return model
