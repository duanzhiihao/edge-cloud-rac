from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as tnf

from models.registry import register_model


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


class VQResNet(nn.Module):
    def __init__(self, num_codes, num_classes=1000, source='torchvision'):
        super().__init__()
        if source == 'torchvision':
            from torchvision.models.resnet import resnet50
            _model = resnet50(pretrained=True, num_classes=num_classes)
            self.act1  = _model.relu
            self.global_pool = _model.avgpool
        elif source == 'timm':
            from timm.models.resnet import resnet50
            _model = resnet50(pretrained=True, num_classes=num_classes)
            self.act1  = _model.act1
            self.global_pool = _model.global_pool
        else:
            raise ValueError()
        self.conv1 = _model.conv1
        self.bn1 = _model.bn1
        self.maxpool = _model.maxpool
        self.layer1 = _model.layer1
        self.layer2 = _model.layer2
        self.layer3 = _model.layer3
        self.layer4 = _model.layer4
        self.fc = _model.fc
        for p in self.parameters():
            p.requires_grad_(False)

        from mycv.models.vae.vqvae.myvqvae import MyCodebookEMA
        self.codebook = MyCodebookEMA(num_codes, embedding_dim=256, commitment_cost=0.25)
        self.dummy = nn.Parameter(torch.zeros(1))

        self.lambdas = [1.0, 0.0, 0.0] # cls, trs, bpp
        self.compress_mode = False

    def forward(self, x, y):
        nB, imC, imH, imW = x.shape
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        vq_loss, x1_quantized, code_indices = self.codebook(x1)
        p_z = self.codebook.get_probs(code_indices)
        x2 = self.layer2(x1_quantized)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = self.global_pool(x4)
        x = torch.flatten(x, 1)
        logits_hat = self.fc(x)

        # bit rate loss
        bppix = -1.0 * torch.log2(p_z).mean(0).sum() / float(imH * imW)
        # label prediction loss
        l_ce = tnf.cross_entropy(logits_hat, y, reduction='mean')

        # l_kd = tnf.kl_div(input=torch.log_softmax(logits_hat, dim=1),
        #                   target=probs_teach, reduction='batchmean')
        # transfer loss
        lmb_cls, lmb_trs, lmb_bpp = self.lambdas
        if lmb_trs > 0:
            raise NotImplementedError()
            l_trs = self.transfer_loss([x1, x2, x3, x4], features_teach)
        else:
            l_trs = [torch.zeros(1, device=x.device) for _ in range(4)]
        # loss = lmb_cls * (0.5*l_ce + 0.5*l_kd) + lmb_trs * sum(l_trs)
        loss = l_ce + vq_loss
        if not loss.requires_grad:
            self.dummy.data.zero_()
            loss = loss + self.dummy

        stats = OrderedDict()
        stats['loss'] = loss
        stats['bppix'] = bppix.item()
        stats['CE'] = l_ce.item()
        stats['VQ'] = vq_loss.item()
        # stats['KD'] = l_kd.item()
        # for i, lt in enumerate(l_trs):
        #     stats[f'trs_{i}'] = lt.item()
        stats['acc'] = (torch.argmax(logits_hat, dim=1) == y).sum().item() / float(nB)
        # if lmb_trs > 0:
        #     stats['t_acc'] = (torch.argmax(probs_teach, dim=1) == y).sum().item() / float(nB)
        # else:
        #     stats['t_acc'] = -1.0
        return stats

    @torch.no_grad()
    def self_evaluate(self, x, y):
        nB, _, imH, imW = x.shape
        if self.compress_mode:
            raise NotImplementedError()
            compressed_obj = self.bottleneck_layer.compress(x)
            num_bits = get_object_size(compressed_obj)
            x1 = self.bottleneck_layer.decompress(compressed_obj)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.act1(x)
            x = self.maxpool(x)
            x1 = self.layer1(x)
            vq_loss, x1_quantized, code_indices = self.codebook(x1)
            p_z = self.codebook.get_probs(code_indices)
        x2 = self.layer2(x1_quantized)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        feature = self.global_pool(x4)
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
            raise NotImplementedError()
            bppix = num_bits / float(nB * imH * imW)
        else:
            bppix = -1.0 * torch.log2(p_z).mean(0).sum() / float(imH * imW)
        stats['bppix'] = bppix
        stats['loss'] = float(l_cls)
        return stats

    def state_dict(self):
        msd = super().state_dict()
        for k in list(msd.keys()):
            if '_teacher' in k:
                msd.pop(k)
        return msd


@register_model
def vqres50(num_codes, num_classes=1000):
    model = VQResNet(num_codes, num_classes=num_classes, source='torchvision')
    return model
