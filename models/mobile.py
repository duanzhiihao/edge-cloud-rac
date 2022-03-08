import torch
import torch.nn as nn
import torch.nn.functional as tnf
import torch.cuda.amp as amp
import torchvision as tv

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from mycv.utils.coding import compute_bpp


class IntegerQuantization(nn.Module):
    def __init__(self, n_ch):
        super().__init__()
        self.momentum = 0.99
        self.register_buffer('estimated_p', torch.ones(n_ch, 512).div_(512))
        self.estimated_p: torch.Tensor
        self.dummy = nn.parameter.Parameter(torch.zeros(1,n_ch,1,1))

    def _update_stats(self, x: torch.Tensor):
        assert not x.requires_grad
        x.clamp_(min=-255, max=256).round_()
        # x = x.to(dtype=torch.int64)
        # x: (nB, nC, nH, nW)
        x = x.permute(1, 0, 2, 3).flatten(1)
        for i, x_i in enumerate(x):
            hist = torch.histc(x_i, bins=512, min=-255, max=256)
            # hist = torch.bincount(x_i, minlength=256)
            assert hist.sum() == x.shape[1], f'hist{hist.shape}sum={hist.sum()}, x{x.shape}'
            pi = hist.float() / x.shape[1]
            self.estimated_p[i, :].mul_(self.momentum).add_(pi, alpha=1-self.momentum)
            assert torch.isclose(self.estimated_p[i, :].sum(), torch.ones(1, device=x.device))
        debug = 1

    def compute_likelihood(self, x: torch.Tensor):
        xhat = x.detach().clone().round_().add_(255).to(torch.int64)
        assert xhat.min() >= 0 and xhat.max() <= 511
        p_x = []
        for i in range(xhat.shape[1]):
            indexs = xhat[:, i, :, :]
            p = self.estimated_p[i, indexs]
            p_x.append(p.unsqueeze(1))
        p_x = torch.cat(p_x, dim=1)
        return p_x

    def forward(self, x: torch.Tensor):
        if self.training:
            # x = x + torch.rand_like(x) - 0.5
            x = torch.clamp(x, min=-255, max=256)
            xd = x.detach()
            x = x + (torch.round(xd) - xd)
            self._update_stats(xd)
        else:
            assert not x.requires_grad
            x.clamp_(min=-255, max=256)
            x = torch.round_(x)
        p_x = self.compute_likelihood(x)
        x = x + self.dummy
        x = x - self.dummy
        return x, p_x


class AEEntropy(nn.Module):
    def __init__(self, n_ch):
        super().__init__()
        self.entropy = EntropyBottleneck(n_ch)
        self.enc = nn.Sequential(
            nn.Conv2d(n_ch, n_ch, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_ch, n_ch, kernel_size=3, stride=1, padding=1),
        )
        self.dec = nn.Sequential(
            nn.Conv2d(n_ch, n_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(n_ch, n_ch, kernel_size=3, stride=2, output_padding=1, padding=1)
        )

    def forward(self, x):
        z = self.enc(x)
        zhat, p_z = self.entropy(z)
        xhat = self.dec(zhat)
        return xhat, p_z


class Identidty(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x, None


def get_entropy_model(name, channels):
    if name == 'bottleneck':
        entropy_model = EntropyBottleneck(channels)
    elif name == 'quantize':
        entropy_model = IntegerQuantization(channels)
    elif name == 'ae_bottleneck':
        entropy_model = AEEntropy(channels)
    elif name == 'hyper':
        raise NotImplementedError()
        entropy_model = GaussianConditional(None)
    elif name == 'identity':
        entropy_model = Identidty()
    else:
        raise ValueError()
    return entropy_model


class MobileCloudBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.testing_stats = (0, 0.0, 0.0) # num, bpp, bits per dim
        self.detach = False
        self.flops_mode = False

    def init_testing(self):
        self.testing_stats = (0, 0.0, 0.0) # num, bpp, bits per dim

    @amp.autocast(enabled=False)
    def forward_entropy(self, z):
        if self.detach:
            z = z.detach().float()
        else:
            z = z.float()
        z, p_z = self.entropy_model(z)
        return z, p_z

    def forward_cls(self, x):
        assert not self.training
        yhat, p_z = self.forward(x)

        # update testing stats
        num, bpp, bpd = self.testing_stats
        nB, _, imh, imw = x.shape
        batch_bpp = -1.0 * torch.log2(p_z).sum() / (imh*imw)
        # batch_bpp = compute_bpp(p_z, imh*imw, batch_reduction='sum')
        bpp = (bpp * num + batch_bpp) / (num + nB)
        batch_bpd = - torch.log2(p_z).mean() * nB
        bpd = (bpd * num + batch_bpd) / (num + nB)
        num += nB
        self.testing_stats = (num, bpp.item(), bpd.item())

        return yhat


class ResNet50MC(MobileCloudBase):
    def __init__(self, cut_after='layer1', entropy_model='bottleneck'):
        super().__init__()
        model = tv.models.resnet50(pretrained=True)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.fc = model.fc

        # 3, 4, 6, 3
        stride2channel = {'layer1': 256, 'layer2': 512}
        channels = stride2channel[cut_after]
        self.entropy_model = get_entropy_model(entropy_model, channels)
        self.cut_after = cut_after

    def forward(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if self.cut_after == 'layer1':
            x, p_z = self.forward_entropy(x)
        x = self.layer2(x)
        if self.cut_after == 'layer2':
            x, p_z = self.forward_entropy(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        yhat = self.fc(x)

        return yhat, p_z


class VGG11MC(MobileCloudBase):
    def __init__(self, cut_after=10, entropy_model='bottleneck'):
        super().__init__()
        model = tv.models.vgg.vgg11(pretrained=True)
        self.features = model.features
        self.avgpool = model.avgpool
        self.classifier = model.classifier

        self.cut_after = int(cut_after)
        channels = {5: 128, 10: 256, 15: 512, 20: 512}
        self.entropy_model = get_entropy_model(entropy_model, channels[self.cut_after])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for mi, module in enumerate(self.features):
            x = module(x)
            if mi == self.cut_after:
                x, p_z = self.forward_entropy(x)
        # x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x, p_z


class MobileV3MC(MobileCloudBase):
    def __init__(self, cut_after=2, entropy_model='bottleneck'):
        super().__init__()
        from timm.models.mobilenetv3 import mobilenetv3_large_100
        model = mobilenetv3_large_100(pretrained=True)
        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        self.act1 = model.act1
        self.blocks = model.blocks
        self.global_pool = model.global_pool
        self.conv_head = model.conv_head
        self.act2 = model.act2
        self.flatten = model.flatten
        self.classifier = model.classifier
        self.drop_rate = 0.2

        self.cut_after = int(cut_after)
        channels = {0: 16, 1: 24, 2: 40, 3: 80}
        self.entropy_model = get_entropy_model(entropy_model, channels[self.cut_after])

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == self.cut_after:
                x, p_z = self.forward_entropy(x)
            debug = 1
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = self.flatten(x)
        if self.drop_rate > 0.:
            x = tnf.dropout(x, p=self.drop_rate, training=self.training)
        x = self.classifier(x)
        return x, p_z


def main():
    from mycv.datasets.imagenet import imagenet_val
    # model = ResNet50MC('layer2', 'quantize')
    # model = VGG11MC(10)
    model = MobileV3MC(cut_after=2, entropy_model='bottleneck')
    # msd = torch.load('runs/best.pt')
    # model.load_state_dict(msd['model'])
    model = model.cuda()
    model.eval()

    # resuls = imagenet_val(model, batch_size=4, workers=0)
    model.init_testing()
    resuls = imagenet_val(model, batch_size=64, workers=8)
    print(resuls)
    print(model.testing_stats)


if __name__ == '__main__':
    main()
