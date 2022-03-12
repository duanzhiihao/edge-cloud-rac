import torch
import torch.nn as nn


class MobileCloudBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.testing_stats = (0, 0.0, 0.0) # num, bpp, bits per dim
        self.detach = False
        self.flops_mode = False

    def init_testing(self):
        self.testing_stats = (0, 0.0, 0.0) # num, bpp, bits per dim

    @torch.autocast('cuda', enabled=False)
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