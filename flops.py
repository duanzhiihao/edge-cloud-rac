import torch
from mycv.utils.torch_utils import flops_benchmark, speed_benchmark
from models.registry import get_model


def main():
    # from models.irvine2022wacv import BottleneckResNetLayerWithIGDN
    # model = BottleneckResNetLayerWithIGDN(num_enc_channels=24, _flops_mode=True)
    # model = BottleneckEncoder(num_enc_channels=24, _flops_mode=True)
    from models.irvine_plus import plus_v1, plus_conv
    model = plus_conv().bottleneck_layer
    model.flops_mode_()

    model.eval()

    shape = (3, 224, 224)
    flops_benchmark(model, input_shape=shape)
    device = torch.device('cpu')
    speed_benchmark(model, input_shapes=[shape], device=device, bs=1, N=4000)

    debug = 1


if __name__ == '__main__':
    main()
