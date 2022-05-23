import torch
import torch.nn as nn
from mycv.utils.torch_utils import flops_benchmark, speed_benchmark
from models.registry import get_model


def main():
    # model = get_model('irvine2022wacv')().bottleneck_layer
    # model = get_model('baseline_vq8')(num_codes=1024).bottleneck_layer
    model = get_model('baseline_s8t')().bottleneck_layer
    model.flops_mode_()

    model.eval()

    shape = (3, 224, 224)
    flops_benchmark(model, input_shape=shape)
    device = torch.device('cpu')
    speed_benchmark(model, input_shapes=[shape], device=device, bs=1, N=4000)

    debug = 1


if __name__ == '__main__':
    main()
