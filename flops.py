import torch
import torch.nn as nn
from mycv.utils.torch_utils import flops_benchmark, speed_benchmark
from models.registry import get_model


def test_encoder():
    name = 's8l'
    # model = get_model('irvine2022wacv')().bottleneck_layer
    # model = get_model('baseline_vq8')(num_codes=1024).bottleneck_layer
    model = get_model(name)().bottleneck_layer
    model.flops_mode_()

    model.eval()

    if True:
    # if False:
        shape = (3, 224, 224)
        flops_benchmark(model, input_shape=shape)
        device = torch.device('cpu')
        # speed_benchmark(model, input_shapes=[shape], device=device, bs=1, N=2000)
    else:
        from ptflops import get_model_complexity_info
        fpath = f'runs/{name}.txt'
        with open(fpath, 'w') as f:
            ptfl_macs, ptfl_params = get_model_complexity_info(model, shape,
                                        as_strings=False, ost=f, verbose=True)
        exit()
    debug = 1


def test_decoder():
    name = 'baseline_s8v2'
    # model = get_model('irvine2022wacv')().bottleneck_layer
    # model = get_model('baseline_vq8')(num_codes=1024).bottleneck_layer
    model = get_model(name)().bottleneck_layer.decoder

    model.eval()

    shape = (64, 28, 28)
    flops_benchmark(model, input_shape=shape)
    device = torch.device('cpu')
    # speed_benchmark(model, input_shapes=[shape], device=device, bs=1, N=2000)

    if True:
        from ptflops import get_model_complexity_info
        fpath = f'runs/{name}_dec.txt'
        with open(fpath, 'w') as f:
            ptfl_macs, ptfl_params = get_model_complexity_info(model, shape,
                                        as_strings=False, ost=f, verbose=True)
        exit()
    debug = 1


if __name__ == '__main__':
    test_encoder()
    # test_decoder()
