from time import time
from tqdm import tqdm
from PIL import Image
import torch
import torchvision.transforms.functional as tvf

from mycv.paths import IMAGENET_DIR

from models.registry import get_model


def read_and_preprocess_im(impath, crop_size=224):
    img = Image.open(impath).convert('RGB')
    img = tvf.resize(img, size=crop_size)
    img = tvf.center_crop(img, output_size=crop_size)
    img: Image.Image
    return img


def speedtest_model(model):
    img_dir = IMAGENET_DIR / 'val'
    img_paths = list(img_dir.rglob('*.*'))[:2000]
    encode_time = 0.0
    for impath in tqdm(img_paths):
        impath = str(impath)
        img = read_and_preprocess_im(impath)
        im = tvf.to_tensor(img).unsqueeze_(0)
        tic = time()
        compressed_obj = model.compress(im)
        encode_time += (time() - tic)

    latency_ms = encode_time / float(len(img_paths)) * 1000
    print(f'{type(model)}: time per image = {latency_ms}ms')
    debug = 1

def speedtest_entropy_coding(model):
    img_paths = range(2000)
    # zC, zH, zW = 24, 56, 56
    zC, zH, zW = 64, 28, 28
    encode_time = 0.0
    for impath in tqdm(img_paths):
        im = torch.randn(1, zC, zH, zW)
        tic = time()
        compressed_obj = model.compress(im)
        # compressed_obj = model.forward(im)
        encode_time += (time() - tic)

    latency_ms = encode_time / float(len(img_paths)) * 1000
    print(f'{type(model)}: time per image = {latency_ms}ms')
    debug = 1


@torch.no_grad()
def main():
    for encoder in [
        # get_model('baseline_s8')().bottleneck_layer,
        # get_model('baseline_s8s')().bottleneck_layer,
        get_model('baseline_s8x')().bottleneck_layer,
    ]:
        encoder.flops_mode_()
        encoder.update()
        speedtest_model(encoder)

    if False:
        from models.entropy_bottleneck import EntropyBottleneck
        encoder = EntropyBottleneck(64)
        encoder.update()
        speedtest_entropy_coding(encoder)

    debug = 1


if __name__ == '__main__':
    main()
