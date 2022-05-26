import io
from time import time
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2
import torch
import torchvision.transforms.functional as tvf

from mycv.paths import IMAGENET_DIR


def read_and_preprocess_im(impath, crop_size=224):
    img = Image.open(impath)
    img = tvf.resize(img, size=crop_size)
    img = tvf.center_crop(img, output_size=crop_size)
    img: Image.Image
    return img


def speedtest_pil(img_format, **kwargs):
    img_dir = IMAGENET_DIR / 'val'
    img_paths = list(img_dir.rglob('*.*'))[:2000]
    encode_time = 0.0
    for impath in tqdm(img_paths):
        impath = str(impath)
        img = read_and_preprocess_im(impath)

        img_byte_arr = io.BytesIO()
        tic = time()
        img.save(img_byte_arr, format=img_format, **kwargs)
        img_byte_arr = img_byte_arr.getvalue()
        encode_time += (time() - tic)

    latency = encode_time / float(len(img_paths))
    print(f'{img_format}: time per image = {latency}s')
    debug = 1


def speedtest_cv2(img_format):
    img_dir = IMAGENET_DIR / 'val'
    img_paths = list(img_dir.rglob('*.*'))[:2000]
    encode_time = 0.0
    for impath in tqdm(img_paths):
        impath = str(impath)
        img = read_and_preprocess_im(impath)
        im = np.array(img)

        tic = time()
        img_str = cv2.imencode(img_format, im)[1].tobytes()
        encode_time += (time() - tic)

    latency = encode_time / float(len(img_paths))
    print(f'time per image: {latency}s')
    debug = 1


if __name__ == '__main__':
    speedtest_cv2('.jpg')
    speedtest_cv2('.jp2')
    speedtest_pil('JPEG')
    # speedtest_pil('JPEG2000')
    speedtest_pil('WebP')
