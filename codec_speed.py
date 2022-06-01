import io
from pathlib import Path
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

    latency_ms = encode_time / float(len(img_paths)) * 1000
    print(f'{img_format}: time per image = {latency_ms}ms')
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

    latency_ms = encode_time / float(len(img_paths)) * 1000
    print(f'{img_format}: time per image = {latency_ms}ms')
    debug = 1


def speedtest_bpg(quality, level=8):
    import subprocess
    bpg_dir = Path('d:/libraries/bpg-0.9.8-win64')
    enc_path = bpg_dir / 'bpgenc.exe'
    dec_path = bpg_dir / 'bpgdec.exe'
    bits_path = 'runs/tmp.bpg'

    img_dir = IMAGENET_DIR / 'val'
    img_paths = list(img_dir.rglob('*.*'))[:2000]

    encode_time = 0.0
    for impath in tqdm(img_paths):
        impath = str(impath)
        img = read_and_preprocess_im(impath)
        tmp_path = 'runs/tmp.png'
        # img.save(tmp_path, format='PNG', compress_level=0)
        im = np.array(img)
        flag = cv2.imwrite(tmp_path, im, params=[cv2.IMWRITE_PNG_COMPRESSION, 0])
        assert flag

        tic = time()
        cmd = f'{enc_path} -o {bits_path} -q {quality} -f 420 -e x265 -c rgb -m {level} {tmp_path}'
        return_obj = subprocess.run(cmd)
        encode_time += (time() - tic)

    latency_ms = encode_time / float(len(img_paths)) * 1000
    print(f'BPG {quality}: time per image = {latency_ms}ms')
    debug = 1


if __name__ == '__main__':
    # speedtest_cv2('.jp2')
    # speedtest_cv2('.jpg')
    # speedtest_pil('JPEG')
    # speedtest_pil('WebP')
    speedtest_bpg(quality=5, level=1)
