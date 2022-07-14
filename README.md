# Efficient Feature Compression for Edge-Cloud Systems


## Install
**Requirements**:
- `pytorch>=1.12`, `tqdm`, `compressai` ([link](https://github.com/InterDigitalInc/CompressAI))
- Code has been tested on Windows and Linux with Intel CPUs and Nvidia GPUs (Python 3.9, CUDA 11.3).

**Download**:
1. Download the repository;
2. Download the pre-trained model checkpoints and put them in the `checkpoints` folder.

**Pre-trained model checkpoints**:
|               | Latency | Link                                                                                               |
|---------------|---------|----------------------------------------------------------------------------------------------------|
| `ours_n0`     | 3.95ms  | [Google Drive](https://drive.google.com/file/d/1fmxiExP13TzUfNgvrnXfK3ApG8kSVLuf/view?usp=sharing) |
| `ours_n4`     | 6.70ms  | [Google Drive](https://drive.google.com/file/d/1rFoUs8r5obwz5KXJI00-DOunQE-uFrlO/view?usp=sharing) |
| `ours_n8`     | 10.2ms  | [Google Drive](https://drive.google.com/file/d/1_wijavWfihU3rnERAomr8KiDLMswZ_D3/view?usp=sharing) |
| `ours_n0_enc` | 3.95ms  | [Google Drive](https://drive.google.com/file/d/1gJAtdMvp8nMjlvzVL-_lUn2OvO0N3fa9/view?usp=sharing) |
| `ours_n4_enc` | 6.70ms  | [Google Drive](https://drive.google.com/file/d/1TtW76UY7-gDQ1miFRPWUhsKCG2fnLxZ2/view?usp=sharing) |
| `ours_n8_enc` | 10.2ms  | [Google Drive](https://drive.google.com/file/d/1vZfBoa4ZzvrRaJXDuNTZjUMbjIV0IZ88/view?usp=sharing) |


## Usage (TBD)
- **Feature compression**: See `demo-sender.ipynb`.
- **Prediction from compressed feature**: See `demo-receiver.ipynb`.


## Evaluation on ImageNet
Evaluate all models on ImageNet:
`python evalutate.py -d /path/to/imagenet/val -b batch_size -w cpu_workers`


## Training
TBD


## License
TBD
