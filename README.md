# Efficient Feature Compression for Edge-Cloud Systems


## Install
**Requirements**:
- `pytorch>=1.9`, `tqdm`, `compressai` ([link](https://github.com/InterDigitalInc/CompressAI))
- Code has been tested on Windows and Linux with Intel CPUs and Nvidia GPUs (Python 3.9, CUDA 11.3).

**Download**:
1. Download the repository;
2. Download the pre-trained model checkpoints and put them in the `qres-vae/checkpoints` folder.

**Pre-trained model checkpoints**:
|     | FLOPs | Latency | Delta accuracy | Link |
|-----|-------|---------|----------------|------|
| N=0 |       |         |                |      |
| N=4 |       |         |                |      |
| N=8 |       |         |                |      |


## Usage (TBD)
- **Feature compression**: See `demo-sender.ipynb`.
- **Prediction from compressed feature**: See `demo-receiver.ipynb`.
- **Evaluation**: See `evaluate.py`.


## Training
TBD


## License
TBD
