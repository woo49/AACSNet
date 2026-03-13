# AACSNet: Angle-Aware and Context-Sensitive Network for Oriented Object Detection

## Requirements

- Python 3.8 (Ubuntu 20.04)
- PyTorch 2.0.0
- CUDA 11.8
- [MMRotate](https://github.com/open-mmlab/mmrotate) 1.x or a compatible framework (mmrotate, mmdet, mmengine)

## Installation

```bash
pip install torch>=1.8.0 torchvision
pip install mmengine>=0.1.0
pip install mmcv>=2.0.0rc2
pip install mmdet>=3.0.0rc2
```

Install MMRotate from source. See the [MMRotate documentation](https://github.com/open-mmlab/mmrotate).

## Directory Structure

```
AACSNet/
├── configs/
│   ├── hrsc.py
│   ├── dotav1.0.py
│   └── dior.py
├── orientedformer/
│   ├── angle_aware_feature_alignment.py
│   ├── cross_level_synergy_module.py
│   ├── oriented_ddq_fcn.py
│   ├── oriented_ddq_rcnn.py
│   └── ...
├── tools/
│   ├── train.py
│   └── test.py
└── README.md
```

## Data Preparation

Follow [MMRotate data preparation](https://github.com/open-mmlab/mmrotate/tree/main/tools/data) for HRSC2016, DOTA, DIOR-R.

Set `data_root` in the corresponding config to your data path.

## Training

```bash
# HRSC2016 (36 epochs)
python tools/train.py configs/hrsc.py

# DOTA v1.0 (12 epochs)
python tools/train.py configs/dotav1.0.py

# DIOR-R (12 epochs)
python tools/train.py configs/dior.py
```

## Testing

```bash
# HRSC2016
python tools/test.py configs/hrsc.py work_dirs/hrsc/epoch_36.pth

# DOTA v1.0
python tools/test.py configs/dotav1.0.py work_dirs/dotav1.0/epoch_12.pth

# DIOR-R
python tools/test.py configs/dior.py work_dirs/dior/epoch_12.pth
```
