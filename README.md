# AACSNet: Angle-Aware and Context-Sensitive Network for Oriented Object Detection

AACSNet 是一个面向旋转目标检测的深度学习模型，基于 OrientedFormer 架构，通过 **AAFAM（角度感知特征对齐模块）** 和 **CLSM（跨层级协同模块）** 增强航空影像中的目标检测性能。

## 核心创新

- **AAFAM (Angle-Aware Feature Alignment Module)**：处理旋转目标的角度周期性与不确定性，支持周期性编码、旋转采样、多角度融合和角度空间注意力
- **CLSM (Cross-Level Synergy Module)**：跨尺度上下文融合，通过层级上下文驱动与邻域协同聚合增强多尺度特征

## 环境依赖

- [MMRotate](https://github.com/open-mmlab/mmrotate) 1.x 或兼容框架（含 mmrotate、mmdet、mmengine）
- PyTorch >= 1.8

## 安装与目录放置

1. 安装 MMRotate 或 OrientedFormer 及其依赖
2. 将本 `AACSNet` 文件夹放入框架的 `projects/` 目录下，整体结构为：

   ```
   框架根目录/
   ├── projects/
   │   └── AACSNet/          
   │       ├── configs/
   │       ├── orientedformer/
   │       └── tools/
   ├── mmrotate/
   └── ...
   ```

## AACSNet 目录结构

```
AACSNet/
├── configs/                
│   ├── ours_le90_r50_hrsc.py      
│   ├── ours_le90_r50_dotav1.0.py  
│   └── ours_le90_r50_dior.py      
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

## 数据准备

请参考 [MMRotate 数据准备](https://github.com/open-mmlab/mmrotate/tree/main/tools/data) 处理 HRSC2016、DOTA、DIOR-R 等数据集。

在对应配置文件中修改 `data_root` 为你的数据路径。

## 训练

在 **框架根目录** 下执行（确保 `projects`、`mmrotate` 位于当前目录）：

```bash
# HRSC2016（36 epochs，2 GPU）
PYTHONPATH=$PWD:$PYTHONPATH torchrun --nproc_per_node=2 projects/AACSNet/tools/train.py projects/AACSNet/configs/ours_le90_r50_hrsc.py --launcher pytorch

# DOTA v1.0（12 epochs，2 GPU）
PYTHONPATH=$PWD:$PYTHONPATH torchrun --nproc_per_node=2 projects/AACSNet/tools/train.py projects/AACSNet/configs/ours_le90_r50_dotav1.0.py --launcher pytorch

# DIOR-R（12 epochs，2 GPU）
PYTHONPATH=$PWD:$PYTHONPATH torchrun --nproc_per_node=2 projects/AACSNet/tools/train.py projects/AACSNet/configs/ours_le90_r50_dior.py --launcher pytorch
```

## 测试

```bash
# HRSC2016
PYTHONPATH=$PWD:$PYTHONPATH torchrun --nproc_per_node=2 projects/AACSNet/tools/test.py projects/AACSNet/configs/ours_le90_r50_hrsc.py work_dirs/ours_le90_r50_hrsc/epoch_36.pth --launcher pytorch

# DOTA v1.0
PYTHONPATH=$PWD:$PYTHONPATH torchrun --nproc_per_node=2 projects/AACSNet/tools/test.py projects/AACSNet/configs/ours_le90_r50_dotav1.0.py work_dirs/ours_le90_r50_dotav1.0/epoch_12.pth --launcher pytorch

# DIOR-R
PYTHONPATH=$PWD:$PYTHONPATH torchrun --nproc_per_node=2 projects/AACSNet/tools/test.py projects/AACSNet/configs/ours_le90_r50_dior.py work_dirs/ours_le90_r50_dior/epoch_12.pth --launcher pytorch
```


