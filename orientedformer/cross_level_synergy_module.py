"""
Cross-Level Synergy Module (CLSM)

该模块面向旋转目标检测中的多尺度特征协同问题，通过三个步骤实现稳健的集成创新：

1. **层级上下文驱动**：对各尺度特征进行全局池化，构建跨尺度上下文向量，经非线性变换得到每个
   尺度的自适应权重，以引导后续融合过程。
2. **邻域协同聚合**：将同一尺度及其上下邻域的投影特征对齐后聚合，配合逐层独立的可学习卷积核，
   形成兼顾局部细节与高层语义的融合结果。
3. **保守残差调制**：通过小比例残差系数和归一化层，将聚合特征与原始特征稳健地结合，严格控制
   激活幅值，避免梯度发散。

整个流程不依赖额外监督项，也不会引入高阶矩阵运算，易于在现有 FPN 结构中集成，同时具备明确的
理论依据：层级上下文类似于 SENet 的通道注意力，邻域协同呼应了多尺度金字塔的优点，而保守残差则
延续了 ResNet 控制梯度的实践。
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrotate.registry import MODELS


@MODELS.register_module()
class CrossLevelSynergyModule(nn.Module):
    """Cross-Level Synergy Module.

    Args:
        in_channels (int): 输入特征通道数。
        embed_channels (int): 跨尺度处理的内部通道数。
        num_scales (int): FPN 输出的尺度数量。
        context_channels (int): 上下文向量的隐藏维度。
        residual_ratio (float): 残差融合时聚合结果的占比。
    """

    def __init__(self,
                 in_channels: int = 256,
                 embed_channels: int = 256,
                 num_scales: int = 5,
                 context_channels: int = 128,
                 residual_ratio: float = 0.3) -> None:
        super().__init__()
        assert 0.0 <= residual_ratio <= 1.0, \
            f"residual_ratio must be in [0, 1], got {residual_ratio}"

        self.in_channels = in_channels
        self.embed_channels = embed_channels
        self.num_scales = num_scales
        self.residual_ratio = residual_ratio

        # 线性投影，统一各尺度的通道维度
        self.pre_projs = nn.ModuleList([
            nn.Conv2d(in_channels, embed_channels, kernel_size=1, bias=True)
            for _ in range(num_scales)
        ])

        # 上下文建模：拼接所有尺度的全局描述后生成尺度权重
        self.context_proj = nn.Sequential(
            nn.Linear(embed_channels * num_scales, context_channels),
            nn.ReLU(inplace=True),
            nn.Linear(context_channels, num_scales),
            nn.Sigmoid()
        )

        # 每个尺度对应一个可学习的融合卷积，包含深度可分和逐点卷积以保持稳定
        self.fusion_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_channels, embed_channels, kernel_size=3, padding=1,
                          groups=max(embed_channels // 4, 1), bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_channels, embed_channels, kernel_size=1, bias=True)
            )
            for _ in range(num_scales)
        ])

        self.output_proj = nn.ModuleList([
            nn.Conv2d(embed_channels, in_channels, kernel_size=1, bias=True)
            for _ in range(num_scales)
        ])

        # 归一化层控制输出幅值
        if in_channels % 32 == 0:
            self.post_norms = nn.ModuleList([
                nn.GroupNorm(32, in_channels) for _ in range(num_scales)
            ])
        else:
            self.post_norms = nn.ModuleList([
                nn.BatchNorm2d(in_channels) for _ in range(num_scales)
            ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """前向传播。"""
        assert len(features) == self.num_scales, \
            f"Expected {self.num_scales} feature levels, got {len(features)}"

        proj_feats = []
        pooled_tokens = []
        for idx, feat in enumerate(features):
            proj = self.pre_projs[idx](feat)
            proj = torch.nan_to_num(proj, nan=0.0, posinf=1e4, neginf=-1e4)
            proj_feats.append(proj)

            pooled = F.adaptive_avg_pool2d(proj, 1).flatten(1)
            pooled_tokens.append(pooled)

        global_token = torch.cat(pooled_tokens, dim=1)  # (B, embed_channels * num_scales)
        scale_weights = self.context_proj(global_token)  # (B, num_scales)
        scale_weights = scale_weights.unsqueeze(-1).unsqueeze(-1)

        enhanced_features: List[torch.Tensor] = []
        for idx in range(self.num_scales):
            ref_feat = proj_feats[idx]
            _, _, H, W = ref_feat.shape

            neighbor_feats = []
            for offset in (-1, 0, 1):
                neighbor_idx = min(max(idx + offset, 0), self.num_scales - 1)
                neighbor = proj_feats[neighbor_idx]
                if neighbor.shape[2:] != (H, W):
                    neighbor = F.interpolate(neighbor,
                                             size=(H, W),
                                             mode='bilinear',
                                             align_corners=False)
                neighbor_feats.append(neighbor)

            stacked = torch.stack(neighbor_feats, dim=0)
            aggregated = stacked.mean(dim=0)

            weight = scale_weights[:, idx:idx + 1]
            aggregated = aggregated * weight + ref_feat * (1.0 - weight)

            fused = self.fusion_blocks[idx](aggregated)
            fused = torch.nan_to_num(fused, nan=0.0, posinf=1e4, neginf=-1e4)

            fused = self.output_proj[idx](fused)

            blended = fused * self.residual_ratio + features[idx] * (1.0 - self.residual_ratio)
            blended = self.post_norms[idx](blended)
            enhanced_features.append(blended)

        return enhanced_features


