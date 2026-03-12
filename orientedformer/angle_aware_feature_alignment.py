"""
Angle-Aware Feature Alignment Module (AAFAM)

This module addresses key challenges in rotated object detection:
1. Angle periodicity: Handles the circular nature of angles (0° = 360°)
2. Angle-guided feature sampling: Adaptively samples features based on rotation
3. Multi-angle feature fusion: Aggregates features from multiple angle hypotheses
4. Angle-spatial joint attention: Enhances features with angle-aware spatial attention

Key Innovations:
- Periodic angle encoding using sin/cos to handle angle wrap-around
- Rotation-aware sampling grid generation
- Angular uncertainty estimation and multi-hypothesis fusion
- Joint angle-spatial attention mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math

from mmrotate.registry import MODELS
from mmcv.ops import DeformConv2d


@MODELS.register_module()
class PeriodicAngleEncoder(nn.Module):
    """
    Encode angles to handle periodicity (0° = 360°).
    Uses sin/cos encoding to map angles to a continuous periodic space.
    """
    def __init__(self, 
                 embed_dim: int = 64,
                 angle_version: str = 'le90'):
        super().__init__()
        self.embed_dim = embed_dim
        self.angle_version = angle_version
        
        # Learnable frequency multipliers for different angle representations
        self.freq_multipliers = nn.Parameter(torch.ones(embed_dim // 2))
        
        # Projection layers
        self.angle_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        """
        Args:
            angles: (..., 1) in radians, range depends on angle_version
        Returns:
            angle_embed: (..., embed_dim) periodic angle encoding
        """
        # Normalize angles based on version
        if self.angle_version == 'le90':
            # Range: [-π/2, π/2], normalize to [-1, 1]
            angles_norm = angles / (math.pi / 2)
        elif self.angle_version == 'oc':
            # Range: [-π, π], normalize to [-1, 1]
            angles_norm = angles / math.pi
        else:
            angles_norm = angles
        
        # Generate base frequencies
        device = angles.device
        dim_t = torch.arange(self.embed_dim // 2, dtype=torch.float32, device=device)
        dim_t = (dim_t / (self.embed_dim // 2)) * math.pi  # [0, π]
        
        # Apply learnable frequency multipliers
        freqs = self.freq_multipliers * dim_t.unsqueeze(0)  # (1, embed_dim//2)
        
        # Compute sin/cos encoding with angle input
        angles_expanded = angles_norm.unsqueeze(-1) * freqs  # (..., 1, embed_dim//2)
        sin_enc = torch.sin(angles_expanded)
        cos_enc = torch.cos(angles_expanded)
        
        # Concatenate sin and cos
        angle_embed = torch.cat([sin_enc, cos_enc], dim=-1)  # (..., embed_dim)
        
        # Project and normalize
        angle_embed = self.angle_proj(angle_embed)
        
        return angle_embed


@MODELS.register_module()
class RotationAwareSampler(nn.Module):
    """
    Generate rotation-aware sampling grids based on predicted angles.
    Samples features along the rotated object's axes.
    """
    def __init__(self,
                 kernel_size: int = 3,
                 num_samples: int = 9,
                 dilation: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_samples = num_samples
        self.dilation = dilation
        
        # Generate base sampling offsets (relative to center)
        base_offsets = self._generate_base_offsets()
        self.register_buffer('base_offsets', base_offsets)
        
    def _generate_base_offsets(self) -> torch.Tensor:
        """Generate base sampling offsets in a grid."""
        offsets = []
        kh = kw = int(math.sqrt(self.num_samples))
        for i in range(kh):
            for j in range(kw):
                y = (i - kh // 2) / (kh // 2) * 0.5
                x = (j - kw // 2) / (kw // 2) * 0.5
                offsets.append([x, y])
        return torch.tensor(offsets, dtype=torch.float32)  # (num_samples, 2)
    
    def forward(self, 
                angles: torch.Tensor,
                centers: torch.Tensor) -> torch.Tensor:
        """
        Args:
            angles: (B, N, 1) rotation angles in radians
            centers: (B, N, 2) center coordinates (y, x)
        Returns:
            sampling_points: (B, N, num_samples, 2) rotated sampling coordinates
        """
        B, N, _ = angles.shape
        
        # Expand base offsets
        base_offsets = self.base_offsets.unsqueeze(0).unsqueeze(0)  # (1, 1, num_samples, 2)
        base_offsets = base_offsets.expand(B, N, -1, -1)  # (B, N, num_samples, 2)
        
        # Apply rotation transformation
        cos_a = torch.cos(angles)  # (B, N, 1)
        sin_a = torch.sin(angles)  # (B, N, 1)
        
        # Rotation matrix: [[cos, -sin], [sin, cos]]
        # Build rotation matrices for each position: (B, N, 2, 2)
        rot_matrix = torch.stack([
            torch.stack([cos_a, -sin_a], dim=-1),  # (B, N, 1, 2)
            torch.stack([sin_a, cos_a], dim=-1)   # (B, N, 1, 2)
        ], dim=-2)  # (B, N, 2, 2)
        
        # Apply rotation to offsets
        # base_offsets: (B, N, num_samples, 2)
        # rot_matrix: (B, N, 2, 2)
        # We want: for each (B, N), rotate each of num_samples points
        # Reshape for batch matrix multiplication
        base_offsets_flat = base_offsets.view(B * N, self.num_samples, 2)  # (B*N, num_samples, 2)
        rot_matrix_flat = rot_matrix.view(B * N, 2, 2)  # (B*N, 2, 2)
        
        # Apply rotation: (B*N, num_samples, 2) x (B*N, 2, 2) -> (B*N, num_samples, 2)
        rotated_offsets_flat = torch.bmm(base_offsets_flat, rot_matrix_flat.transpose(1, 2))  # (B*N, num_samples, 2)
        rotated_offsets = rotated_offsets_flat.view(B, N, self.num_samples, 2)  # (B, N, num_samples, 2)
        
        # Add center coordinates
        sampling_points = centers.unsqueeze(2) + rotated_offsets  # (B, N, num_samples, 2)
        
        return sampling_points


@MODELS.register_module()
class AngleAwareFeatureAlignment(nn.Module):
    """
    Main Angle-Aware Feature Alignment Module.
    Aligns and enhances features based on predicted rotation angles.
    """
    def __init__(self,
                 in_channels: int = 256,
                 embed_dim: int = 256,
                 num_angle_hypotheses: int = 3,
                 angle_version: str = 'le90',
                 kernel_size: int = 3,
                 num_samples: int = 9,
                 enable_periodic_encoding: bool = True,
                 enable_rotation_sampling: bool = True,
                 enable_multi_angle_fusion: bool = True,
                 enable_angle_spatial_attention: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_angle_hypotheses = num_angle_hypotheses
        self.angle_version = angle_version
        self.enable_periodic_encoding = enable_periodic_encoding
        self.enable_rotation_sampling = enable_rotation_sampling
        self.enable_multi_angle_fusion = enable_multi_angle_fusion
        self.enable_angle_spatial_attention = enable_angle_spatial_attention
        
        # Angle encoding
        self.angle_embed_dim = embed_dim // 4
        if self.enable_periodic_encoding:
            self.angle_encoder = PeriodicAngleEncoder(
                embed_dim=self.angle_embed_dim,
                angle_version=angle_version
            )
        else:
            self.angle_encoder = None
        
        # Rotation-aware sampler
        if self.enable_rotation_sampling:
            self.rotation_sampler = RotationAwareSampler(
                kernel_size=kernel_size,
                num_samples=num_samples
            )
        else:
            self.rotation_sampler = None
        
        # Angle prediction head (optional, can use external angle predictions)
        self.angle_predictor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1)
        )
        
        # Deformable convolution for angle-guided sampling
        # Use DeformConv2d (not Pack) to support custom offsets
        self.deform_conv = DeformConv2d(
            in_channels,
            embed_dim,
            kernel_size=3,
            padding=1,
            stride=1
        )
        
        # Multi-angle feature fusion
        if self.enable_multi_angle_fusion and num_angle_hypotheses > 1:
            self.angle_attention = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=8,
                batch_first=True
            )
        else:
            self.angle_attention = None
        
        # Angle-spatial joint attention
        if self.enable_angle_spatial_attention:
            self.joint_attention = nn.Sequential(
                nn.Conv2d(embed_dim + self.angle_embed_dim, embed_dim, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dim, embed_dim, 1)
            )
        else:
            self.joint_attention = None
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.GroupNorm(32, embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, 1)
        )
        
    def forward(self,
                features: torch.Tensor,
                angles: Optional[torch.Tensor] = None,
                centers: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            features: (B, C, H, W) input feature maps
            angles: (B, H*W, 1) optional angle predictions in radians.
                    If None, will predict from features
            centers: (B, H*W, 2) optional center coordinates (y, x).
                     If None, will use grid centers
        Returns:
            enhanced_features: (B, embed_dim, H, W) angle-aware enhanced features
        """
        B, C, H, W = features.shape
        
        # Predict angles if not provided
        if angles is None:
            angle_logits = self.angle_predictor(features)  # (B, 1, H, W)
            angles = angle_logits.view(B, H * W, 1)
        
        # Generate centers if not provided
        if centers is None:
            y_coords = torch.arange(H, dtype=torch.float32, device=features.device)
            x_coords = torch.arange(W, dtype=torch.float32, device=features.device)
            # Use compatible meshgrid for older PyTorch versions
            try:
                y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
            except TypeError:
                y_grid, x_grid = torch.meshgrid(y_coords, x_coords)
            centers = torch.stack([y_grid.flatten(), x_grid.flatten()], dim=-1)  # (H*W, 2)
            centers = centers.unsqueeze(0).expand(B, -1, -1)  # (B, H*W, 2)
        
        if self.enable_periodic_encoding:
            angle_embed = self.angle_encoder(angles)  # (B, H*W, embed_dim//4)
        else:
            angle_embed = features.new_zeros(B, H * W, self.angle_embed_dim)
        
        if self.enable_rotation_sampling:
            # Generate rotation-aware sampling points (kept for potential diagnostics)
            _ = self.rotation_sampler(angles, centers)
            offsets = self._generate_deform_offsets(angles, features.shape[-2:])
        else:
            offsets = torch.zeros(
                (B, 18, H, W), dtype=features.dtype, device=features.device)
        
        aligned_features = self.deform_conv(features, offsets)  # (B, embed_dim, H, W)
        
        if (self.enable_multi_angle_fusion and self.num_angle_hypotheses > 1
                and self.angle_attention is not None and self.enable_rotation_sampling):
            angle_candidates = self._generate_angle_hypotheses(angles)
            multi_angle_features = []
            for angle_cand in angle_candidates:
                offset_cand = self._generate_deform_offsets(angle_cand, features.shape[-2:])
                feat_cand = self.deform_conv(features, offset_cand)
                multi_angle_features.append(feat_cand)
            
            # Lightweight fusion: stack and use simple attention per spatial location
            # Instead of flattening all spatial locations together, process each position independently
            feat_stack = torch.stack(multi_angle_features, dim=2)  # (B, C, num_hypo, H, W)
            feat_stack = feat_stack.permute(0, 2, 1, 3, 4)  # (B, num_hypo, C, H, W)
            
            # Use query from original angle
            query = aligned_features.unsqueeze(1)  # (B, 1, C, H, W)
            
            # Reshape for per-position attention: (B*H*W, 1, C) and (B*H*W, num_hypo, C)
            # Note: B, H, W already defined at function start
            _, C_fused, H_fused, W_fused = aligned_features.shape
            num_hypo = self.num_angle_hypotheses
            
            query_flat = query.permute(0, 3, 4, 1, 2).reshape(B * H_fused * W_fused, 1, C_fused)  # (B*H*W, 1, C)
            key_value_flat = feat_stack.permute(0, 3, 4, 1, 2).reshape(B * H_fused * W_fused, num_hypo, C_fused)  # (B*H*W, num_hypo, C)
            
            # Apply attention per spatial location (much more memory efficient)
            fused_flat, _ = self.angle_attention(query_flat, key_value_flat, key_value_flat)  # (B*H*W, 1, C)
            
            # Reshape back: (B, H, W, 1, C) -> (B, C, H, W)
            fused_features = fused_flat.reshape(B, H_fused, W_fused, 1, C_fused).squeeze(3).permute(0, 3, 1, 2)  # (B, C, H, W)
        else:
            fused_features = aligned_features
        
        if self.enable_angle_spatial_attention and self.joint_attention is not None:
            angle_embed_spatial = angle_embed.view(B, H, W, -1).permute(0, 3, 1, 2)  # (B, embed_dim//4, H, W)
            joint_input = torch.cat([fused_features, angle_embed_spatial], dim=1)  # (B, embed_dim+embed_dim//4, H, W)
            joint_attn = torch.sigmoid(self.joint_attention(joint_input))  # (B, embed_dim, H, W)
            enhanced_features = fused_features * joint_attn + fused_features
        else:
            enhanced_features = fused_features
        enhanced_features = self.output_proj(enhanced_features)
        
        return enhanced_features
    
    def _generate_deform_offsets(self, 
                                 angles: torch.Tensor,
                                 spatial_shape: Tuple[int, int]) -> torch.Tensor:
        """Generate deformable convolution offsets from angles."""
        B, N, _ = angles.shape
        H, W = spatial_shape
        
        # Generate base offsets for 3x3 kernel
        base_offsets = torch.tensor([
            [-1, -1], [0, -1], [1, -1],
            [-1,  0], [0,  0], [1,  0],
            [-1,  1], [0,  1], [1,  1]
        ], dtype=torch.float32, device=angles.device)  # (9, 2)
        
        base_offsets = base_offsets.unsqueeze(0).unsqueeze(0)  # (1, 1, 9, 2)
        base_offsets = base_offsets.expand(B, N, -1, -1)
        
        # Apply rotation
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)
        
        rot_matrix = torch.stack([
            torch.stack([cos_a, -sin_a], dim=-1),
            torch.stack([sin_a, cos_a], dim=-1)
        ], dim=-2)  # (B, N, 2, 2)
        
        # Apply rotation using bmm (same fix as in RotationAwareSampler)
        num_samples = 9  # 3x3 kernel has 9 sampling points
        base_offsets_flat = base_offsets.view(B * N, num_samples, 2)  # (B*N, num_samples, 2)
        rot_matrix_flat = rot_matrix.view(B * N, 2, 2)  # (B*N, 2, 2)
        
        rotated_offsets_flat = torch.bmm(base_offsets_flat, rot_matrix_flat.transpose(1, 2))  # (B*N, num_samples, 2)
        rotated_offsets = rotated_offsets_flat.view(B, N, num_samples, 2)  # (B, N, num_samples, 2)
        
        # Reshape to deformable conv format: (B, 18, H, W) where 18 = 2 * 9
        offsets = rotated_offsets.view(B, H, W, 9, 2)
        offsets = offsets.permute(0, 3, 4, 1, 2)  # (B, 9, 2, H, W)
        offsets = offsets.contiguous().view(B, 18, H, W)
        
        return offsets
    
    def _generate_angle_hypotheses(self, 
                                   base_angles: torch.Tensor) -> list:
        """Generate multiple angle hypotheses around the base angle."""
        hypotheses = []
        angle_step = math.pi / (2 * self.num_angle_hypotheses)
        
        for i in range(self.num_angle_hypotheses):
            offset = (i - self.num_angle_hypotheses // 2) * angle_step
            hyp_angle = base_angles + offset
            hypotheses.append(hyp_angle)
        
        
        return hypotheses

