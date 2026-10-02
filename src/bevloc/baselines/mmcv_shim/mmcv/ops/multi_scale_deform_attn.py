"""``mmcv.ops.multi_scale_deform_attn`` without the compiled kernel.

``multi_scale_deformable_attn_pytorch`` is mmcv's own reference implementation (the path mmcv takes on
CPU, and the one its unit tests hold the CUDA kernel to); ``MultiScaleDeformableAttention`` is the
Deformable-DETR module with mmcv's parameter names (``sampling_offsets``, ``attention_weights``,
``value_proj``, ``output_proj``) and the same forward, always taking the PyTorch path.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def multi_scale_deformable_attn_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    """value (bs, num_keys, heads, dim); value_spatial_shapes (levels, 2) as (H, W);
    sampling_locations (bs, num_queries, heads, levels, points, 2) in [0, 1] (x, y);
    attention_weights (bs, num_queries, heads, levels, points). Returns (bs, num_queries, heads * dim)."""
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    shapes = [(int(h), int(w)) for h, w in value_spatial_shapes]
    value_list = value.split([h * w for h, w in shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampled = []
    for level, (h, w) in enumerate(shapes):
        value_l = value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, h, w)
        grid_l = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)   # (bs*heads, queries, points, 2)
        sampled.append(F.grid_sample(value_l, grid_l, mode="bilinear", padding_mode="zeros", align_corners=False))
    attention_weights = attention_weights.transpose(1, 2).reshape(bs * num_heads, 1, num_queries, num_levels * num_points)
    output = (torch.stack(sampled, dim=-2).flatten(-2) * attention_weights).sum(-1).view(bs, num_heads * embed_dims, num_queries)
    return output.transpose(1, 2).contiguous()


class MultiScaleDeformableAttention(nn.Module):
    def __init__(self, embed_dims=256, num_heads=8, num_levels=4, num_points=4, im2col_step=64, dropout=0.1,
                 batch_first=False, norm_cfg=None, init_cfg=None, value_proj_ratio=1.0):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims must be divisible by num_heads, got {embed_dims} and {num_heads}")
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.im2col_step = im2col_step
        self.embed_dims, self.num_levels, self.num_heads, self.num_points = embed_dims, num_levels, num_heads, num_points
        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        value_proj_size = int(embed_dims * value_proj_ratio)
        self.value_proj = nn.Linear(embed_dims, value_proj_size)
        self.output_proj = nn.Linear(value_proj_size, embed_dims)
        self.init_weights()

    def init_weights(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid = (grid / grid.abs().max(-1, keepdim=True)[0]).view(self.num_heads, 1, 1, 2)
        grid = grid.repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid.view(-1))
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(self, query, key=None, value=None, identity=None, query_pos=None, key_padding_mask=None,
                reference_points=None, spatial_shapes=None, level_start_index=None, **kwargs):
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        if int((spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum()) != num_value:
            raise ValueError("spatial_shapes do not cover the value tokens")
        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1).view(bs, num_query, self.num_heads, self.num_levels, self.num_points)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(f"last dim of reference_points must be 2 or 4, got {reference_points.shape[-1]}")
        output = multi_scale_deformable_attn_pytorch(value, spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)
        if not self.batch_first:
            output = output.permute(1, 0, 2)
        return self.dropout(output) + identity
