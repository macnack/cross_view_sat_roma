"""``mmcv.cnn.bricks.transformer.FFN`` with mmcv's module layout (``layers.0.0`` = first Linear,
``layers.1`` = last Linear), so ``grd_ffn*.layers.*`` keys of the FG² checkpoints load strictly."""
from __future__ import annotations

import torch.nn as nn


def _activation(act_cfg):
    kind = act_cfg.get("type", "ReLU")
    if kind == "ReLU":
        return nn.ReLU(inplace=act_cfg.get("inplace", False))
    if kind == "GELU":
        return nn.GELU()
    raise NotImplementedError(f"mmcv shim: activation {kind!r}")


class FFN(nn.Module):
    def __init__(self, embed_dims=256, feedforward_channels=1024, num_fcs=2, act_cfg=None, ffn_drop=0.0,
                 dropout_layer=None, add_identity=True, init_cfg=None, layer_scale_init_value=0.0):
        super().__init__()
        if num_fcs < 2:
            raise ValueError("num_fcs should be no less than 2")
        if dropout_layer is not None or layer_scale_init_value > 0:
            raise NotImplementedError("mmcv shim: dropout_layer / layer_scale are not used by FG²")
        act_cfg = act_cfg or dict(type="ReLU", inplace=True)
        self.embed_dims, self.feedforward_channels, self.num_fcs = embed_dims, feedforward_channels, num_fcs
        layers, in_channels = [], embed_dims
        for _ in range(num_fcs - 1):
            layers.append(nn.Sequential(nn.Linear(in_channels, feedforward_channels), _activation(act_cfg),
                                        nn.Dropout(ffn_drop)))
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)
        self.dropout_layer = nn.Identity()
        self.add_identity = add_identity

    def forward(self, x, identity=None):
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)
