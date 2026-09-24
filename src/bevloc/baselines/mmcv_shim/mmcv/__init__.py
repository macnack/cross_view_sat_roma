"""Minimal pure-PyTorch stand-in for the two mmcv modules ``third_party/FG2`` imports.

Why: the Eagle container runs torch 2.13 + CUDA 13, for which OpenMMLab publishes no mmcv wheel with
compiled ops, and building mmcv from source needs a CUDA toolkit inside the container. FG² needs only
``mmcv.cnn.bricks.transformer.FFN`` and ``mmcv.ops.multi_scale_deform_attn.MultiScaleDeformableAttention``;
both are reproduced here with mmcv's parameter names (the released checkpoints load strictly) and
mmcv's own pure-PyTorch reference forward. ``bevloc.baselines.fg2.ensure_fg2_on_path`` puts this
directory on ``sys.path`` only when the real mmcv is not importable. Ported from mmcv (Apache-2.0,
OpenMMLab); nothing in ``third_party/`` is modified.
"""
__version__ = "2.1.0+bevloc-shim"
