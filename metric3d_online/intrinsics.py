"""Camera/canonical-space constants for the online Metric3D (ViT-Large) depth path.

DSEC_* are the DSEC left event camera intrinsics (camRect0, rectified), sourced from
metric3d/camera_paramters.yaml the same way metric3d/infer.py defines them.

CANONICAL_FOCAL / DEPTH_RANGE / CROP_SIZE mirror the `data_basic` block of
metric3d/mono/configs/HourglassDecoder/vit.raft5.large.py. Only fx/fy and these constants
are needed here: the model's `cam_model` input (built from cx/cy) is accepted but never
read by RAFTDepthNormalDPT5.forward, so it is dropped entirely from the online path (see
metric3d_online/README.md).
"""
import numpy as np

DSEC_FX = 583.3081203392971
DSEC_FY = 583.3081203392971
DSEC_CX = 336.83414459228516
DSEC_CY = 220.91131019592285

CANONICAL_FOCAL = 1000.0
DEPTH_RANGE = (0, 1)
CROP_SIZE = (616, 1064)  # (H, W), fixed network input resolution, must be a multiple of 28

IMAGENET_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(3, 1, 1)
PAD_VALUE = [123.675, 116.28, 103.53]
