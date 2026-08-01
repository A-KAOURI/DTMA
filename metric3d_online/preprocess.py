import cv2
import numpy as np

from .intrinsics import CANONICAL_FOCAL, CROP_SIZE, IMAGENET_MEAN, IMAGENET_STD, PAD_VALUE


def resize_for_input(rgb: np.ndarray, crop_size=CROP_SIZE):
    """Aspect-preserving resize + mean-pad to `crop_size`, matching
    mono.utils.do_test.resize_for_input exactly (minus the unused camera-model map)."""
    h, w, _ = rgb.shape
    to_scale_ratio = min(crop_size[0] / h, crop_size[1] / w)

    reshape_h = int(to_scale_ratio * h)
    reshape_w = int(to_scale_ratio * w)

    pad_h = max(crop_size[0] - reshape_h, 0)
    pad_w = max(crop_size[1] - reshape_w, 0)
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2

    resized = cv2.resize(rgb, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        resized, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
        cv2.BORDER_CONSTANT, value=PAD_VALUE,
    )

    pad = (pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half)
    label_scale_factor = 1.0 / to_scale_ratio
    return padded, pad, label_scale_factor


def preprocess(rgb: np.ndarray, fx: float, fy: float):
    """
    rgb: HxWx3 uint8, RGB channel order (e.g. as returned by imageio.imread on a DSEC image).

    Note: metric3d's own infer.py swaps channels with cv2.COLOR_BGR2RGB before calling
    transform_test_data_scalecano, which swaps again internally -- the two swaps cancel out,
    so a genuinely RGB-ordered input (as here) needs no swap at all to match its net effect.

    Returns (input_tensor[1,3,H,W] float32, pad, label_scale_factor, (ori_h, ori_w)), matching
    mono.utils.do_test.transform_test_data_scalecano exactly.
    """
    ori_h, ori_w = rgb.shape[:2]
    ori_focal = (fx + fy) / 2.0
    cano_label_scale_ratio = CANONICAL_FOCAL / ori_focal

    resized, pad, resize_label_scale_ratio = resize_for_input(rgb)
    label_scale_factor = cano_label_scale_ratio * resize_label_scale_ratio

    chw = resized.transpose(2, 0, 1).astype(np.float32)
    chw = (chw - IMAGENET_MEAN) / IMAGENET_STD
    input_tensor = chw[None].astype(np.float32)

    return input_tensor, pad, label_scale_factor, (ori_h, ori_w)
