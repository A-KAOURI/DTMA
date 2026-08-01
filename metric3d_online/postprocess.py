import cv2
import numpy as np
import matplotlib

from .intrinsics import DEPTH_RANGE


def unpad_and_resize(raw_depth: np.ndarray, pad, ori_shape):
    top, bottom, left, right = pad
    h, w = raw_depth.shape
    cropped = raw_depth[top:h - bottom, left:w - right]
    return cv2.resize(cropped, dsize=(ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_LINEAR)


def postprocess(raw_depth: np.ndarray, pad, label_scale_factor: float, ori_shape) -> np.ndarray:
    """Converts the raw canonical-space network output into metric depth (meters), matching
    mono.utils.do_test.get_prediction's rescaling exactly."""
    depth = unpad_and_resize(raw_depth, pad, ori_shape)
    depth = depth * DEPTH_RANGE[1] / label_scale_factor
    depth[depth < 0] = 0
    return depth.astype(np.float32)


def gray_to_colormap(img: np.ndarray, cmap='rainbow') -> np.ndarray:
    """Percentile-normalized rainbow colorization of a metric depth map. Matches
    mono.utils.transform.gray_to_colormap exactly -- this is the function that produced the
    `depth_color` PNGs the DTMA dataloader was trained on, so online mode must reproduce it."""
    assert img.ndim == 2
    img = img.copy()
    img[img < 0] = 0
    mask_invalid = img < 1e-10

    max_value = np.percentile(img.flatten(), q=98)
    img = img / (max_value + 1e-8)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.1)
    cmap_m = matplotlib.colormaps.get_cmap(cmap)
    mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_m)
    colormap = (mappable.to_rgba(img)[:, :, :3] * 255).astype(np.uint8)
    colormap[mask_invalid] = 0
    return colormap
