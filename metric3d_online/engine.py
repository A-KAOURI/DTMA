import os
from pathlib import Path

import numpy as np

from .intrinsics import DSEC_FX, DSEC_FY
from .preprocess import preprocess
from .postprocess import postprocess, gray_to_colormap

DEFAULT_TRT_CACHE_DIR = Path(__file__).resolve().parent.parent / "weights" / "trt_cache"


class OnlineDepthEngine:
    """Runs an already-exported Metric3D ViT-Large ONNX model through ONNX Runtime, preferring
    the TensorRT execution provider (FP16, engine cached to disk after the first run) and
    falling back to plain CUDA (then CPU) if TensorRT isn't installed.

    Reproduces metric3d/mono/utils/do_test.py's pre/post-processing exactly, so this is a
    drop-in replacement for reading a precomputed `depth_color` PNG from disk.
    """

    def __init__(self, onnx_path, fx: float = DSEC_FX, fy: float = DSEC_FY,
                 trt_cache_dir=None, fp16: bool = True):
        import onnxruntime as ort

        onnx_path = str(onnx_path)
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(f"No Metric3D ONNX model found at '{onnx_path}'.")

        trt_cache_dir = str(trt_cache_dir or DEFAULT_TRT_CACHE_DIR)
        os.makedirs(trt_cache_dir, exist_ok=True)

        providers = [
            ("TensorrtExecutionProvider", {
                "trt_fp16_enable": fp16,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": trt_cache_dir,
            }),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.fx = fx
        self.fy = fy

    def infer(self, rgb: np.ndarray):
        """rgb: HxWx3 uint8, RGB order.
        Returns (colorized_depth_uint8_HWC, raw_metric_depth_float32_HW)."""
        input_tensor, pad, label_scale_factor, ori_shape = preprocess(rgb, self.fx, self.fy)
        raw_output = self.session.run(None, {self.input_name: input_tensor})[0]
        raw_depth = raw_output[0, 0]  # [1,1,H,W] -> [H,W]
        metric_depth = postprocess(raw_depth, pad, label_scale_factor, ori_shape)
        colorized = gray_to_colormap(metric_depth)
        return colorized, metric_depth
