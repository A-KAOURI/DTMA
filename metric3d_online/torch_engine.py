import sys
import types

import numpy as np

from .intrinsics import DSEC_FX, DSEC_FY
from .postprocess import postprocess, gray_to_colormap
from .preprocess import preprocess


def _stub_mmcv():
    """mono/utils/comm.py does `from mmcv.utils import collect_env`, but collect_env is never
    actually called on the inference path -- mmcv is otherwise unused (Config loading already
    falls back to mmengine.Config when mmcv isn't importable). Real mmcv's setup.py fails to
    build on newer Python, so stub the one symbol instead of requiring the real package."""
    if 'mmcv' in sys.modules:
        return
    mmcv = types.ModuleType('mmcv')
    mmcv_utils = types.ModuleType('mmcv.utils')
    mmcv_utils.collect_env = lambda: {}
    mmcv.utils = mmcv_utils
    sys.modules['mmcv'] = mmcv
    sys.modules['mmcv.utils'] = mmcv_utils


class TorchDepthEngine:
    """Runs Metric3D (ViT-Large) natively in PyTorch straight from a metric3d checkout +
    checkpoint -- no ONNX export needed. Slower to build (loads the full model, ~5s) than
    OnlineDepthEngine's ONNX session, but skips the export step entirely; meant for one-off
    local/batch depth generation (see generate_metric3d_depth.py), not per-step online use
    during training (see OnlineDepthEngine for that).

    Reproduces metric3d/mono/utils/do_test.py's pre/post-processing exactly, matching
    OnlineDepthEngine's output.
    """

    def __init__(self, metric3d_root: str, checkpoint: str, fx: float = DSEC_FX, fy: float = DSEC_FY,
                 device: str = 'cuda'):
        _stub_mmcv()
        if metric3d_root not in sys.path:
            sys.path.insert(0, metric3d_root)

        import torch
        from mmengine import Config
        from mono.model.monodepth_model import get_configured_monodepth_model
        from mono.utils.running import load_ckpt

        cfg = Config.fromfile(f'{metric3d_root}/mono/configs/HourglassDecoder/vit.raft5.large.py')
        model = get_configured_monodepth_model(cfg)
        model, _, _, _ = load_ckpt(checkpoint, model, strict_match=False)
        self.model = model.to(device).eval()

        self._torch = torch
        self.device = device
        self.fx = fx
        self.fy = fy

    def infer(self, rgb: np.ndarray):
        """rgb: HxWx3 uint8, RGB order.
        Returns (colorized_depth_uint8_HWC, raw_metric_depth_float32_HW)."""
        input_tensor, pad, label_scale_factor, ori_shape = preprocess(rgb, self.fx, self.fy)
        x = self._torch.from_numpy(input_tensor).to(self.device)
        with self._torch.no_grad():
            out = self.model.depth_model(x)
        raw_depth = out['prediction'][0, 0].cpu().numpy()
        metric_depth = postprocess(raw_depth, pad, label_scale_factor, ori_shape)
        colorized = gray_to_colormap(metric_depth)
        return colorized, metric_depth
