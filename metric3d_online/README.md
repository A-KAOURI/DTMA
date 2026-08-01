# metric3d_online

Two ways to get Metric3D depth into this repo, sharing the same pre/post-processing
(`preprocess.py`, `postprocess.py`, matching `mono/utils/do_test.py` exactly) so their output
matches regardless of which one is used. See the root [README.md](../README.md) for how each
fits into the overall pipeline.

This works because Metric3D's `cam_model` input (built from the camera intrinsics) is accepted
by `RAFTDepthNormalDPT5.forward` but never actually read in the model body -- all
intrinsics-awareness happens in `mono/utils/do_test.py`'s pre/post-processing arithmetic, not
inside the network graph. The DSEC-specific focal rescaling and rainbow colorization are
reimplemented here in plain numpy/opencv.

## `TorchDepthEngine` -- local/offline generation, native PyTorch

Used by `generate_metric3d_depth.py` (Option A in the root README). Runs the actual Metric3D
(ViT-Large) PyTorch model straight from a [Metric3D](https://github.com/YvanYin/Metric3D)
checkout + `.pth` checkpoint -- no ONNX export needed. Requires `mmengine` and `timm` (see
`requirements.txt`); deliberately does *not* require `mmcv` -- the only thing Metric3D's own
code imports from it on the inference path is one dead `collect_env` import
(`mono/utils/comm.py`, never actually called), which `torch_engine.py` stubs out instead of
requiring a real `mmcv` install (its packaging doesn't build cleanly on newer Python anyway).
xFormers is optional in Metric3D's own code (falls back to plain attention) and isn't needed
either.

## `OnlineDepthEngine` -- on-the-fly, ONNX

Used by `dataloader/dsec_full.py`'s `depth_source="online"` path (`--depth_source online
--onnx_path <...>`) and `generate_metric3d_depth.py`'s Option B. Assumes you already have a
Metric3D model exported to ONNX -- this module doesn't do that conversion. Loads the given
`.onnx` file via `onnxruntime-gpu`, preferring the TensorRT execution provider (FP16, engine
cached under `weights/trt_cache/` after the first run so the ~minute-long TensorRT build only
happens once) and falling back to plain CUDA (then CPU) if TensorRT isn't installed -- the
pipeline still runs end-to-end either way, just without the FP16 speedup. See
`tests/test_online_depth_engine.py` for usage.

## Known gap on this machine: onnxruntime-gpu needs CUDA 13 / cuDNN 9 runtime DLLs

`OnnxRuntime.InferenceSession` is built with the provider list
`["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]` and falls back
down the list automatically if a provider fails to load -- so the pipeline always produces
correct output (verified end-to-end by `tests/test_online_depth_engine.py`), but on this specific
machine (Python 3.14 `TMA` env, CUDA 12.6 driver/toolkit), the `onnxruntime-gpu` wheels available
for that Python version (1.24.1+) are built against CUDA 13.x/cuDNN 9.x and fail to load both the
TensorRT and CUDA execution providers (`cublas64_13.dll`/`cublasLt64_13.dll` missing), silently
falling all the way back to plain CPU execution. This is a real, unresolved gap, not a design
choice: to get actual GPU/TensorRT-accelerated online depth, install a matching CUDA 13 toolkit +
cuDNN 9 (see [ONNX Runtime's CUDA execution provider requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)),
or use a `TMA` env Python version with `onnxruntime-gpu` wheels built against the CUDA 12.x
runtime already on this machine, if/when NVIDIA publishes one for that Python version.

## `num_workers` caveat

Each `DataLoader` worker process needs its own CUDA context + ONNX Runtime session, so
`OnlineDepthEngine` is built lazily per-process (first access after a worker forks). If you
combine `--depth_source online` with training (`num_workers > 0`), expect one engine instance
(and its GPU memory) per worker -- keep `--num_workers` modest, or prefer online mode for
single-process evaluation/inference where the storage savings matter more than throughput.
