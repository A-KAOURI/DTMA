# DTMA

This repository contains the code for the training and evalution of the work: "Depth-Aware Multi-Modal Learning for Optical Flow Estimation from Event Cameras"

## 1. Environment setup

```bash
conda create -n DTMA python=3.10
conda activate DTMA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Install ffmpeg and add it to `PATH`, then run once in an interactive Python shell:

```python
import imageio
imageio.plugins.freeimage.download()
```

## 2. Dataset: download and generation

Download the full [DSEC dataset](https://dsec.ifi.uzh.ch/dsec-datasets/download/), including
`train_events`, `train_optical_flow`, `train_calibration`, `test_events`,
`test_forward_optical_flow_timestamps`, and `train/test_images`.

```bash
cd data_utils
python gen_dsec.py --dsec <path_to_DSEC> --split [train]/test/all -w -f
```

### Metric3D depth (the model's input feature)

Option A -- generate it locally once, running Metric3D natively in PyTorch (Needs only a [Metric3D](https://github.com/YvanYin/Metric3D) checkout + a ViT-Large checkpoint), then train/evaluate with `--depth_source precomputed`:

```bash
python generate_metric3d_depth.py --data_root datasets/dsec_full/trainval --metric3d_root <path_to_metric3d_checkout> --checkpoint <path_to_metric3d.pth>
python generate_metric3d_depth.py --data_root datasets/dsec_full/test --metric3d_root <path_to_metric3d_checkout> --checkpoint <path_to_metric3d.pth>
```

Option B -- compute it on the fly with `--depth_source online --onnx_path <path_to_metric3d.onnx>`,
which requires an existing Metric3D ONNX export instead.

**Recommended**: use `--depth_source online` only for inference (`test.py`); train with
`--depth_source precomputed` (Option A).

## 3. Training

```bash
python train.py --checkpoint_dir "./ckpts/<label>" --num_steps 200000 --lr 2e-4 --wandb
```

Key flags:
`--depth_source` (`precomputed` [default, recommended] or `online`), `--onnx_path
<path_to_metric3d.onnx>` (required if `--depth_source online`), `--validate`, `--batch_size`,
`--num_workers`, `--lr`, `--iters`.

## 4. Evaluation / submission

```bash
python test.py -c <checkpoint.pth> -s <save_path> -v --depth_source precomputed
```

## Acknowledgments

This work builds on the codebase of **TMA**, the original event-based optical flow model and
DSEC data pipeline. We thank the TMA authors for making their implementation publicly
available, which served as the foundation for this repository.