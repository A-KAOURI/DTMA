"""
Batch-generate `depth_color/` (Metric3D colorized depth) PNGs for every frame under
datasets/dsec_full/{trainval,test}/<sequence>/images/, using native PyTorch Metric3D inference
(metric3d_online.TorchDepthEngine) against a Metric3D checkout + checkpoint -- no ONNX export
needed.

Run this once per split after gen_dsec.py, so training/evaluation can read
--depth_source precomputed. --depth_source online (metric3d_online.OnlineDepthEngine, ONNX) is
for one-off on-the-fly inference on sequences you haven't precomputed, and requires a prior
ONNX export instead of the native checkpoint used here.
"""
import argparse
from pathlib import Path

import imageio.v2 as imageio
from tqdm import tqdm


def generate_depth_color(data_root: Path, metric3d_root: str, checkpoint: str, overwrite: bool = False):
    from metric3d_online import TorchDepthEngine

    engine = TorchDepthEngine(metric3d_root, checkpoint)

    seq_dirs = sorted(p for p in data_root.iterdir() if p.is_dir())
    for seq_dir in seq_dirs:
        images_dir = seq_dir / "images"
        if not images_dir.is_dir():
            continue

        depth_dir = seq_dir / "depth_color"
        depth_dir.mkdir(exist_ok=True)

        image_paths = sorted(images_dir.glob("*.png"))
        for image_path in tqdm(image_paths, ncols=60, desc=seq_dir.name):
            out_path = depth_dir / image_path.name
            if out_path.exists() and not overwrite:
                continue

            img = imageio.imread(image_path)
            colorized, _ = engine.infer(img)
            imageio.imwrite(out_path, colorized)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", type=str, required=True,
                         help="Folder containing the per-sequence dataset, e.g. datasets/dsec_full/trainval")
    parser.add_argument("--metric3d_root", type=str, required=True,
                         help="Path to a Metric3D checkout (the `mono` package)")
    parser.add_argument("--checkpoint", type=str, required=True,
                         help="Path to a Metric3D ViT-Large checkpoint (.pth)")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate PNGs that already exist")
    args = parser.parse_args()

    generate_depth_color(Path(args.data_root), args.metric3d_root, args.checkpoint, args.overwrite)
