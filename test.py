import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import os
from pathlib import Path
import imageio
from tqdm import tqdm

from dataloader.dsec_full import DSECfull

import flow_vis
from utils.visualization import visualize_optical_flow


def visualize_flow_submission(save_dir: Path, flow: np.ndarray, file_index: int):
    # flow_u(u,v) = ((float)I(u,v,1)-2^15)/128.0;
    # flow_v(u,v) = ((float)I(u,v,2)-2^15)/128.0;
    # valid(u,v)  = (bool)I(u,v,3);
    # [-2**15/128, 2**15/128] = [-256, 256]
    #flow_map_16bit = np.rint(flow_map*128 + 2**15).astype(np.uint16)
    _, h,w = flow.shape
    flow_map = np.rint(flow*128 + 2**15)
    flow_map = flow_map.astype(np.uint16).transpose(1,2,0)
    flow_map = np.concatenate((flow_map, np.zeros((h,w,1), dtype=np.uint16)), axis=-1)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    file_name = '{:06d}.png'.format(file_index)

    imageio.imwrite(save_dir / file_name, flow_map, format='PNG-FI')


def visualize_flow_image(save_dir:Path, flow: np.ndarray, file_index: int, method = "new"):
    method = method.lower()
    assert method in ["old", "new"]

    save_dir.mkdir(parents=True, exist_ok=True)
    file_name = '{:06d}.png'.format(file_index)

    if method == "new":
        flow= flow.transpose(1, 2, 0)
        flow_img = flow_vis.flow_to_color(flow, convert_to_bgr = False)
        imageio.imwrite(save_dir / file_name, flow_img, format='PNG')
    else:
        visualize_optical_flow(flow, savepath = str(save_dir / file_name))

@torch.no_grad()
def generate_submission(model, save_path:str, visualize_flow = False, visualization_method = "new", depth_source = "precomputed", onnx_path = None):
    # model.eval()
    test_dataset = DSECfull('test', depth_source=depth_source, onnx_path=onnx_path)

    bar = tqdm(test_dataset,total=len(test_dataset), ncols=60)
    bar.set_description('Test')

    save_path = Path(save_path)
    if visualize_flow:
        vis_path = save_path / "visualization"
        save_path = save_path / "submission"

    for voxel1, voxel2, img1, img2, depth, submission_coords in bar:
        voxel1 = voxel1[None].cuda()
        voxel2 = voxel2[None].cuda()
        depth = depth[None].cuda()
        img1 = img1[None].cuda()

        flow_pred, *_ = model(voxel1, voxel2, depth, img1)#[1,2,H,W]
        # flow_pred = flow_pred[0].cpu()
        flow_pred = flow_pred[-1][0].cpu()
        sequence, file_index = submission_coords
        save_dir = save_path / sequence

        visualize_flow_submission(save_dir, flow_pred.numpy(), file_index)

        if visualize_flow:
            visualize_flow_image(vis_path / sequence, flow_pred.numpy(), file_index, visualization_method)


if __name__ == "__main__":
    from argparse import ArgumentParser
    from importlib import import_module
    from subprocess import run

    parser =ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str, help="Path to a saved checkpoint file (.pth)")
    parser.add_argument("-b", "--input_bins", type=int, default=15, help="Number of input bins")
    parser.add_argument("-s", "--save_path", type=str, default="./sbumission", help="Submission save path")
    parser.add_argument("-v", "--visualize", action="store_true", help="Visualize optical flow")
    parser.add_argument("--old_vis_method", action="store_true", help="Use the old method of optical flow visualization")
    parser.add_argument('--depth_source', type=str, default='precomputed', choices=['precomputed', 'online'], help="Where the Metric3D depth input comes from: precomputed 'depth_color' PNGs on disk, or online ONNX/TensorRT inference on the fly")
    parser.add_argument('--onnx_path', type=str, default=None, help="Path to an existing Metric3D ONNX export (required when --depth_source online)")

    parser.add_argument('--model_name', type=str, default="DTMA", help="Model variant name")

    args = parser.parse_args()

    # Load model
    if "." in args.model_name:
        model_name = args.model_name.split('.')[0]
    else:
        model_name = args.model_name

    module = import_module("model.{}".format(model_name))
    DTMA = getattr(module, 'DTMA')

    model = DTMA(input_bins=15)
    model.load_state_dict(torch.load(args.checkpoint), strict=False)
    model.cuda()

    vis_method = "old" if args.old_vis_method else "new"

    generate_submission(model, args.save_path, args.visualize, vis_method, args.depth_source, args.onnx_path)
    run(["explorer", args.save_path])