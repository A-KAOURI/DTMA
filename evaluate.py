from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import os
# from dataloader.dsec_split import DSECsplit
from dataloader.dsec_full import DSECfull
from pathlib import Path

from utils.visualization import visualize_optical_flow
from utils.supervision import sequence_loss

@torch.no_grad()
def get_vis_prog(model):
    visualizations = []
    loader = DSECfull('prog')

    for index, (voxel1, voxel2, flow_map, valid2D, img1, img2, depth) in enumerate(loader):
        voxel1 = voxel1[None].cuda()
        voxel2 = voxel2[None].cuda()
        depth = depth[None].cuda()
        img1 = img1[None].cuda()

        flow_pred, img2_pred = model(voxel1, voxel2, depth, img1)
        flow_pred = flow_pred[0].cpu()

        flow_map = flow_map.numpy()
        valid2D = valid2D.numpy()

        # Optical flow
        flow_pred_vis, _ = visualize_optical_flow(flow_pred.numpy())
        flow_pred_vis = (flow_pred_vis * 255).astype('uint8')
        
        flow_map = flow_map.transpose(1, 2, 0)
        flow_map[valid2D == 0] = 0
        flow_gt_vis, _ = visualize_optical_flow(flow_map.transpose(2, 0, 1))
        flow_gt_vis = (flow_gt_vis * 255).astype('uint8')

        # Image
        img = img2.numpy().transpose(1, 2, 0) * 255
        img = img.astype('uint8')

        img_pred = img2_pred[0].detach().cpu().numpy().transpose(1, 2, 0) * 255
        img_pred = img_pred.astype('uint8')
        
        # Depth
        depth = depth[0].cpu().numpy().transpose(1, 2, 0) * 255
        depth = depth.astype('uint8')


        # construct the visualization frame
        row0 = np.hstack([flow_pred_vis, img])
        row1 = np.hstack([flow_gt_vis, img_pred])

        vis = np.vstack([row0, row1])
        visualizations.append((index, vis))

        break

    return visualizations

@torch.no_grad()
def validate_dsec(model, val_loader, gamma_weight = 0.8, max_flow = 400):
    if not model.training:
        # We need model in training mode to get all predictions
        model.train()

    metrics_avg = {}

    bar = tqdm(val_loader,total=len(val_loader), ncols=60, leave=False, desc="Validation")
    for (voxel1, voxel2, flow_map, valid2D, img1, img2, depth) in bar:
        flow_preds, img2_pred, _ = model(voxel1.cuda(), voxel2.cuda(), depth.cuda(), img1.cuda())

        _ , metrics = sequence_loss(flow_preds, flow_map.cuda(), valid2D.cuda(),
                                                    img2_pred, img2.cuda(),
                                                    gamma = gamma_weight, max_flow = max_flow)


        # Add Averaged Flow Metrics
        for key in metrics.keys():
            val_key = "/".join(['Val', key])
            if not val_key in metrics_avg:
                metrics_avg[val_key] = 0.0
            metrics_avg[val_key] += metrics[key] / len(val_loader)

    return metrics_avg




if __name__ == "__main__":
    loader = DSECfull('val')
    for item in tqdm(loader):
        pass