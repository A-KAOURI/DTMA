from event_slicer import EventSlicer
import h5py
import hdf5plugin
import numpy as np
import os
from glob import glob
from tqdm import tqdm
import imageio.v2 as imageio

import torch
import cv2

from pathlib import Path

from representation import VoxelGrid
from visualization import events_to_event_image
from warper import Warper

RESOLUTION = (480, 640)
TEMPORAL_BINS = 15


representation = VoxelGrid((TEMPORAL_BINS, *RESOLUTION), normalize=True)
# representation = StackedHistogram(bins=10, height=RESOLUTION[0], width=RESOLUTION[1], fastmode=False)
# representation = MixedDensityEventStack(bins=15, height=RESOLUTION[0], width=RESOLUTION[1])

def rectify_and_write(rect_map, events_curr, events_prev, output_dir, idx):

    #rectify events and convert to voxel grids
    p = events_prev['p']
    t = events_prev['t']
    x = events_prev['x']
    y = events_prev['y']
    xy_rect = rect_map[y, x]
    x_rect = xy_rect[:, 0]
    y_rect = xy_rect[:, 1]
    events0 = events_to_voxel_grid(x_rect, y_rect, p, t)

    xy_rect_out = xy_rect

    p = events_curr['p']
    t = events_curr['t']
    x = events_curr['x']
    y = events_curr['y']
    xy_rect = rect_map[y, x]
    x_rect = xy_rect[:, 0]
    y_rect = xy_rect[:, 1]
    events1 = events_to_voxel_grid(x_rect, y_rect, p, t)

    #write events
    output_name = os.path.join(output_dir, '{:06d}'.format(idx))
    if os.path.exists(output_name):
        return xy_rect_out
    # np.savez(output_name, events_prev=events0, events_curr=events1)
    np.savez(output_name, events_prev=events0, events_curr=events1)
    return xy_rect_out

    # Also save raw rectified events
    # raw_output_dir = output_dir / 'raw_events'
    # raw_output_dir.mkdir(parents=True, exist_ok = True)
    # raw_output_name = raw_output_dir / '{:06d}'.format(idx)
    # np.savez(raw_output_name, p = p, t = t, x = x_rect, y=y_rect)
    # Event image

    # output_dir = output_dir.parent / 'event_images'
    # output_dir.mkdir(parents=True, exist_ok=True)
    # imageio.imwrite(output_dir / '{:06d}.png'.format(idx), events_vis)


def events_to_voxel_grid(x, y, p, t):
    t = (t - t[0]).astype('float32')
    t = (t/t[-1])
    # t = t.astype('uint8')
    # sort_indices = np.argsort(t)
    # t = t[sort_indices]
    x = x.astype('float32')
    y = y.astype('float32')
    p = p.astype('float32')
    event_data_torch = {
        'p': torch.from_numpy(p),
        't': torch.from_numpy(t),
        'x': torch.from_numpy(x),
        'y': torch.from_numpy(y),
        }
    return representation.convert(event_data_torch)

def get_zero_mask(img):
    """Get the mask of zero values in an image"""
    img = img.astype('uint8')
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY_INV)
    return mask.astype(bool)

def gen_dsec(dsec_path:Path, split = 'train', align_to_flow = True, fill_missing = True):
    """
        Assuming that dsec_path has the following folders:
            - train_events
            - train_optical_flow
            - train_calibration
            - train_images
            - test_events
            - test_calibration
            - test_images
        This function exports the dataset by converting events to voxel grids and saving them as .npz files,
        the optical flows as .npy files, and the images as .png files. Images are always rectified into the
        event camera's rectified coordinate frame (via Warper, from each sequence's own calibration) before
        being written out -- DSEC's plain `images/left/rectified` PNGs plus rectify_map.h5/cam_to_cam.yaml
        are all that's needed, no separate distorted-image download is used.
    """
    assert dsec_path.is_dir()
    assert split.lower() in ['train', 'trainval', 'test']
    if split == 'trainval' : split  = 'train'

    event_path = dsec_path / f'{split}_events'
    flow_path = dsec_path / f'{split}_optical_flow'
    calibration_path = dsec_path / f'{split}_calibration'

    images_path = dsec_path / f'{split}_images'

    if split == 'train':
        output_root = Path(f'datasets/dsec_full/trainval')
        sequences = [seq.name for seq in flow_path.iterdir()]
    else:
        output_root = Path(f'datasets/dsec_full/test')
        flow_path= Path('E:/DSEC/test_forward_optical_flow_timestamps')
        sequences = [path.stem for path in flow_path.iterdir()]

    for seq in sequences:
        # if seq != "zurich_city_01_a":
        #     continue
        # Get event files
        event_h5 = event_path / seq / 'events/left/events.h5'
        rectify_h5 = event_path / seq / 'events/left/rectify_map.h5'

        h5f = h5py.File(event_h5, 'r')
        slicer = EventSlicer(h5f)
        with h5py.File(rectify_h5, 'r') as h5_rect:
            rectify_map = h5_rect['rectify_map'][()]

        # Optical flow and flow timestamps
        if split == 'train':
            ts_file_flow = flow_path / seq / 'flow/forward_timestamps.txt'
            flow_dir = flow_path / seq / 'flow/forward'
            flow_list = sorted(flow_dir.glob("*.png")) 
            timestamps_flow = np.genfromtxt(ts_file_flow, delimiter=',')
            assert timestamps_flow.shape[0]== len(flow_list)
            # timestamps_flow  = np.append(timestamps_flow[:,0], timestamps_flow[-1,1])
        else:
            ts_file_flow = flow_path / f'{seq}.csv'
            timestamps_flow = np.genfromtxt(ts_file_flow, delimiter=',')
            timestamps_flow, indexs = timestamps_flow[:,:2], timestamps_flow[:,2]
        
        # Images and image timestamps
        ts_file_img = images_path / seq / 'images' / 'timestamps.txt'
        img_dir = images_path / seq / 'images/left/rectified'

        timestamps_img = np.genfromtxt(ts_file_img, delimiter=',')
        img_list = sorted(img_dir.glob("*.png"))
        assert timestamps_img.shape[0] == len(img_list)


        output_dir = output_root / seq
        output_dir.mkdir(parents=True, exist_ok=True)

        output_dir_flow = output_dir / 'flow_forward'
        output_dir_events = output_dir / 'voxel_grids'

        for i in tqdm(range(len(timestamps_flow)), ncols=60, desc=seq):
            #current events
            t_curr = timestamps_flow[i,0]
            t_next = timestamps_flow[i,1]
            events_curr = slicer.get_events(t_curr, t_next)
            if events_curr == None:
                print(f'None data can be converted to voxel in {seq} at {i}th timestamps for current condition!')
                continue
            
            if split == "train":
                save_idx = int(os.path.basename(flow_list[i]).split(".")[0])
            else:
                save_idx = int(indexs[i])

            # previous events
            dt = 100 * 1000#us
            t_prev = t_curr - dt
            events_prev = slicer.get_events(t_prev, t_curr)
            if events_prev == None:
                print(f'None data can be converted to voxel in {seq} at {i}th timestamps for previous condition!')
                continue

            output_dir_events.mkdir(parents=True, exist_ok=True)
            xy_rect = rectify_and_write(rectify_map, events_curr, events_prev, output_dir_events, save_idx)

            # optical flow
            if split == "train":
                flow_16bit = imageio.imread(flow_list[i], format='PNG-FI')
                output_dir_flow.mkdir(parents=True, exist_ok=True)
                np.save(os.path.join(output_dir_flow, '{:06d}.npy'.format(save_idx)), flow_16bit)

            # image
            image_index = np.where(timestamps_img == t_curr)[0].item()
            output_img = output_root / seq / 'images'
            output_img.mkdir(parents=True, exist_ok=True)

            img = imageio.imread(img_list[image_index])

            # cv2.imshow('original', img)
            # cv2.waitKey(0)

            calibration_file = calibration_path / seq / 'calibration/cam_to_cam.yaml'
            warper = Warper(rectify_h5, calibration_file)
            img = warper(img)

            h, w = RESOLUTION
            if not (img.shape[0] == h and img.shape[1] == w):
                img = cv2.resize(img, (w, h))

            # cv2.imshow('Calibrated (with events)', img)
            # cv2.waitKey(0)

            if align_to_flow:
                img, missing_grid_mask = warper.forward_warp(img)

            # cv2.imshow('Calibrated (with rectified events / optical flow)', img)
            # cv2.waitKey(0)


            if fill_missing:
                # Perform inpainting using Telea's method
                # Fill only the grid
                inpaint_mask = missing_grid_mask.astype('uint8')
                img = cv2.inpaint(img, inpaint_mask, inpaintRadius = 3, flags=cv2.INPAINT_NS)

                # img[missing_grid_mask] = [0, 0, 255]
                # cv2.imshow('Filled missing grid', img)
                # cv2.waitKey(0)

            # # TEMPORARY VISUALIZATION
            # p = events_prev['p']
            # p = 2.0 * p - 1.0
            # t = events_prev['t']
            # # x = events_prev['x']
            # # y = events_prev['y']
            # x_rect = xy_rect[:, 0]
            # y_rect = xy_rect[:, 1]
            # events = np.concatenate([c[None] for c in [t, x_rect, y_rect, p]], axis = 0).transpose()
            # overlay = events_to_event_image(events, *RESOLUTION, background=torch.from_numpy(img).permute(2, 0, 1))
            # overlay = overlay.cpu().numpy().transpose(1, 2, 0)
            # cv2.imshow('Image and Events', overlay)
            # cv2.waitKey(0)

            save_path = output_img / '{:06d}.png'.format(save_idx)
            imageio.imwrite(save_path, img)


        h5f.close()



if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--dsec", type=Path, required=True, help="Path to the original DSEC dataset")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to generate [[train]/test/all]")
    parser.add_argument("-w", "--warp_image", default=False, action="store_true", help="Warp the images to the rectified events domain")
    parser.add_argument("-f", "--fill_grid", default=False, action="store_true", help="Fill the missing grid of pixels after warping")

    args = parser.parse_args()

    splits = ['train' / 'test'] if args.split == "all" else [args.split]

    for split in splits:
        print(f"Generating DTMA version of DSEC {split} split :")
        gen_dsec(args.dsec, split, args.warp_image, args.fill_grid)