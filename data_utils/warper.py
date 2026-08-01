
from pathlib import Path
import yaml
import h5py
import hdf5plugin
import cv2
import imageio.v2 as imageio

import numpy as np

class Warper(object):
    def __init__(self, rectify_map_file:Path, calibration_file:Path):
        
        with h5py.File(rectify_map_file) as fh:
            self.rectify_map = {k: fh[k][()] for k in fh.keys()}

        with calibration_file.open() as fh:
            self.calibration = yaml.load(fh, Loader=yaml.UnsafeLoader)

        self.remapping_map = self._compute_remapping()

    @staticmethod
    def _conf_to_K(conf):
        K = np.eye(3)
        K[[0, 1, 0, 1], [0, 1, 2, 2]] = conf
        return K
    
    def _compute_remapping(self):
        mapping = self.rectify_map['rectify_map']

        K_r0 = self._conf_to_K(self.calibration['intrinsics']['camRect0']['camera_matrix'])
        K_r1 = self._conf_to_K(self.calibration['intrinsics']['camRect1']['camera_matrix'])

        R_r0_0 = np.array(self.calibration['extrinsics']['R_rect0'])
        R_r1_1 = np.array(self.calibration['extrinsics']['R_rect1'])
        R_1_0 = np.array(self.calibration['extrinsics']['T_10'])[:3, :3]

        # read from right to left:
        # rect. cam. 1 -> norm. rect. cam. 1 -> norm. cam. 1 -> norm. cam. 0 -> norm. rect. cam. 0 -> rect. cam. 0
        P_r0_r1 = K_r0 @ R_r0_0 @ R_1_0.T @ R_r1_1.T @ np.linalg.inv(K_r1)

        H, W = mapping.shape[:2]
        coords_hom = np.concatenate((mapping, np.ones((H, W, 1))), axis=-1)
        mapping = (np.linalg.inv(P_r0_r1) @ coords_hom[..., None]).squeeze()
        mapping = mapping[...,:2] / mapping[..., -1:]
        mapping = mapping.astype('float32')

        return mapping
    
    def forward_warp(self, img, out_shape=None, fill_value=0):
        """
        img: (H,W) or (H,W,C)
        rectify_map: (H,W,2) where rectify_map[y,x] = (new_x, new_y)
        out_shape: (H_out, W_out) or None -> same as img
        """
        H, W = img.shape[:2]
        if out_shape is None:
            H2, W2 = H, W
        else:
            H2, W2 = out_shape

        out = np.full((H2, W2) + img.shape[2:], fill_value, dtype=img.dtype)
        filled = np.zeros((H2, W2), dtype=bool)

        # Source coordinates (vectorized)
        ys, xs = np.indices((H, W), dtype=np.int32)

        # Dest coordinates
        nx = np.rint(self.rectify_map['rectify_map'][..., 0]).astype(np.int32)
        ny = np.rint(self.rectify_map['rectify_map'][..., 1]).astype(np.int32)

        valid = (nx >= 0) & (nx < W2) & (ny >= 0) & (ny < H2)

        # Scatter: source -> destination
        out[ny[valid], nx[valid]] = img[ys[valid], xs[valid]]

         # Mark filled pixels
        filled[ny[valid], nx[valid]] = True

        # Missing pixels = those never written to
        missing = ~filled

        return out, missing
    
    def __call__(self, image):
        image = cv2.remap(image, self.remapping_map, None, interpolation=cv2.INTER_CUBIC)
        return image

if __name__ == "__main__":
    root = Path("E:/DSEC")
    sequence = "zurich_city_01_a"
    img_index = 30

    rectified_image = root / 'train_images' / sequence / 'images/left/rectified' / "{:06d}.png".format(img_index)
    img = cv2.imread(rectified_image)

    rectification_map_file = root / f"train_events" / sequence / "events/left/rectify_map.h5"
    cam_to_cam_file = root / f"train_calibration" / sequence / "calibration/cam_to_cam.yaml"

    warper = Warper(rectification_map_file, cam_to_cam_file, warp_to_flow=True)

    warped_img = warper(img)

    cv2.imshow('warped', warped_img)
    cv2.waitKey(0) 
