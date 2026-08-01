import numpy as np
import torch
import torch.utils.data as data

import random
import os
from pathlib import Path
import glob

import imageio.v2 as imageio

from .augment import Augmentor

DATA_SPLIT = {
    'train' : ["zurich_city_03_a",
               "zurich_city_09_a",
               "zurich_city_10_a",
               "zurich_city_08_a",
               "zurich_city_01_a",
               "zurich_city_02_a",
               "zurich_city_02_c",
               "zurich_city_02_d",
               "zurich_city_02_e",
               "zurich_city_11_a",
               "zurich_city_11_b",
               "zurich_city_05_a",
               "zurich_city_07_a",
               "thun_00_a"],
    
    'val'   : ["zurich_city_10_b",
               "zurich_city_11_c",
               "zurich_city_05_b",
               "zurich_city_06_a"]
}

def get_sequence_from_filepath(path: Path) -> str | None:
    keywords = {"zurich", "thun", "interlaken"}

    for parent in path.parents:
        name_lower = parent.name.lower()
        if any(keyword in name_lower for keyword in keywords):
            return parent.name

    return None

class DSECfull(data.Dataset):
    def __init__(self, phase, crop:bool = True, flip:bool = True, spatial_aug:bool = True, depth_source:str = "precomputed", onnx_path:str = None):
        assert phase in ["train", "val", "trainval", "test", "prog"]
        assert depth_source in ["precomputed", "online"], f"depth_source must be 'precomputed' or 'online', got '{depth_source}'"
        assert depth_source != "online" or onnx_path, "depth_source='online' requires onnx_path to point at an existing Metric3D ONNX export"

        self.init_seed = False
        self.phase = phase
        self.depth_source = depth_source
        self.onnx_path = onnx_path
        self.depth_dirname = 'depth_color'
        self._online_depth_engine = None
        self.voxels = []
        self.flows = []

        ### Please change the root to satisfy your data saving setting.
        root = './datasets/dsec_full'
        if phase == 'train' or phase == 'trainval':
            self.root = os.path.join(root, 'trainval')

            self.augment = crop or flip or spatial_aug
            crop_size = [288, 384] if crop else None
            self.augmentor = Augmentor(crop_size, do_flip=flip, spatial_aug = spatial_aug)

        elif phase == 'val':
            self.root = os.path.join(root, 'trainval')
            self.augment = False

        elif phase == 'test':
            self.root = os.path.join(root, 'test')
            self.augment = False
        
        else:
            self.root = 'datasets/dsec_prog/trainval'
            self.augment = False


        # Event files
        self.voxels = glob.glob(os.path.join(self.root, '*', 'voxel_grids', '*.npz'))

        # Flow files
        self.flows = glob.glob(os.path.join(self.root, '*', 'flow_forward', '*.npy'))

        # Image files
        self.images = glob.glob(os.path.join(self.root, '*', 'images', '*.png'))

        # Depth files (only needed when reading precomputed depth from disk)
        if self.depth_source == "precomputed":
            self.depths = glob.glob(os.path.join(self.root, '*', self.depth_dirname, '*.png'))
            self.depths.sort()

        self.voxels.sort()
        self.flows.sort()
        self.images.sort()

        if self.phase == 'train' or self.phase == 'val':
            self.filter_files_by_phase()


    def filter_files_by_phase(self):
        self.flows = [f for f in self.flows if get_sequence_from_filepath(Path(f)) in DATA_SPLIT[self.phase]]
        self.voxels = [f for f in self.voxels if get_sequence_from_filepath(Path(f)) in DATA_SPLIT[self.phase]]
        self.images = [f for f in self.images if get_sequence_from_filepath(Path(f)) in DATA_SPLIT[self.phase]]
        if self.depth_source == "precomputed":
            self.depths = [f for f in self.depths if get_sequence_from_filepath(Path(f)) in DATA_SPLIT[self.phase]]

    @property
    def online_depth_engine(self):
        # Imported lazily so "precomputed" mode never requires onnxruntime to be installed.
        # Also built lazily per-process: a DataLoader worker must not inherit a CUDA/ONNX
        # Runtime session created in the parent process.
        if self._online_depth_engine is None:
            from metric3d_online import OnlineDepthEngine
            self._online_depth_engine = OnlineDepthEngine(self.onnx_path)
        return self._online_depth_engine

    
    def get_data_sample(self, index):
        # if not self.init_seed:
        #     worker_info = torch.utils.data.get_worker_info()
        #     if worker_info is not None:
        #         torch.manual_seed(worker_info.id)
        #         np.random.seed(worker_info.id)
        #         random.seed(worker_info.id)
        #         self.init_seed = True
        
        #events
        events_file = np.load(self.voxels[index])
        voxel1 = events_file['events_prev'].transpose(1, 2, 0)
        voxel2 = events_file['events_curr'].transpose(1, 2, 0)

        #image
        img1 = imageio.imread(self.images[index])

        if self.phase != 'test':
            img2 = imageio.imread(self.images[index+1])
        else:
            img2 = np.zeros_like(img1)

        #depth
        if self.depth_source == "online":
            depth, _ = self.online_depth_engine.infer(img1)
        else:
            depth = imageio.imread(self.depths[index])

        #flow
        if self.phase != 'test':
            flow_16bit = np.load(self.flows[index])
            flow_map, valid2D = flow_16bit_to_float(flow_16bit)

            if self.augment:
                voxel1, voxel2, flow_map, valid2D, img1, img2, depth = self.augmentor(voxel1, voxel2, flow_map, valid2D, img1, img2, depth)

            flow_map = torch.from_numpy(flow_map).permute(2, 0, 1).float()
            valid2D = torch.from_numpy(valid2D).float()


        img1 = torch.from_numpy(img1).permute(2, 0, 1).float() / 255.0
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float() / 255.0
        depth = torch.from_numpy(depth).permute(2, 0, 1).float() / 255.0
        # depth = torch.from_numpy(depth).unsqueeze(0).float()
        
        voxel1 = torch.from_numpy(voxel1).permute(2, 0, 1).float()
        voxel2 = torch.from_numpy(voxel2).permute(2, 0, 1).float()

        if self.phase == "test":
            # Include submission coordinates (seuence name, file index)
            file_path = Path(self.voxels[index])
            sequence_name = file_path.parents[1].name
            file_index = int(file_path.stem)
            submission_coords = (sequence_name, file_index)
            return voxel1, voxel2, img1, img2, depth, submission_coords
        
        return voxel1, voxel2, flow_map, valid2D, img1, img2, depth


    def __getitem__(self, index):
        if self.phase != 'test':
            img_file = self.images[index]
            img_idx = int(os.path.basename(img_file).split('.')[0])
            next_img_file = os.path.join(os.path.dirname(img_file), "{:06d}.png".format(img_idx + 2))
            if not os.path.exists(next_img_file):
                try:
                    return self.get_data_sample(index + 1)
                except IndexError:
                    return self.__getitem__(random.randint(0, len(self)-1))
            
        return self.get_data_sample(index)
        

    def __len__(self):
        return len(self.voxels)
    
def flow_16bit_to_float(flow_16bit: np.ndarray):
    assert flow_16bit.dtype == np.uint16
    assert flow_16bit.ndim == 3
    h, w, c = flow_16bit.shape
    assert c == 3

    valid2D = flow_16bit[..., 2] == 1
    assert valid2D.shape == (h, w)
    assert np.all(flow_16bit[~valid2D, -1] == 0)
    valid_map = np.where(valid2D)

    # to actually compute something useful:
    flow_16bit = flow_16bit.astype('float')

    flow_map = np.zeros((h, w, 2))
    flow_map[valid_map[0], valid_map[1], 0] = (flow_16bit[valid_map[0], valid_map[1], 0] - 2 ** 15) / 128
    flow_map[valid_map[0], valid_map[1], 1] = (flow_16bit[valid_map[0], valid_map[1], 1] - 2 ** 15) / 128
    return flow_map, valid2D

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def make_data_loader(phase, batch_size, num_workers, crop = True, flip = True, spatial_aug = True, init_seed = 1, depth_source = "precomputed", onnx_path = None):
    g = torch.Generator()
    g.manual_seed(init_seed)

    dset = DSECfull(phase, crop, flip, spatial_aug, depth_source, onnx_path)
    loader = data.DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=True,
        drop_last=True)
    
    return loader


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import time

    dset = DSECfull('test', crop=False, flip=False, spatial_aug=False)

    items = dset.get_data_sample(415)

    print(len(items))
    quit()

    for item in dset:
        v1, v2, img1, img2, *_, coords = item
        print(coords, v1.shape, img1.shape, sep = "\t")


    v1, v2, flow, valid, img1, img2, depth = dset[0]
    print(v1.shape, v2.shape, flow.shape, valid.shape, img1.shape, img2.shape, depth.shape)


    depth = depth[0].numpy()

    plt.subplot(1, 2, 1)
    plt.imshow(depth)
    
    plt.subplot(1, 2, 2)
    plt.imshow(img1.permute(1, 2, 0).numpy().astype('uint8'))
    
    plt.show()

