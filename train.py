import sys
sys.path.append('model')
import time
import os
import random
from tqdm import tqdm
import wandb
import torch
from torch import nn
import numpy as np
from subprocess import run
import glob

from utils.file_utils import get_logger
from dataloader.dsec_full import make_data_loader

from utils.supervision import sequence_loss

####Important####
# from model.DTMA import DTMA
####Important####

from datetime import datetime
from torchvision.transforms import v2
from flow_vis import flow_to_color

from utils.visualization import writer_add_features
from evaluate import get_vis_prog, validate_dsec

from importlib import import_module

MAX_FLOW = 400
SUM_FREQ = 100
VIS_FREQ = 1000
VAL_FREQ = 5000
SAVE_FREQ = 10000

CROP_HEIGTH = 288
CROP_WIDTH = 384

READY = False

# Half precision
#torch.set_default_dtype(torch.float16)

class Loss_Tracker:
    def __init__(self, wandb, sum_frequency = 100):
        self.running_loss = {}
        self.wandb = wandb
        self.sum_frequency = sum_frequency
        self.first_step = True

    def push(self, metrics, step):
        # self.total_steps += 1
        
        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0
            
            self.running_loss[key] += metrics[key]

        if self.first_step:
            self.first_step = False
            return
        
        if step % self.sum_frequency == 0:
            if self.wandb:
                for key in self.running_loss:
                    wandb.log({key: self.running_loss[key]/SUM_FREQ}, step = step)
            self.running_loss = {}
        
    def state_dict(self):
        return {"running_loss":self.running_loss,
                "wandb":self.wandb}
    
    def load_state_dict(self, state_dict:dict):
        keys = ["running_loss", "wandb"]
        for key in keys:
            self.__setattr__(key, state_dict[key])

            

class Trainer:
    def __init__(self, args):
        self.args = args

        if "." in args.model_name:
            model_name = args.model_name.split('.')[0]
        else:
            model_name = args.model_name

        module = import_module("model.{}".format(model_name))
        DTMA = getattr(module, 'DTMA')

        self.model = DTMA(input_bins=15)
        self.model = self.model.cuda()
        # self.model = nn.DataParallel(self.model, device_ids=[0])

        self.date_label = datetime.now().strftime("%Y-%m-%d")
        self.param_label = "LR{}_BS{}_IT{}_W{}".format(args.lr,args.batch_size, args.iters, args.weight)
        self.save_path = os.path.join(args.checkpoint_dir, self.date_label, self.param_label)

        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)

        #Loader
        phase = 'trainval' if not args.validate else 'train'
        self.train_loader = make_data_loader(phase, args.batch_size, args.num_workers, init_seed = args.init_seed, depth_source = args.depth_source, onnx_path = args.onnx_path)
        print('train_loader done!')

        if args.validate:
            self.val_loader = make_data_loader('val', batch_size = args.batch_size // 2, num_workers = 0, init_seed = args.init_seed, depth_source = args.depth_source, onnx_path = args.onnx_path)

        #Optimizer and scheduler for training
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=0.0001
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, 
            max_lr=args.lr,
            total_steps=args.num_steps + 100,
            pct_start=0.01,
            cycle_momentum=False,
            anneal_strategy='linear')
        

        #Logger
        self.checkpoint_dir = args.checkpoint_dir
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.writer = get_logger(os.path.join(self.checkpoint_dir, self.date_label, 'train.log'))
        self.tracker = Loss_Tracker(args.wandb, sum_frequency = SUM_FREQ)

        
        #Loading checkpoint
        self.continue_training = args.continue_training
        self.old_ckpt_path = args.model_path
        self.previous_step = None

        if self.old_ckpt_path == "":
            if self.continue_training:
                print("Cannot continue training without a pretrained model checkpoint. Please provide '--model_path'")
                self.continue_training = False
        else:
            if os.path.isfile(self.old_ckpt_path):
                params_ckpt_path = os.path.join(os.path.dirname(self.old_ckpt_path), "params_checkpoint")
                params_ckpt_path = params_ckpt_path if self.continue_training else None
                self.previous_step = self.load_ckpt(self.old_ckpt_path, params_ckpt_path)
                self.writer.info(f"Loaded the checkpoint at '{self.old_ckpt_path}'.")
                if self.continue_training:
                    self.writer.info("Loaded parameters for continuous learning.")
            else:
                print("Couldn't find a checkpoint file at '{}'".format(self.old_ckpt_path))

        self.writer.info('====A NEW TRAINING PROCESS====')

    def train(self):
        # self.writer.info(self.model)
        self.writer.info(self.args)
        self.model.train()
        
        total_steps = 0
        vis_steps = 0
        if self.continue_training:
            total_steps = self.previous_step
            vis_steps = int(total_steps / VIS_FREQ)

        keep_training = True
        while keep_training:

            bar = tqdm(self.train_loader,total=len(self.train_loader), ncols=60)
            for (voxel1, voxel2, flow_map, valid2D, img1, img2, depth) in bar:
                # voxel1, voxel2, flow_map, valid2D = self.apply_transforms(data_items)
                self.optimizer.zero_grad()

                flow_preds, img2_pred, vis_output = self.model(voxel1.cuda(), voxel2.cuda(), depth.cuda(), img1.cuda())

                flow_loss, metrics = sequence_loss(flow_preds, flow_map.cuda(), valid2D.cuda(),
                                                   img2_pred, img2.cuda(),
                                                   gamma= self.args.weight, max_flow= MAX_FLOW)

                flow_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)

                self.optimizer.step()
                self.scheduler.step()

                bar.set_description(f'Step: {total_steps}/{self.args.num_steps}')
                self.tracker.push(metrics, step=total_steps)
                total_steps += 1
                if total_steps and total_steps % VIS_FREQ == 0 and self.args.wandb:
                    self.model.eval()
                    vis_steps += 1
                    with torch.no_grad():
                        # Get progress visualizations
                        visualizations = get_vis_prog(self.model)
                        for i, vis in visualizations:
                            wandb.log({'progress_{}'.format(i+1): wandb.Image(vis)})

                        #Features
                        for key, value in vis_output.items():
                            value = writer_add_features(value)
                            wandb.log({key:wandb.Image(value)})
                    self.model.train()

                if total_steps and total_steps % VAL_FREQ == 0 and self.args.validate:
                    val_metrics = validate_dsec(self.model, self.val_loader, gamma_weight=self.args.weight, max_flow=MAX_FLOW)
                    for key in val_metrics:
                        wandb.log({key: val_metrics[key]}, step = total_steps)

                if total_steps and total_steps % SAVE_FREQ == 0:
                    # Checkpoint savepath
                    ckpt = os.path.join(self.save_path, f'checkpoint_{total_steps}')

                    # Save checkpoint with parameters for continuous training
                    params_state = {"step":total_steps,
                            "model":self.model.state_dict(),
                            "optimizer":self.optimizer.state_dict(),
                            "scheduler":self.scheduler.state_dict(),
                            "loss_tracker":self.tracker.state_dict()}

                    torch.save(params_state, ckpt)

                if total_steps > self.args.num_steps:
                    keep_training = False
                    break
            
            time.sleep(0.03)
        
        # Save final Checkpoint
        model_ckpt_path = os.path.join(self.save_path, "checkpoint.pth")
        torch.save(self.model.state_dict(), model_ckpt_path)

        return model_ckpt_path
    
    def load_ckpt(self, ckpt_path:str, continuous = False):
        if os.path.isfile(ckpt_path):
            # Load the model
            checkpoint = torch.load(ckpt_path)
            if "model" in checkpoint.keys():
                self.model.load_state_dict(checkpoint["model"], strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)

            # Load training params
            if continuous:
                try:
                    self.optimizer.load_state_dict(checkpoint["optimizer"])
                    self.scheduler.load_state_dict(checkpoint["scheduler"])
                    self.tracker.load_state_dict(checkpoint["loss_tracker"])
                except KeyError:
                    print("It seems like one or more parameters are missing in the checkpoint. Cannot continue learning.")
                    self.continue_training = False
                    return
                
                return checkpoint["step"]
        else:
            print("Warning : No checkpoint was found at '{}'".format(ckpt_path))


    def generate_submission(self):
        """Generate submission to test the model"""
        checkpoint = os.path.join(self.save_path, "checkpoint.pth")
        if not os.path.exists(checkpoint):
            print("checkpoint.pth does not exist, trying with last checkpoint")
            ckpt_list = glob.glob(self.save_path)
            ckpt_list.sort()
            if ckpt_list:
                checkpoint = ckpt_list[-1]
                print("Found last checkpoint : ", os.path.basename(checkpoint))
            else:
                print("No checkpoint found ! Aborting test ...")
                return
            
        print("Running test :")
        submission_path = os.path.join("submissions", self.checkpoint_dir, self.date_label, self.param_label)
        test_cmd = ["test.py", "-c", checkpoint, "-s", submission_path, "-v", "--old_vis_method", "--depth_source", self.args.depth_source]
        if self.args.onnx_path:
            test_cmd += ["--onnx_path", self.args.onnx_path]
        run(test_cmd)

        return(submission_path)

    # def apply_transforms(self, data_items):
    #     transformed_items = data_items
    #     if self.crop:
    #         crop_size = (CROP_HEIGTH, CROP_WIDTH)
    #         rand_crop = v2.RandomCrop(crop_size)
    #         i, j, h, w = rand_crop.get_params(data_items[0], output_size = crop_size)

    #         transformed_items = [v2.functional.crop(item, i, j, h, w) for item in transformed_items]

    #     if self.hflip and torch.rand() > 0.5:
    #         transformed_items = [v2.functional.hflip(item) for item in transformed_items]

    #     return transformed_items
      

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


        
if __name__=='__main__':
    import argparse


    parser = argparse.ArgumentParser(description='DTMA')
    #training setting
    parser.add_argument('--num_steps', type=int, default=200000)
    parser.add_argument('--checkpoint_dir', type=str, default='')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--validate', action='store_true', help="Train with validation")

    #dataloader setting
    parser.add_argument('--init_seed', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--depth_source', type=str, default='precomputed', choices=['precomputed', 'online'], help="Where the Metric3D depth input comes from: precomputed 'depth_color' PNGs on disk, or online ONNX/TensorRT inference on the fly")
    parser.add_argument('--onnx_path', type=str, default=None, help="Path to an existing Metric3D ONNX export (required when --depth_source online)")

    #model setting
    parser.add_argument('--model_name', type=str, default="DTMA", help="Model variant name")
    parser.add_argument('--grad_clip', type=float, default=1)
    parser.add_argument('--iters', type=int, default=6, help='Number of GRU itrations')

    # loss setting
    parser.add_argument('--weight', type=float, default=0.8)

    #wandb setting
    parser.add_argument('--wandb', action='store_true', default=False)

    #Loading pretrained models
    parser.add_argument('--model_path', type=str, default="", help="Path to existing model to be loaded")
    parser.add_argument('--continue_training', action='store_true', default=False, help="Continue learning with previous params")
        
    args = parser.parse_args()

    set_seed(args.init_seed)
    
    if args.wandb:
        wandb_name = args.checkpoint_dir.split('/')[-1]
        wandb.init(name=wandb_name, project='DTMA_DSEC_full')

    trainer = Trainer(args)
    trainer.train()
    submission_path = trainer.generate_submission()
    run(["explorer", submission_path])
    
