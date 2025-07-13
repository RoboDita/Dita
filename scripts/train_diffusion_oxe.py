import os
import argparse
from dataclasses import dataclass
import logging
import time
import random
import cv2
import datetime
import importlib
import subprocess
import sys

current_path = os.getcwd()
sys.path.append(current_path)
sys.path.append(os.path.join(current_path, "utils/"))
sys.path.append(os.path.join(current_path, "scripts"))
sys.path.append(os.path.join(current_path, "openvla"))

from openvla.prismatic.util.data_utils import PaddedCollatorForActionPrediction
from openvla.prismatic.vla.datasets.datasets import RLDSDataset
from openvla.prismatic.util import set_global_seed
from utils.ddp_utils import init_distributed_mode

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from llama_dp import RobotTransformerNet
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from functools import partial
import math

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from utils import resume_or_load_checkpoint

import tensorflow_io as tfio
import numpy as np
from pytorch3d.transforms import (
    Transform3d,
    matrix_to_euler_angles,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_to_matrix,
)
from scipy.spatial.transform import Rotation as R
from transformers import AutoTokenizer, CLIPModel
from einops import rearrange, reduce

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["AWS_ACCESS_KEY_ID"] = "Your Key"
os.environ["AWS_SECRET_ACCESS_KEY"] = "Your Key"
os.environ["S3_ENDPOINT"] = "Your EndPoint"
os.environ["S3_USE_HTTPS"] = "0"
os.environ["S3_VERIFY_SSL"] = "0"

def unnormalize(x):
    x = x.clone()
    for i in range(3):
        x[..., i] = x[..., i] * IMAGENET_DEFAULT_STD[i] + IMAGENET_DEFAULT_MEAN[i]
    return x

def adjust_learning_rate(iter, configs):
    """Decay the learning rate with half-cycle cosine after warmup"""
    warmup_iters = configs['warmup_iters']
    total_iters = configs['iters']
    min_lr_scale = configs['min_lr_scale']

    if iter < configs['warmup_iters']:
        lr_scaler = 1.0 * iter / warmup_iters
    else:
        lr_scaler = min_lr_scale + (1.0 - min_lr_scale) * 0.5 * \
            (1.0 + math.cos(math.pi * (iter - warmup_iters) / (total_iters - warmup_iters)))
    return lr_scaler

def reduce_and_average(data):
    torch.distributed.all_reduce(data, op=torch.distributed.ReduceOp.AVG)
    return data

def param_groups_weight_decay(model: nn.Module, lr=1e-4, weight_decay=1e-5, no_weight_decay_list=(), lr_mult=1.0, pretrained_weight_list=()):
    no_weight_decay_list = set(no_weight_decay_list)

    pretrained_decay = []
    pretrained_no_decay = []
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_list:
            if len(list(filter(lambda x: x in name, pretrained_weight_list))) > 0:
                pretrained_no_decay.append(param)
            else:
                no_decay.append(param)
        else:
            if len(list(filter(lambda x: x in name, pretrained_weight_list))) > 0:
                pretrained_decay.append(param)
            else:
                decay.append(param)

    return [
        {"params": no_decay, "weight_decay": 0.0, "lr": lr},
        {"params": decay, "weight_decay": weight_decay, "lr": lr},
        {"params": pretrained_no_decay, "weight_decay": 0.0, "lr": lr * lr_mult},
        {"params": pretrained_decay, "weight_decay": weight_decay, "lr": lr * lr_mult},
    ]

@dataclass
class DataAdapterForOpenx:        
    def __call__(self, rlds_batch, cfg):
        loss_weight = torch.logical_not(torch.tensor(rlds_batch['action_past_goal']))
        if 'load_proprio' in cfg:
            dataset_name, action = rlds_batch["dataset_name"], torch.tensor(rlds_batch["proprio"])
            state = torch.tensor(rlds_batch['observation']["state"])
        else:
            dataset_name, action = rlds_batch["dataset_name"], torch.tensor(rlds_batch["action"])
            state = torch.tensor(rlds_batch['action'])
        lang = [item.decode().strip() for item in rlds_batch["task"]["language_instruction"].tolist()]
        dataset_name = [item.decode() for item in rlds_batch["dataset_name"].tolist()]

        pixel_values = torch.tensor(rlds_batch["observation"]["image_primary"])
        # Normalize 
        pixel_values = (pixel_values / 255. - torch.tensor(IMAGENET_DEFAULT_MEAN)) / torch.tensor(IMAGENET_DEFAULT_STD)
        pixel_values = pixel_values.permute(0, 1, 4, 2, 3)
        del rlds_batch
        return dict(pixel_values=pixel_values, action=action, state=state, dataset_name=dataset_name, language_instruction=lang, loss_weight=loss_weight)

@hydra.main(version_base=None, config_path=os.path.join(os.path.dirname(__file__), "..", "config"), config_name="config_diffusion_openx")
def train(cfg: DictConfig):
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    args = argparse.Namespace()
    print(args)

    # TODO You should revise this part according to your machines.
    init_distributed_mode(args, cfg)
    
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = 0
    if 'RANK' in os.environ:
        rank = int(os.environ["RANK"])
    else:
        rank = 0
    
    print(cfg.dataset.data_path)
    
    data_path = cfg.dataset.data_path
    shuffle_buffer_size = cfg.shuffle_buffer_size

    # oxe_magic_soup_plus_minus
    vla_dataset_openx = RLDSDataset(
        data_path, 
        cfg.dataname, 
        DataAdapterForOpenx(), 
        resize_resolution=(224, 224), 
        shuffle_buffer_size=shuffle_buffer_size, 
        train=True, 
        image_aug=cfg.image_aug if 'image_aug' in cfg else False,
        window_size=cfg.dataset.traj_length + 1 - cfg.num_pred_action, 
        future_action_window_size=cfg.num_pred_action-1, 
        batch_size=cfg.batch_size, 
    )

    train_dataloader = vla_dataset_openx
    
    if args.distributed:
        DEVICE = "cuda:" + str(os.environ["LOCAL_RANK"]) if torch.cuda.is_available() else "cpu"
    else:
        DEVICE = 'cuda'

    network = RobotTransformerNet(
        output_tensor_spec=None,
        vocab_size=cfg.model.vocab_size,
        trajectory_dim=7,
        time_sequence_length=cfg.dataset.traj_length,
        num_layers=cfg.model.num_layers,
        dropout_rate=cfg.model.dropout_rate,
        include_prev_timesteps_actions=cfg.model.include_prev_timesteps_actions,
        freeze_backbone=cfg.model.freeze_backbone,
        use_qformer=cfg.model.use_qformer,
        use_wrist_img=cfg.model.use_wrist_img,
        use_depth_img=cfg.model.use_depth_img,
        prediction_type=cfg.prediction_type,
        dim_align_type=cfg.dim_align_type if 'dim_align_type' in cfg else 0,
        input_size='(224, 224)',
        scheduler_type=cfg.scheduler_type,
        num_inference_steps=cfg.num_inference_steps,
        attn_implementation=cfg.attn_implementation,
        use_action_head_diff=cfg.use_action_head_diff,
    )
    
    clip_tokenizer = AutoTokenizer.from_pretrained(
        "openai/clip-vit-large-patch14", use_fast=False
    )
    clipmodel = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)

    params = param_groups_weight_decay(network, lr=cfg.lr, weight_decay=0.05, lr_mult=0.1, pretrained_weight_list=("image_tokenizer.tokenizer",))

    if cfg.optimizer.name == "adamw":
        optimizer = create_optimizer_v2(
            params, opt=cfg.optimizer.name, lr=cfg.lr, weight_decay=cfg.optimizer.weight_decay, betas=(cfg.optimizer.betas_0, cfg.optimizer.betas_1)
        )

    scheduler, _ = create_scheduler_v2(
        optimizer,
        sched=cfg.scheduler.sched,
        warmup_lr=cfg.lr,
        warmup_epochs=0,
        num_epochs=cfg.scheduler.num_epochs,
        decay_epochs=cfg.scheduler.decay_epochs,
        updates_per_epoch=len(train_dataloader),
        step_on_epochs=cfg.scheduler.step_on_epochs,
    )

    if "use_adjust_scheduler" in cfg and cfg.use_adjust_scheduler:
        print("use adjust scheduler!!!!!!!", flush=True)
        lr_scheduler_configs = {
            'warmup_iters': 1000,
            'iters': 100000,
            'min_lr_scale': cfg.min_lr_scale
        }
        lr_lambda = partial(adjust_learning_rate, configs=lr_scheduler_configs)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    start_epoch = 0
    total_iter_num = 0
    start_epoch, total_iter_num, checkpoint_path, tensorboard_path, log_path, run_dir = resume_or_load_checkpoint(cfg, network, optimizer, None)  
    
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(DEVICE)

    if args.distributed:
        network = torch.nn.parallel.DistributedDataParallel(network.cuda(local_rank), device_ids=[local_rank], find_unused_parameters=False)
        network_module = network.module
    else:
        network_module = network
        network = network.to(DEVICE)

    if 'eval_only' in cfg:
        print('Currently, we only support libero')
        if "libero" in cfg.dataname:
            close_loop_eval = getattr(importlib.import_module("close_loop_eval_diffusion_libero"), "close_loop_eval_libero")
            _ = close_loop_eval(
                model=network,
                test_episodes_num=cfg.close_loop_eval.test_episodes_num,
                args=args,
                stride=cfg.dataset.stride,
                root_folder=os.path.join(HydraConfig.get().runtime.cwd, HydraConfig.get().run.dir, "close_loop_videos", f"{total_iter_num}_iters"),
                cfg=cfg,
                dataset_statistics=vla_dataset_openx.dataset_statistics,
            )
        exit()

    writer = None

    if rank == 0:
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path, exist_ok=True)
        if cfg.task_name != "test":
            if not os.path.exists(tensorboard_path):
                os.makedirs(tensorboard_path, exist_ok=True)
            writer = SummaryWriter(tensorboard_path)
        logging.basicConfig(filename=log_path, level=logging.INFO, format="%(asctime)s : %(message)s")
        print("Training!", flush=True)
    
    network.train()

    for epoch in range(start_epoch, cfg.epoch):
        running_loss = 0.0
        data_start_time = time.time()
        iter_start_time = time.time()
        
        print('begin')
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            obs_new = {}
            obs_new['language_instruction'] = batch['language_instruction']
            obs_new['image'] = batch['pixel_values'].to(DEVICE).clone()

            if 'load_proprio' in cfg:
                actions = batch["action"][:, :].to(DEVICE).clone()
                obs_new['poses'] = batch['state'][:, :1, :]
            else:
                actions = batch["action"].to(DEVICE).clone()

            train_start_time = time.time()
            loss_mask = batch['loss_weight'].to(DEVICE).clone()
            trajectory = actions

            noise = torch.randn(trajectory.shape, device=trajectory.device)
            bsz = trajectory.shape[0]

            timesteps = torch.randint(0, network_module.noise_scheduler.config.num_train_timesteps, (bsz,), device=trajectory.device).long()
            noisy_trajectory = network_module.noise_scheduler.add_noise(trajectory, noise, timesteps)

            inputs = clip_tokenizer(text=obs_new['language_instruction'], return_tensors="pt", max_length=77, padding="max_length", truncation=True)
            for key in inputs:
                inputs[key] = inputs[key].to(DEVICE)

            ccontext = clipmodel.text_model(**inputs)[0].squeeze(0).detach()
            ccontext = ccontext[:, None,...].repeat(1, obs_new['image'].shape[1], 1, 1)
            obs_new['natural_language_embedding'] = ccontext

            with torch.cuda.amp.autocast():
                pred = network(obs_new, None, noisy_action_tokens=noisy_trajectory, timesteps=timesteps, num_pred_action=cfg.num_pred_action,)
                
                if network_module.noise_scheduler.config.prediction_type == 'epsilon':
                    target = noise
                elif network_module.noise_scheduler.config.prediction_type == 'sample':
                    target = trajectory
                elif network_module.noise_scheduler.config.prediction_type == 'v_prediction':
                    target = network_module.noise_scheduler.get_velocity(trajectory, noise, timesteps)
                else:
                    raise ValueError(f"Unsupported prediction type {network_module.noise_scheduler.config.prediction_type}")
                
                b, num, dim = pred.shape
                logits = pred
                loss = F.mse_loss(logits[...,:,:], target[..., :,:, ], reduction='none')
                orig_loss = loss

                running_loss = loss.detach()
                loss = loss[loss_mask]
                loss = reduce(loss, 'b ... -> b (...)', 'mean')
                extra_loss = reduce(orig_loss[:, :-cfg.dataset.traj_length], 'b ... -> b (...)', 'mean').detach()
                loss = loss.mean()
                
            loss.backward()
            optimizer.step()

            loss_rota = running_loss[...,3:6].mean()
            loss_world_vector = running_loss[...,:3].mean()
            loss_grip_close = running_loss[...,6:8].mean()
            running_loss = running_loss.mean()
            
            running_loss = reduce_and_average(running_loss)
            loss_rota = reduce_and_average(loss_rota)
            loss_world_vector = reduce_and_average(loss_world_vector)
            loss_grip_close = reduce_and_average(loss_grip_close)

            if "use_adjust_scheduler" in cfg and cfg.use_adjust_scheduler:
                scheduler.step()
            else:
                scheduler.step_update(epoch * len(train_dataloader) + i)
            
            if rank == 0:
                if i % 10 == 0:
                    iter_end_time = time.time()
                    iter_time = (iter_end_time - iter_start_time) / 10
                    train_time = (iter_end_time - train_start_time)
                    iter_start_time = time.time()
                    print("[epoch {}, iter {}, iter_time {}, train_time {}, ] lr: {} loss: {}, world_vector:{}, rota:{}, grip:{}".
                        format(epoch + 1, i + 1, iter_time, train_time, optimizer.param_groups[0]["lr"], running_loss, loss_world_vector, loss_rota, loss_grip_close), flush=True)

                if writer is not None:
                    writer.add_scalar("MSE_loss", running_loss, total_iter_num)
                    writer.add_scalar("MSE_loss_rota", loss_rota, total_iter_num)
                    writer.add_scalar("MSE_loss_world_vector", loss_world_vector, total_iter_num)
                    writer.add_scalar("MSE_loss_grip_close", loss_grip_close, total_iter_num)
            
            sys.stdout.flush()

            if (rank == 0 and total_iter_num != 0 and (total_iter_num % 1000 == 0) and 'no_checkpoints' not in cfg):
                checkpoint = {
                    "parameter": network_module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "iter": i,
                    "total_iter_num": total_iter_num,
                    "loss": running_loss,

                }
                print("save checkpoint!", os.path.join(checkpoint_path, f"ckpt_{total_iter_num}.pth"))
                torch.save(checkpoint, os.path.join(checkpoint_path, f"ckpt_{total_iter_num}.pth"))


            # EVAL PART
            if "libero" in cfg.dataname and total_iter_num % cfg.close_loop_eval.eval_iters == 0 and total_iter_num != 0 and cfg.use_close_loop_eval:
                close_loop_eval = getattr(importlib.import_module("close_loop_eval_diffusion_libero"), "close_loop_eval_libero")
                close_loop_eval_start_time = time.time()
                _ = close_loop_eval(
                    model=network,
                    test_episodes_num=cfg.close_loop_eval.test_episodes_num,
                    args=args,
                    stride=cfg.dataset.stride,
                    root_folder=os.path.join(HydraConfig.get().runtime.cwd, HydraConfig.get().run.dir, "close_loop_videos", f"{total_iter_num}_iters"),
                    cfg=cfg,
                    dataset_statistics=vla_dataset_openx.dataset_statistics,
                )

            total_iter_num += 1
            data_start_time = time.time()

    checkpoint = {
        "parameter": network_module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "iter": i,
        "total_iter_num": total_iter_num,
        "loss": running_loss,

    }
    print("save checkpoint!", os.path.join(checkpoint_path, f"ckpt_{total_iter_num}.pth"))

    torch.save(checkpoint, os.path.join(checkpoint_path, f"ckpt_{total_iter_num}.pth"))

if __name__ == "__main__":
    train()
