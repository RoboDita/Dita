import os




os.environ['LD_PRELOAD']='/xxx/xxx/Anonymous/lib/libtcmalloc_minimal.so.4.5.8'
import argparse
from dataclasses import dataclass
import datetime
from genericpath import isdir
import logging
import os
import subprocess
import sys
import time
import random
import cv2
import datetime

current_path = os.getcwd()
sys.path.append(current_path)
sys.path.append(os.path.join(current_path, "scripts/openx_utils/"))
sys.path.append(os.path.join(current_path, "../embodied_foundation/scripts"))
sys.path.append(os.path.join(current_path, "../embodied_foundation/openvla"))

# export PYTHONPATH="$(pwd)":"$(pwd)/rt1_pytorch/openx_utils/":"$(pwd)/../":"$(pwd)/../embodied_foundation/rt1_pytorch":$PYTHONPATH



from openvla.prismatic.util.data_utils import PaddedCollatorForActionPrediction
from openvla.prismatic.vla.datasets.datasets import RLDSDataset
from openvla.prismatic.util import set_global_seed
from utils.ddp_utils import init_distributed_mode

import hydra
import torch
def get_action_spec(action_spec, DEVICE):

    new_action_spec = {}
    for k in action_spec:
        new_action_spec[k] = {}
        new_action_spec[k]["tensor"] = torch.empty((action_spec[k]["tensor"],), dtype=torch.float32).to(DEVICE)
        new_action_spec[k]["minimum"] = torch.tensor([action_spec[k]["minimum"]], dtype=torch.float32).to(DEVICE)
        new_action_spec[k]["maximum"] = torch.tensor([action_spec[k]["maximum"]], dtype=torch.float32).to(DEVICE)
        if k == "terminate_episode":
            for kk in new_action_spec[k]:
                new_action_spec[k][kk] = new_action_spec[k][kk].to(torch.int32)

    return new_action_spec
# from Dataset_Sim.SimDataset import SimDataset
from Dataset_Lab.LabDataset import LabDataset
from llama_dp import RobotTransformerNet
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import importlib
from torch.utils.data import DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["AWS_ACCESS_KEY_ID"] = "Your Key"
os.environ["AWS_SECRET_ACCESS_KEY"] = "Your Key"
# os.environ["AWS_REGION"] = "us-east-1"
os.environ["S3_ENDPOINT"] = "Your EndPoint"
os.environ["S3_USE_HTTPS"] = "0"
os.environ["S3_VERIFY_SSL"] = "0"
import tensorflow_io as tfio
import numpy as np
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import (
    Transform3d,
    matrix_to_euler_angles,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_to_matrix,
)
from scipy.spatial.transform import Rotation as R
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from timm.utils import NativeScaler
from torch.utils.tensorboard import SummaryWriter
from utils import resume_or_load_checkpoint

def unnormalize(x):
    x = x.clone()
    for i in range(3):
        x[..., i] = x[..., i] * IMAGENET_DEFAULT_STD[i] + IMAGENET_DEFAULT_MEAN[i]

    return x



def get_args_parser():

    parser = argparse.ArgumentParser()
    return parser

def calc_l2_loss(act, output, camera_extrinsic_cv):

    gt_list = []
    out_list = []
    cam2world = Transform3d(matrix=camera_extrinsic_cv.mT).inverse()

    def get_world_translation_rotation(output):

        translation_delta = output["world_vector"].flatten(0, 1)
        rotation_delta = output["rotation_delta"].flatten(0, 1)
        # rotation_delta[..., 0] += 1.0

        pose_1_cam = Transform3d(device=translation_delta.device)
        pose_2_cam = pose_1_cam.rotate(quaternion_to_matrix(rotation_delta)).translate(translation_delta)

        pose1_world = pose_1_cam.compose(cam2world)
        pose2_world = pose_2_cam.compose(cam2world)
        translation_delta_world = pose2_world.get_matrix()[:, -1, :3] - pose1_world.get_matrix()[:, -1, :3]
        rotation_delta_world = matrix_to_quaternion(pose1_world.inverse().compose(pose2_world).get_matrix()[:, :3, :3])

        return translation_delta_world, rotation_delta_world

    translation_pred, rotation_pred = get_world_translation_rotation(output)
    translation_gt, rotation_gt = get_world_translation_rotation(act)

    for k in ["world_vector", "rotation_delta", "gripper_closedness_action", "terminate_episode"]:
        if k == "world_vector":
            gt_list.append(translation_gt)
            out_list.append(translation_pred)
        elif k == "rotation_delta":
            gt_list.append(rotation_gt)
            out_list.append(rotation_pred)
        else:
            gt_list.append(act[k].flatten(0, 1))
            out_list.append(output[k].flatten(0, 1))

    gt = torch.cat(gt_list, dim=-1)
    out = torch.cat(out_list, dim=-1)

    criterion = F.mse_loss

    loss = criterion(gt, out).detach()
    loss_wv = criterion(gt[..., :3], out[..., :3]).detach()
    loss_rota_delta = criterion(gt[..., 3:7], out[..., 3:7]).detach()
    loss_grip_close = criterion(gt[..., 7:8], out[..., 7:8]).detach()
    loss_term = criterion(gt[..., 8:].to(torch.float32), out[..., 8:].to(torch.float32)).detach()

    return loss, loss_wv, loss_rota_delta, loss_grip_close, loss_term


def calc_terminate_recall(labels, outputs):

    labels = ~(labels[:, :, -1].to(torch.bool))
    outputs = ~(outputs[:, :, -1].to(torch.bool))

    TP = ((labels == outputs) & (outputs)).sum()
    TP_and_FN = labels.sum()
    return TP, TP_and_FN


def Check_Wrong_tokens(labels, pred):

    wrong_map = ~(labels == pred)
    wrong_indices = torch.nonzero(wrong_map)
    print(wrong_indices, flush=True)
    return


def calc_acc_and_reduce(pred, label):
    acc = (pred == label).sum() / (label.numel())
    torch.distributed.all_reduce(acc, op=torch.distributed.ReduceOp.AVG)
    return acc


def reduce_and_average(data):
    torch.distributed.all_reduce(data, op=torch.distributed.ReduceOp.AVG)
    return data

def reduce_and_sum(data):
    torch.distributed.all_reduce(data, op=torch.distributed.ReduceOp.SUM)
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

    # return [{"params": no_decay, "weight_decay": 0.0}, {"params": decay, "weight_decay": weight_decay}]
    return [
        {"params": no_decay, "weight_decay": 0.0, "lr": lr},
        {"params": decay, "weight_decay": weight_decay, "lr": lr},
        {"params": pretrained_no_decay, "weight_decay": 0.0, "lr": lr * lr_mult},
        {"params": pretrained_decay, "weight_decay": weight_decay, "lr": lr * lr_mult},
    ]


@hydra.main(version_base=None, config_path=os.path.join(os.path.dirname(__file__), "..", "config"), config_name="config_diffusion_openx")
def train(cfg: DictConfig):

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


    args = argparse.Namespace()
    print(args)
    init_distributed_mode(args, cfg)
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    print(cfg.dataset.data_path)
    @dataclass
    class DataAdapterForOpenx:        
        def __call__(self, rlds_batch ):
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
            return dict(pixel_values=pixel_values, action=action, state=state, dataset_name=dataset_name, language_instruction= lang, loss_weight=loss_weight)
    data_path = 's3://openx'

    shuffle_buffer_size = cfg.shuffle_buffer_size

    
    # oxe_magic_soup_plus_minus
    vla_dataset_openx = RLDSDataset(data_path, cfg.dataname, DataAdapterForOpenx(), resize_resolution=(224, 224), shuffle_buffer_size=shuffle_buffer_size, train=True, image_aug=cfg.image_aug if 'image_aug' in cfg else False,
        window_size= cfg.dataset.traj_length + 1 - cfg.num_pred_action, 
        future_action_window_size= cfg.num_pred_action-1, batch_size=cfg.batch_size, 
        )

    cur_ = time.time()
    ii = 0

    train_dataloader = vla_dataset_openx

    DEVICE = "cuda:" + str(os.environ["LOCAL_RANK"]) if torch.cuda.is_available() else "cpu"
    # DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        attn_implementation=cfg.attn_implementation,
        use_action_head_diff=cfg.use_action_head_diff,

    )
    from transformers import AutoTokenizer, CLIPModel
    clip_tokenizer = AutoTokenizer.from_pretrained(
        "/xxx/xxx/share_data/Anonymous/clip-vit-large-patch14/", use_fast=False
    )
    clipmodel = CLIPModel.from_pretrained("/xxx/xxx/share_data/Anonymous/clip-vit-large-patch14/").to(DEVICE)

    params = param_groups_weight_decay(network, lr=cfg.lr, weight_decay=0.05, lr_mult=0.1, pretrained_weight_list=("image_tokenizer.tokenizer",))

    # import ipdb; ipdb.set_trace()
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

    start_epoch = 0
    total_iter_num = 0
    start_epoch, total_iter_num, checkpoint_path, tensorboard_path, log_path, run_dir = resume_or_load_checkpoint(cfg, network, optimizer, None)  
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(DEVICE)

    # network = network.to(DEVICE)
    network = torch.nn.parallel.DistributedDataParallel(network.cuda(local_rank), device_ids=[local_rank], find_unused_parameters=False)



    criterion = torch.nn.CrossEntropyLoss()
    L2_loss = torch.nn.MSELoss()
    loss_scaler = NativeScaler()


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
    import signal, sys

    for epoch in range(start_epoch, cfg.epoch):

        running_loss = 0.0
        data_start_time = time.time()

        iter_start_time = time.time()
        
        try:
            print('begin')
            for i, batch in enumerate(train_dataloader):
            # for i in range(1000):

                optimizer.zero_grad()

                # batch_new = dict_to_gpu(batch, DEVICE)
                
                obs_new = {}
                obs_new['language_instruction'] = batch['language_instruction']
                
                obs_new['image'] = batch['pixel_values'].to(DEVICE).clone()

                # print('load_proprio' in cfg)
                if 'load_proprio' in cfg:
                    # import ipdb;ipdb.set_trace()
                    actions = batch["action"][:, :].to(DEVICE).clone()
                    obs_new['poses'] = batch['state'][:, :1, :]
                else:
                    actions = batch["action"].to(DEVICE).clone()

                train_start_time = time.time()
                
                # loss_mask = torch.sum(act_new['loss_weight'], dim=-1) != 0
                loss_mask = batch['loss_weight'].to(DEVICE).clone()

                trajectory = actions
                # if rank == 0 and i % 10 == 0:
                #     print(actions[0, :])

                noise = torch.randn(trajectory.shape, device=trajectory.device)
                bsz = trajectory.shape[0]

                timesteps = torch.randint(0, network.module.noise_scheduler.config.num_train_timesteps, (bsz,), device=trajectory.device).long()

                noisy_trajectory = network.module.noise_scheduler.add_noise(trajectory, noise, timesteps)


                inputs = clip_tokenizer(text=obs_new['language_instruction'], return_tensors="pt", max_length=77, padding="max_length", truncation=True)
                for key in inputs:
                    inputs[key] = inputs[key].to(DEVICE)

                ccontext = clipmodel.text_model(**inputs)[0].squeeze(0).detach()
                ccontext = ccontext[:, None,...].repeat(1, obs_new['image'].shape[1], 1, 1)
                obs_new['natural_language_embedding'] = ccontext

                with torch.cuda.amp.autocast():
                    pred = network(obs_new, None, noisy_action_tokens=noisy_trajectory,timesteps=timesteps, num_pred_action=cfg.num_pred_action,)
                    if network.module.noise_scheduler.config.prediction_type == 'epsilon':
                        target = noise
                        if 'no_abs_diff' in cfg and cfg.no_abs_diff:
                            target = torch.cat([trajectory[:, :1], noise[:, 1:]], dim=1)
                    elif network.module.noise_scheduler.config.prediction_type == 'sample':
                        target = trajectory
                        pass
                    elif network.module.noise_scheduler.config.prediction_type == 'v_prediction':
                        target = network.module.noise_scheduler.get_velocity(trajectory, noise, timesteps)
                        pass
                    else:
                        raise ValueError(f"Unsupported prediction type {network.module.noise_scheduler.config.prediction_type}")
                    b, num, dim = pred.shape
                    # import ipdb;ipdb.set_trace()
                    logits = pred
                    loss = F.mse_loss(logits[...,:,:], target[..., :,:, ], reduction='none')
                    orig_loss = loss

                    running_loss = loss.detach()
                    from einops import rearrange, reduce
                    loss = loss[loss_mask]
                    loss = reduce(loss, 'b ... -> b (...)', 'mean')
                    extra_loss = reduce(orig_loss[:, :-cfg.dataset.traj_length], 'b ... -> b (...)', 'mean').detach()
                    # import ipdb;ipdb.set_trace()
                    
                    loss = loss.mean()
                loss_scaler(loss, optimizer)

                loss_rota = running_loss[...,3:6].mean()
                loss_world_vector = running_loss[...,:3].mean()
                loss_grip_close = running_loss[...,6:8].mean()
                # loss_terminate = running_loss[...,8:].mean()
                
                running_loss = running_loss.mean()
                
                
                extra_loss_rota = extra_loss[...,3:6].mean()
                extra_loss_trans = extra_loss[...,:3].mean()

                running_loss = reduce_and_average(running_loss)

                extra_loss_trans = reduce_and_average(extra_loss_trans)
                extra_loss_rota = reduce_and_average(extra_loss_rota)

                loss_rota = reduce_and_average(loss_rota)
                loss_world_vector = reduce_and_average(loss_world_vector)
                loss_grip_close = reduce_and_average(loss_grip_close)

                scheduler.step_update(epoch * len(train_dataloader) + i)

                
                
                
                # data_time = iter_time - train_time

                if rank == 0:
                    if i % 10 == 0:
                        iter_end_time = time.time()
                        iter_time = (iter_end_time - iter_start_time) / 10
                        train_time = (iter_end_time - train_start_time)
                        iter_start_time = time.time()
                        print("[epoch {}, iter {}, iter_time {}, train_time {}, ] lr: {} loss: {}, ext_vec: {}, extra_rota: {}, world_vector:{}, rota:{}, grip:{}, ".
                            format(epoch + 1, i + 1, iter_time,train_time, optimizer.param_groups[0]["lr"], running_loss, extra_loss_trans, extra_loss_rota, loss_world_vector, loss_rota, loss_grip_close), flush=True)

                    if writer is not None:
                        writer.add_scalar("MSE_loss", running_loss, total_iter_num)
                        writer.add_scalar("MSE_loss_rota", loss_rota, total_iter_num)
                        writer.add_scalar("MSE_loss_world_vector", loss_world_vector, total_iter_num)
                        writer.add_scalar("MSE_loss_grip_close", loss_grip_close, total_iter_num)
                        # writer.add_scalar("MSE_loss_terminate", loss_terminate, total_iter_num)
                        
                        writer.add_scalar("MSE_loss_extra_rota", extra_loss_rota, total_iter_num)
                        writer.add_scalar("MSE_loss_extra_world_vector", extra_loss_trans, total_iter_num)
                        
                # running_loss = 0.0
                sys.stdout.flush()

                if (
                    rank == 0
                    and total_iter_num != 0
                    and (total_iter_num % 1000 == 0) and 'no_checkpoints' not in cfg
                ):
                    checkpoint = {
                        "parameter": network.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": epoch,
                        "iter": i,
                        "total_iter_num": total_iter_num,
                        "loss": running_loss,
                        "loss_scaler": loss_scaler.state_dict(),
                    }
                    print("save checkpoint!", os.path.join(checkpoint_path, f"ckpt_{total_iter_num}.pth"))
                    torch.save(checkpoint, os.path.join(checkpoint_path, f"ckpt_{total_iter_num}.pth"))
                    for item in sorted(os.listdir(checkpoint_path), key=lambda x: os.path.getmtime(os.path.join(checkpoint_path, x)))[:-1]:
                        if item.endswith('pth'):
                            os.system('rm {}'.format(os.path.join(checkpoint_path, item)))
                    

                # *******************************************************#
                # EVAL PART

                total_iter_num += 1
                data_start_time = time.time()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(e)
            pass
    checkpoint = {
        "parameter": network.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        # "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "iter": i,
        "total_iter_num": total_iter_num,
        "loss": running_loss,
        "loss_scaler": loss_scaler.state_dict(),
    }
    print("save checkpoint!", os.path.join(checkpoint_path, f"ckpt_{total_iter_num}.pth"))
    for item in sorted(os.listdir(checkpoint_path), key=lambda x: os.path.getmtime(os.path.join(checkpoint_path, x)))[:]:
        os.system('rm {}'.format(os.path.join(checkpoint_path, item)))
    torch.save(checkpoint, os.path.join(checkpoint_path, f"ckpt_{total_iter_num}.pth"))


if __name__ == "__main__":
    SLURM_STEP_NODELIST = os.environ['SLURM_STEP_NODELIST']
    import subprocess
    output = subprocess.check_output("scontrol show hostname {} | head -n1".format(SLURM_STEP_NODELIST), shell=True)
    os.environ['MASTER_ADDR'] = output.strip().decode('ascii')
    train()
