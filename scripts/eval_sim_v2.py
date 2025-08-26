import argparse
import datetime
from genericpath import isdir
import logging
import os
import subprocess
import sys
import time

current_path = os.getcwd()
sys.path.append(current_path)
sys.path.append(os.path.join(current_path, "rt1_pytorch/openx_utils/"))
sys.path.append(os.path.join(current_path, "../embodied_foundation/rt1_pytorch"))
# export PYTHONPATH="$(pwd)":"$(pwd)/rt1_pytorch/openx_utils/":"$(pwd)/../":"$(pwd)/../embodied_foundation/rt1_pytorch":$PYTHONPATH

import hydra
import torch
from Dataset_HF.utils import get_action_spec
from Dataset_Sim.SimDataset import SimDataset
from RT1_llama_dp import RT1Net
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import importlib
from torch.utils.data import DataLoader

from diffusion.normalizer import LinearNormalizer


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MS2_RENDERER_LOG_LEVEL"] = "error"
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



def dict_to_gpu(dict, DEVICE):

    gpu_dict = {}
    for k in dict:
        if k == "camera_extrinsic_cv":
            continue
        b, sample_per_episode = dict[k].shape[:2]
        gpu_dict[k] = dict[k].reshape(b * sample_per_episode, *dict[k].shape[2:]).to(DEVICE, non_blocking=True)
        # if k == 'image':
        #     gpu_dict[k] = gpu_dict[k].permute(0,1,3,4,2).contiguous()
    return gpu_dict

def unnormalize(x):
    x = x.clone()
    for i in range(3):
        x[..., i] = x[..., i] * IMAGENET_DEFAULT_STD[i] + IMAGENET_DEFAULT_MEAN[i]

    return x



def get_args_parser():

    parser = argparse.ArgumentParser()
    # parser.add_argument("--sequence_length", default=15, type=int)
    # parser.add_argument("--traj_per_episode", default=8, type=int)
    # parser.add_argument("--num_action_tokens", default=11, type=int)
    # parser.add_argument("--num_pred_action", default=1, type=int)
    # parser.add_argument("--abs_pose", default=0, type=int)
    # parser.add_argument("--prediction_type", default='epsilon', type=str)
    # parser.add_argument("--layer_size", default=256, type=int)
    # parser.add_argument("--vocab_size", default=768, type=int)
    # parser.add_argument("--num_image_tokens", default=81, type=int)
    # parser.add_argument("--batch_size", default=8, type=int)
    # parser.add_argument("--epoch", default=5000000, type=int)
    # parser.add_argument("--num_layers", default=12, type=int)
    # parser.add_argument("--output_dir", default="./output", type=str)
    # parser.add_argument("--iter_per_epoch", default=36000, type=int)
    # parser.add_argument("--world_size", default=8, type=int)
    # parser.add_argument("--data_path", default="/xxx/xxx/share_data/Anonymous/Dataset/Sim_Data/PickCube-v0-0321", type=str)
    # parser.add_argument("--task_name", default="test", type=str)
    # parser.add_argument("--ckpt_path", default=None, type=str)
    # parser.add_argument("--dataset_split_type", default="fix_traj", type=str)
    # parser.add_argument("--use_baseframe_action", action="store_true")
    # parser.set_defaults(use_baseframe_action=False)
    # parser.add_argument("--tensorboard_output_dir", default="tensorboard/embodied", type=str)
    return parser


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


def init_distributed_mode(args, cfg):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = rank % torch.cuda.device_count()

        world_size = int(os.environ["SLURM_NTASKS"])

        args.rank = rank
        args.gpu = local_rank
        args.local_rank = local_rank
        args.world_size = world_size

        try:
            local_size = int(os.environ["SLURM_NTASKS_PER_NODE"])
        except:
            local_size = int(os.environ.get("LOCAL_SIZE", 1))

        if "MASTER_PORT" not in os.environ:
            port = 22110

            print(f"MASTER_PORT = {port}")
            os.environ["MASTER_PORT"] = str(port)

            time.sleep(3)

        node_list = os.environ["SLURM_STEP_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr

        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["LOCAL_SIZE"] = str(local_size)
        os.environ["LOCAL_WORLD_SIZE"] = str(local_size)
        os.environ["WORLD_SIZE"] = str(world_size)

    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(local_rank)
    args.dist_backend = "nccl"
    print("| distributed init (rank {})".format(rank), flush=True)
    dist_backend = "nccl"
    print(HydraConfig.get(), )
    # init_method = os.path.join(HydraConfig.get().runtime.cwd, cfg.task_name, "initial_method.txt")
    init_method = os.path.join(HydraConfig.get().runtime.cwd, HydraConfig.get().run.dir, "initial_method.txt")
    # init_method = os.path.join('/xxx/xxx/Anonymous/Code/embodied_foundation', cfg.task_name+"initial_method.txt")
    # print(init_method,int(os.environ["WORLD_SIZE"]), int(os.environ["RANK"]), args)
    torch.distributed.init_process_group(
        backend=dist_backend,  # init_method=args.dist_url,
        init_method=f"file://{init_method}",
        world_size=int(os.environ["WORLD_SIZE"]),
        rank=int(os.environ["RANK"]),
    )
    torch.distributed.barrier()
    print(torch.distributed.get_world_size())
    setup_for_distributed(rank == 0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        flush = kwargs.pop("flush", True)
        if is_master or force:
            builtin_print(*args, **kwargs, flush=flush)

    __builtin__.print = print


def calc_l2_loss(act, output, camera_extrinsic_cv):

    gt_list = []
    out_list = []
    cam2world = Transform3d(matrix=camera_extrinsic_cv.mT).inverse()

    def get_world_translation_rotation(output):

        translation_delta = output["world_vector"].flatten(0, 1)
        rotation_delta = output["rotation_delta"].flatten(0, 1)
        rotation_delta[..., 0] += 1.0

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


@hydra.main(version_base=None, config_path=os.path.join(os.path.dirname(__file__), "..", "config"), config_name="config_diffusion_nowrist")
def train(cfg: DictConfig):

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


    # parser = get_args_parser()
    # args = parser.parse_args()

    # args = None
    args = argparse.Namespace()
    print(args)
    init_distributed_mode(args, cfg)
    # args.world_size = 1
    # args.rank = 0
    
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    print(cfg.dataset.data_path)
    # train_dataset = SimDataset(
    #     data_path=cfg.dataset.data_path,
    #     language_embedding_path=cfg.dataset.language_embedding_path,
    #     dataset_type=0,
    #     use_baseframe_action=cfg.dataset.use_baseframe_action,
    #     split_type=cfg.dataset.split_type,
    #     traj_per_episode=cfg.dataset.traj_per_episode,
    #     traj_length=cfg.dataset.traj_length,
    #     data_cam_list=cfg.dataset.train_data_list,
    #     stride=cfg.dataset.stride,
    # )  # dataset_type 0 for train and 1 for eval
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # train_dataloader = DataLoader(
    #     train_dataset,
    #     batch_size=cfg.batch_size,
    #     sampler=train_sampler,
    #     drop_last=True,
    #     num_workers=8,
    #     prefetch_factor=4,
    #     pin_memory=True,
    #     persistent_workers=True,
    # )

    # eval_dataset = SimDataset(
    #     data_path=cfg.dataset.data_path,
    #     language_embedding_path=cfg.dataset.language_embedding_path,
    #     dataset_type=1,
    #     use_baseframe_action=cfg.dataset.use_baseframe_action,
    #     split_type=cfg.dataset.split_type,
    #     traj_per_episode=2,
    #     traj_length=cfg.dataset.traj_length,
    #     data_cam_list=cfg.dataset.eval_data_list,
    #     stride=cfg.dataset.stride,
    # )  # dataset_type 0 for train and 1 for eval
    # eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset)
    # eval_dataloader = DataLoader(
    #     eval_dataset,
    #     batch_size=cfg.batch_size,
    #     sampler=eval_sampler,
    #     drop_last=True,
    #     num_workers=8,
    #     prefetch_factor=4,
    #     pin_memory=True,
    #     persistent_workers=True,
    # )

    DEVICE = "cuda:" + str(os.environ["LOCAL_RANK"]) if torch.cuda.is_available() else "cpu"
    # DEVICE = 
    # DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    action_spec = get_action_spec(cfg.action_spec, DEVICE)

    network = RT1Net(
        output_tensor_spec=action_spec,
        vocab_size=cfg.model.vocab_size,
        time_sequence_length=cfg.model.time_sequence_length,
        num_layers=cfg.model.num_layers,
        dropout_rate=cfg.model.dropout_rate,
        include_prev_timesteps_actions=cfg.model.include_prev_timesteps_actions,
        freeze_backbone=cfg.model.freeze_backbone,
        use_qformer=cfg.model.use_qformer,
        use_wrist_img=cfg.model.use_wrist_img,
        use_depth_img=cfg.model.use_depth_img,
        prediction_type=cfg.prediction_type,
        dim_align_type=cfg.dim_align_type if 'dim_align_type' in cfg else 0,
    )
    print('create model')
    


    start_epoch = 0
    total_iter_num = 0
    # current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    # if 'ckpt_path' in cfg and cfg.ckpt_path =='auto':
    #     current_time = 'auto'
    # tensorboard_output_path = os.path.join(cfg.tensorboard_output_dir, cfg.task_name + "_" + current_time)
    # checkpoint_path = os.path.join(tensorboard_output_path, "checkpoints")
    # tensorboard_path = os.path.join(tensorboard_output_path, "tensorboard")
    # log_path = os.path.join(tensorboard_output_path, "output.log") 
    
    if "ckpt_path" in cfg and cfg.ckpt_path != "None":
        ckpt_path = cfg.ckpt_path
       
        print('load ', cfg.ckpt_path)
        if os.path.isdir(ckpt_path):
            ckpt_path = os.path.join(ckpt_path, sorted(os.listdir(ckpt_path), key=lambda x: os.path.getmtime(os.path.join(ckpt_path, x)))[-1])
        if os.path.exists(ckpt_path):
            print('load ', ckpt_path)
            ckpt = torch.load(ckpt_path, 'cpu')
            print(network.load_state_dict(ckpt["parameter"]))
        # optimizer.load_state_dict(ckpt["optimizer"])
        # scheduler.load_state_dict(ckpt["scheduler"])

        # start_epoch = ckpt["epoch"]
        # total_iter_num = ckpt["total_iter_num"] + 1
            
            # current_time = ckpt_path.split(cfg.task_name)[1][1:].split('/')[0]
            # if 'ckpt_path' in cfg and cfg.ckpt_path =='auto':
            #     current_time = 'auto'
            
            # tensorboard_output_path = os.path.join(cfg.tensorboard_output_dir, cfg.task_name + "_" + current_time)
            # checkpoint_path = os.path.join(tensorboard_output_path, "checkpoints")
            # tensorboard_path = os.path.join(tensorboard_output_path, "tensorboard")
            # log_path = os.path.join(tensorboard_output_path, "output.log") 
# tensorboard/embodied/diff_epred_policy_20240415220918/checkpoints/ckpt_12000.pth
    # for state in optimizer.state.values():
    #     for k, v in state.items():
    #         if isinstance(v, torch.Tensor):
    #             state[k] = v.to(DEVICE)

    # network = network.to(DEVICE)
    # network = network.cuda(local_rank)
    network = torch.nn.parallel.DistributedDataParallel(network.cuda(local_rank), device_ids=[local_rank], find_unused_parameters=False)

# #***********************************debug
    close_loop_eval = getattr(importlib.import_module("close_loop_eval_diffusion"), "close_loop_eval_v2")
    close_loop_eval_start_time = time.time()
    success_num, total_num, total_success_rate = close_loop_eval(
                    model=network,
                    test_episodes_num=cfg.close_loop_eval.test_episodes_num,
                    eval_data_list=cfg.dataset.close_loop_eval_data_list,
                    args=args,
                    rand_seed=25000,
                    stride=cfg.dataset.stride,
                    camera_coord = not cfg.dataset.use_baseframe_action,
                    root_folder = os.path.join(HydraConfig.get().runtime.cwd, HydraConfig.get().run.dir, "close_loop_videos", f"{total_iter_num}_iters"),
                    # root_folder = '/xxx/xxx/Anonymous/Code/embodied_foundation/outputs/2024-06-06/19-40-53/close_loop_videos/0_iters',
                    data_root_path = cfg.dataset.data_path,
                    cfg = cfg,
                )
    total_success_rate = reduce_and_average(torch.tensor(total_success_rate, device=DEVICE))
    for k in success_num:
        success_num[k] = reduce_and_sum(torch.tensor(success_num[k], device = DEVICE))
    # for k in total_num:
    #     total_num[k] = reduce_and_sum(torch.tensor(total_num[k], device = DEVICE))
    print('rate:', total_success_rate, 0, success_num)
    # #***********************************debug
    sys.exit()



if __name__ == "__main__":

    train()
