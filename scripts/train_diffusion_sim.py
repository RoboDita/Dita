import argparse
import datetime
from genericpath import isdir
import logging
import os
import subprocess
import sys
import time


from Dataset_VLA.calvin_dataset import CalvinDataset_Policy
from Dataset_VLA.dataset_calvin import CalvinDataset
from utils import resume_or_load_checkpoint
from utils.data_utils import add_noise_to_euler, add_noise_to_quaternion, add_noise_to_translation
from utils.ddp_utils import init_distributed_mode
from functools import partial
current_path = os.getcwd()
sys.path.append(current_path)
sys.path.append(os.path.join(current_path, "utils/"))
sys.path.append(os.path.join(current_path, "../scripts"))
sys.path.append(os.path.join(current_path, "../openvla"))
# export PYTHONPATH="$(pwd)":"$(pwd)/rt1_pytorch/openx_utils/":"$(pwd)/../":"$(pwd)/../embodied_foundation/rt1_pytorch":$PYTHONPATH




import hydra
import torch
from Dataset_Sim.SimDataset import SimDataset

from llama_dp import RobotTransformerNet
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import importlib
from torch.utils.data import DataLoader



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
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from timm.utils import NativeScaler
from torch.utils.tensorboard import SummaryWriter
import math

def dict_to_gpu(dict, DEVICE):

    gpu_dict = {}
    for k in dict:
        if k == "camera_extrinsic_cv":
            continue
        if k == 'language_instruction':
            continue
        b, sample_per_episode = dict[k].shape[:2]
        gpu_dict[k] = dict[k].reshape(b * sample_per_episode, *dict[k].shape[2:]).to(DEVICE, non_blocking=True)
        # if k == 'image':
        #     gpu_dict[k] = gpu_dict[k].permute(0,1,3,4,2).contiguous()
    return gpu_dict

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

def get_parameter_num(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def unnormalize(x):
    x = x.clone()
    for i in range(3):
        x[..., i] = x[..., i] * IMAGENET_DEFAULT_STD[i] + IMAGENET_DEFAULT_MEAN[i]

    return x



def get_args_parser():

    parser = argparse.ArgumentParser()
    return parser


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



def evaluate_act(network, eval_dataloader, DEVICE, tokens_per_context_image, 
                 tokens_per_step, criterion, writer, args, total_iter_num, noise_scheduler,
                 generator, num_pred_action, cfg, clipmodel=None, clip_tokenizer=None):
    with torch.no_grad():
        network.eval()
        eval_loss_total = 0.0
        eval_loss_per_iter = 0.0

        acc_total = 0.0
        loss_l2_total = 0.0
        TP_total = 0.0
        TP_and_FN_total = 0.0

        mse_loss_world_vector_total = 0.0
        mse_loss_rotation_delta_total = 0.0
        acc_gripper_closedness_total = 0.0
        acc_terminate_total = 0.0

        loss_wv_l2_total = 0.0
        loss_rota_delta_l2_total = 0.0
        loss_grip_close_l2_total = 0.0
        loss_term_l2_total = 0.0
        i = 0
        gripper_idx = 7
        if cfg.trajectory_dim == 7:
            gripper_idx = 6
        for i, batch in enumerate(eval_dataloader):
            if i % 50 == 0:
                print(f"{i}/{len(eval_dataloader)}")

            obs_dict = batch["observation"]
            act_dict = batch["action"]
            obs_new = dict_to_gpu(obs_dict, DEVICE)
            act_new = dict_to_gpu(act_dict, DEVICE)
            camera_extrinsic_cv = batch["observation"]["camera_extrinsic_cv"].flatten(0, 2).to(DEVICE, non_blocking=True)
            if cfg.dataset.use_baseframe_action:
                camera_extrinsic_cv = torch.eye(4).to(DEVICE).unsqueeze(0).repeat(camera_extrinsic_cv.shape[0], 1, 1)



            trajectory_gt = get_gt_trajectory(cfg, act_new)

            # set_obs_poses(cfg, obs_new, act_new)
            cond_data = torch.zeros(size=(len(trajectory_gt), trajectory_gt.shape[1], trajectory_gt.shape[-1]), device=DEVICE, dtype=trajectory_gt.dtype)

            if cfg.dataname in ['rlbench', 'calvin', 'calvin_mc', 'metaworld']:
                inputs = clip_tokenizer(text=batch['instruction'], return_tensors="pt", max_length=77, padding="max_length", truncation=True)
                for key in inputs:
                    inputs[key] = inputs[key].to(DEVICE)

                ccontext = clipmodel.text_model(**inputs)[0].squeeze(0).detach()
                ccontext = ccontext[:, None,...].repeat(batch["observation"]['image'].shape[1], obs_new['image'].shape[1], 1, 1)

                obs_new['natural_language_embedding'] = ccontext            
            
            trajectory = torch.randn(
                size=cond_data.shape, 
                dtype=cond_data.dtype,
                device=cond_data.device,
                generator=None)

            if cfg.dataname in ['calvin', 'rlbench', 'calvin_mc']:
                if cfg.use_action_head_diff in [2, 4, 5]:
                    if hasattr(network, 'module'):
                        model_output = network_module.inference_withfeats(obs_new, num_pred_action=num_pred_action,cfg=0)
                    else:
                        model_output = network.inference_withfeats(obs_new, num_pred_action=num_pred_action,cfg=0)
                else:
                    if hasattr(network, 'module'):
                        model_output = network_module.inference(obs_new, num_pred_action=num_pred_action, abs_pose=1, cfg=0) # abs_pose here indicates we do not scale the data.
                    else:
                        model_output = network.inference(obs_new, num_pred_action=num_pred_action, abs_pose=1, cfg=0)
                if cfg.dataname in ['rlbench']:
                    #  'gripper_closedness_action': trajectory[...,7:8],
                    # 'terminate_episode': trajectory[...,8:]
                    output = torch.cat([model_output["world_vector"].cpu(), model_output["rotation_delta"].cpu(),model_output["gripper_closedness_action"].cpu(), model_output["terminate_episode"].cpu(),], dim=-1).cuda()
                else:
                    output = torch.cat([model_output["world_vector"].cpu(), model_output["rotation_delta"].cpu(),model_output["gripper_closedness_action"].cpu(),], dim=-1).cuda()
                trajectory = output
            else:
                for t in noise_scheduler.timesteps:

                    model_output = network(obs_new, act_new, noisy_action_tokens=trajectory, 
                                        timesteps=t.repeat(len(trajectory)), num_pred_action=num_pred_action)
                    # model_output = network(obs_new, noisy_trajectory=trajectory, timesteps=t)
                    b, num, dim = model_output.shape
                    trajectory = noise_scheduler.step(model_output, t, 
                                                        trajectory[...,:,:], generator=generator,).prev_sample
                    
            


                trajectory = trajectory[:, -num_pred_action:]
            trajectory_gt = trajectory_gt[:, -num_pred_action:]
            
            eval_loss_per_iter = reduce_and_average(F.mse_loss(trajectory[:, :, :3], trajectory_gt[:, :, :3]))
            mse_world_vector = reduce_and_average(F.mse_loss(trajectory[:, :, :3], trajectory_gt[:, :, :3]))
            mse_rotation_delta = reduce_and_average(F.mse_loss(trajectory[:, :, 3:gripper_idx], trajectory_gt[:, :, 3:gripper_idx]))
            acc_gripper_closedness = calc_acc_and_reduce((trajectory[:, :, gripper_idx] > 0.).float() * 2 - 1, trajectory_gt[:, :, gripper_idx])
            mse_terminate = reduce_and_average(F.mse_loss(trajectory[:, :, -1] > 0.5, trajectory_gt[:, :, -1]))

            eval_loss_total += eval_loss_per_iter


            mse_loss_world_vector_total += mse_world_vector
            mse_loss_rotation_delta_total += mse_rotation_delta
            acc_gripper_closedness_total += acc_gripper_closedness
            acc_terminate_total += mse_terminate

            eval_loss_per_iter = 0
            if i > 30:
                break

        eval_loss_total /= i + 1
        acc_total /= i + 1
        loss_l2_total /= i + 1
        TP_total /= i + 1
        TP_and_FN_total /= i + 1

        mse_loss_world_vector_total /= i + 1
        mse_loss_rotation_delta_total /= i + 1
        acc_gripper_closedness_total /= i + 1
        acc_terminate_total /= i + 1

        # loss_wv_l2_total /= i + 1
        # loss_grip_close_l2_total /= i + 1
        # loss_rota_delta_l2_total /= i + 1
        # loss_term_l2_total /= i + 1
        print(args.rank)
        if args.rank == 0:

            print(
                "EVAL: loss_total: {}, world_vector: {}, rotation_delta: {}, gripper_closedness: {}, Terminate_recall: {}".format(
                    eval_loss_total, mse_loss_world_vector_total, mse_loss_rotation_delta_total, acc_gripper_closedness_total, acc_terminate_total
                )
            )
            logging.info(
                "EVAL: loss_total: {}, world_vector: {}, rotation_delta: {}, gripper_closedness: {}, Terminate_recall: {}".format(
                    acc_total, mse_loss_world_vector_total, mse_loss_rotation_delta_total, acc_gripper_closedness_total, acc_terminate_total
                )
            )
            if writer is not None:
                # writer.add_scalar("EVAL_MSE_loss", acc_total, total_iter_num)
                # writer.add_scalar("EVAL_MSELoss", loss_l2, total_iter_num)
                writer.add_scalar("EVAL_CrossEntropyLoss", eval_loss_total, total_iter_num)
                writer.add_scalar("EVAL_MSE_WorldVector", mse_loss_world_vector_total, total_iter_num)
                writer.add_scalar("EVAL_MSE_RotationDelta", mse_loss_rotation_delta_total, total_iter_num)
                writer.add_scalar("EVAL_Acc_GripperClosed", acc_gripper_closedness_total, total_iter_num)
                writer.add_scalar("EVAL_Acc_Terminate", acc_terminate_total, total_iter_num)
    
    pass


@hydra.main(version_base=None, config_path=os.path.join(os.path.dirname(__file__), "..", "config"), config_name="config_diffusion_nowrist")
def train(cfg: DictConfig):

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


    # parser = get_args_parser()
    # args = parser.parse_args()
    # print(args)
    # args = None
    args = argparse.Namespace()
    print(args)
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
    if cfg.dataname == 'maniskill':
        print(cfg.dataname)
        train_dataset = SimDataset(
            data_path=cfg.dataset.data_path,
            language_embedding_path=cfg.dataset.language_embedding_path,
            dataset_type=0,
            use_baseframe_action=cfg.dataset.use_baseframe_action,
            split_type=cfg.dataset.split_type,
            traj_per_episode=cfg.dataset.traj_per_episode,
            traj_length=cfg.dataset.traj_length,
            data_cam_list=cfg.dataset.train_data_list,
            stride=cfg.dataset.stride,
            num_given_observation= cfg.dataset.num_given_observation,
            include_target=cfg.dataset.include_target,
            aug_gripper_status_pose=cfg.dataset.aug_gripper_status_pose if 'aug_gripper_status_pose' in cfg.dataset else 0,
            use_euler=cfg.dataset.use_euler,
            
        )  # dataset_type 0 for train and 1 for eval

        eval_dataset = SimDataset(
            data_path=cfg.dataset.data_path,
            language_embedding_path=cfg.dataset.language_embedding_path,
            dataset_type=1,
            use_baseframe_action=cfg.dataset.use_baseframe_action,
            split_type=cfg.dataset.split_type,
            traj_per_episode=2,
            traj_length=cfg.dataset.traj_length,
            data_cam_list=cfg.dataset.eval_data_list,
            stride=cfg.dataset.stride,
            num_given_observation= cfg.dataset.num_given_observation,
            include_target=cfg.dataset.include_target,
            use_euler=cfg.dataset.use_euler,
        )  # dataset_type 0 for train and 1 for eval
    elif cfg.dataname == 'dumpy':
        from Dataset_Sim.SimDataset import SimDatasetDumpy 
        train_dataset = SimDatasetDumpy()
        eval_dataset = SimDatasetDumpy()
    elif cfg.dataname == 'calvin_mc':
        if cfg.wrap_grmg_data == 0:
            act_len = cfg.dataset.traj_length + cfg.num_pred_action - 1
            seq_len=cfg.dataset.num_given_observation
        elif cfg.wrap_grmg_data == 1:
            act_len = cfg.num_pred_action
            seq_len=cfg.dataset.num_given_observation
            seq_len=1
        elif cfg.wrap_grmg_data == 2:
            seq_len=cfg.dataset.num_given_observation
            seq_len=2
            act_len = cfg.num_pred_action
        train_dataset = CalvinDataset_Policy(
            's3://Anonymous/',
            seq_len=seq_len,
            act_len=act_len,
            forward_n_max=25,
            mode='train',
            subfolder=cfg.taskname,
            use_data_augmentation=False,
            use_play=False,
            task_num=10000,
            wrap_grmg_data=cfg.wrap_grmg_data)
        eval_dataset = CalvinDataset_Policy(
            's3://Anonymous/',
            seq_len=seq_len,
            act_len=act_len,
            forward_n_max=25,
            mode='validate',
            subfolder=cfg.taskname,
            use_data_augmentation=False,
            use_play=False,
            task_num=10000,
            wrap_grmg_data=cfg.wrap_grmg_data)
    elif cfg.dataname == 'calvin':
        # 
        train_dataset = CalvinDataset(
            root='s3://Anonymous/{}/training'.format(cfg.taskname),
            taskvar=[
                ("A", 0), ("B", 0), ("C", 0), ("D", 0),
            ],
            max_episode_length=cfg.dataset.traj_length,
            cache_size=0,
            max_episodes_per_task=-1,
            num_iters=600000,
            training=True,
            image_rescale=tuple(
                float(x) for x in "0.75,1.25".split(",")
            ),
            return_low_lvl_trajectory=False,
            dense_interpolation=1,
            interpolation_length=20,
            relative_action=bool(1)
        )
        eval_dataset = CalvinDataset(
            root='s3://Anonymous/{}/validation'.format(cfg.taskname),
            taskvar=[
                ("A", 0), ("B", 0), ("C", 0), ("D", 0),
            ],
            max_episode_length=cfg.dataset.traj_length,
            cache_size=0,
            max_episodes_per_task=-1,
            num_iters=600000,
            training=True,
            image_rescale=tuple(
                float(x) for x in "0.75,1.25".split(",")
            ),
            return_low_lvl_trajectory=False,
            dense_interpolation=1,
            interpolation_length=20,
            relative_action=bool(1)
        )
        if 'DEBUG' in os.environ and int(os.environ['DEBUG']) == 2:
            eval_dataset = train_dataset    
        print('use calvin')
    else:
        raise("dataset not supported")

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset)
    else:
        train_sampler = None
        eval_sampler = None
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        drop_last=True,
        num_workers=8 if cfg.dataname== 'maniskill' else 16,
        prefetch_factor=4,
        pin_memory=True,
        persistent_workers=True,
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=cfg.batch_size,
        sampler=eval_sampler,
        drop_last=True,
        num_workers=8,
        prefetch_factor=4,
        pin_memory=True,
        persistent_workers=True,
    )

    if args.distributed:
        DEVICE = "cuda:" + str(os.environ["LOCAL_RANK"]) if torch.cuda.is_available() else "cpu"
    else:
        DEVICE = 'cuda'
    # DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    time_sequence_length = cfg.num_pred_action if 'wrap_grmg_data' in cfg and cfg.wrap_grmg_data == 1 else cfg.dataset.traj_length
    if 'wrap_grmg_data' in cfg and cfg.wrap_grmg_data == 2:
        time_sequence_length = cfg.num_pred_action + 1
    network = RobotTransformerNet(
        output_tensor_spec=None,
        input_size='('+str(cfg.input_size)+','+str(cfg.input_size)+')' if 'input_size' in cfg else None,
        vocab_size=cfg.model.vocab_size,
        # time_sequence_length=cfg.dataset.traj_length,
        time_sequence_length=time_sequence_length,
        num_layers=cfg.model.num_layers,
        dropout_rate=cfg.model.dropout_rate,
        include_prev_timesteps_actions=cfg.model.include_prev_timesteps_actions,
        freeze_backbone=cfg.model.freeze_backbone,
        use_qformer=cfg.model.use_qformer,
        use_wrist_img=cfg.model.use_wrist_img,
        use_depth_img=cfg.model.use_depth_img,
        prediction_type=cfg.prediction_type,
        dim_align_type=cfg.dim_align_type if 'dim_align_type' in cfg else 0,
        use_action_head_diff=cfg.use_action_head_diff,
        scheduler_type=cfg.scheduler_type,
        num_inference_steps=cfg.num_inference_steps,
        trajectory_dim=cfg.trajectory_dim,
        vit_forward_version = cfg.model.vit_forward_version if 'vit_forward_version' in cfg.model else None,
    )

    params = param_groups_weight_decay(network, lr=cfg.lr, weight_decay=0.05, lr_mult=0.1, pretrained_weight_list=("image_tokenizer.tokenizer",))

    # import ipdb; ipdb.set_trace()
    if cfg.optimizer.name == "adamw":
        optimizer = create_optimizer_v2(
            params, opt=cfg.optimizer.name, lr=cfg.lr, weight_decay=cfg.optimizer.weight_decay, betas=(cfg.optimizer.betas_0, cfg.optimizer.betas_1)
        )


    scheduler, _ = create_scheduler_v2(
        optimizer,
        sched=cfg.scheduler.sched,
        warmup_lr=cfg.scheduler.warmup_lr,
        warmup_epochs=cfg.scheduler.warmup_epochs,
        num_epochs=cfg.scheduler.num_epochs,
        decay_epochs=cfg.scheduler.decay_epochs,
        updates_per_epoch=len(train_dataloader),
        step_on_epochs=cfg.scheduler.step_on_epochs,
    )

    if "use_adjust_scheduler" in cfg and cfg.use_adjust_scheduler:

        print("use adjust scheduler!!!!!", flush = True)
        lr = cfg.lr

        iter_per_epoch = int(16000.0 / (cfg.batch_size / 64)  / int(os.environ["SLURM_NTASKS"])) 
        warm_up_steps = iter_per_epoch * cfg.scheduler.warmup_epochs
        lr_scheduler_configs = {
            'warmup_iters': warm_up_steps,
            'iters': cfg.epoch * iter_per_epoch,
            'min_lr_scale': cfg.min_lr_scale
        }
        lr_lambda = partial(adjust_learning_rate, configs=lr_scheduler_configs)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    start_epoch = 0
    total_iter_num = 0

    start_epoch, total_iter_num, checkpoint_path, tensorboard_path, log_path, run_dir = resume_or_load_checkpoint(cfg, network, optimizer, scheduler) 
# tensorboard/embodied/diff_epred_policy_20240415220918/checkpoints/ckpt_12000.pth
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(DEVICE)

    # network = network.to(DEVICE)
    if args.distributed:
        network = torch.nn.parallel.DistributedDataParallel(network.cuda(local_rank), device_ids=[local_rank], find_unused_parameters=False)
    else:
        network = network.to(DEVICE)

    
    if args.distributed:
        network_module = network.module
    else:
        network_module = network

    criterion = torch.nn.CrossEntropyLoss()
    L2_loss = torch.nn.MSELoss()
    loss_scaler = NativeScaler()
    clipmodel = None
    if cfg.dataname in ['metaworld', 'calvin', 'calvin_mc']:
        from transformers import AutoTokenizer, CLIPModel
        clip_tokenizer = AutoTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14", use_fast=False
        )
        clipmodel = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)

    writer = None

    if rank == 0:
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path, exist_ok=True)
        if cfg.task_name != "test":
            if not os.path.exists(tensorboard_path):
                os.makedirs(tensorboard_path, exist_ok=True)
            writer = SummaryWriter(tensorboard_path)
            print(tensorboard_path, 'tensorboard_path')
        
        print("Training!", flush=True)
    
        logging.basicConfig(filename=log_path+str(rank), level=logging.INFO, format="%(asctime)s : %(message)s")

    if 'eval_only' in cfg and cfg.eval_only:
        close_loop_eval_ddp(cfg, args, rank, eval_dataset, DEVICE, network, total_iter_num, run_dir, writer, )
        exit()
    network.train()
    import signal, sys

    gripper_idx = 7
    if cfg.trajectory_dim == 7:
        gripper_idx = 6

    for epoch in range(start_epoch, cfg.epoch):

        if args.distributed:
            train_sampler.set_epoch(epoch)

        running_loss = 0.0

        for i, batch in enumerate(train_dataloader):
            iter_start_time = time.time()

            optimizer.zero_grad()
            obs_dict = batch["observation"]
            act_dict = batch["action"]
            camera_extrinsic_cv = batch["observation"]["camera_extrinsic_cv"].flatten(0, 2).to(DEVICE, non_blocking=True)
            assert cfg.dataset.use_baseframe_action, cfg.dataset.use_baseframe_action
            if cfg.dataset.use_baseframe_action:
                camera_extrinsic_cv = torch.eye(4).to(DEVICE).unsqueeze(0).repeat(camera_extrinsic_cv.shape[0], 1, 1)
            obs_new = dict_to_gpu(obs_dict, DEVICE)
            act_new = dict_to_gpu(act_dict, DEVICE)

            train_start_time = time.time()
            
            
            # normalize 
            trajectory = get_gt_trajectory(cfg, act_new)
            
            loss_mask = torch.sum(act_new['loss_weight'], dim=-1) != 0  # for actions that are beyond input
            if cfg.abs_sup in [3, 4, 5]:
                if cfg.abs_sup == 4:
                    gripper_change_pose = act_new['gripper_first_change_pose']
                else:
                    gripper_change_pose = act_new['gripper_change_pose']
                gripper_change_pose[:, gripper_idx:] = torch.zeros_like(gripper_change_pose[:, gripper_idx:])

            noise = torch.randn(trajectory.shape, device=trajectory.device)
            bsz = trajectory.shape[0]

            timesteps = torch.randint(
                0, network_module.noise_scheduler.config.num_train_timesteps, 
                (bsz,), device=trajectory.device
            ).long()

            noisy_trajectory = network_module.noise_scheduler.add_noise(trajectory, noise, timesteps)
            if cfg.dataname in ['rlbench', 'calvin','calvin_mc', 'metaworld']:
                inputs = clip_tokenizer(text=batch['instruction'], return_tensors="pt", max_length=77, padding="max_length", truncation=True)
                for key in inputs:
                    inputs[key] = inputs[key].to(DEVICE)

                ccontext = clipmodel.text_model(**inputs)[0].squeeze(0).detach()
                ccontext = ccontext[:, None,...].repeat(batch["observation"]['image'].shape[1], obs_new['image'].shape[1], 1, 1)

                obs_new['natural_language_embedding'] = ccontext
            reg_token_nums = int(cfg.abs_sup > 0)
            pred = network(obs_new, act_new, noisy_action_tokens=noisy_trajectory,timesteps=timesteps, num_pred_action=cfg.num_pred_action,
                            aug_img_tokens=True if 'aug_img_tokens' in cfg and cfg.aug_img_tokens else False,
                            reg_token_nums=reg_token_nums)



            # (b,t,c,h,w)
            if network_module.noise_scheduler.config.prediction_type == 'epsilon':
                target = noise
                if 'no_abs_diff' in cfg and cfg.no_abs_diff:
                    target = torch.cat([trajectory[:, :1], noise[:, 1:]], dim=1)
            else:
                raise ValueError(f"Unsupported prediction type {network_module.noise_scheduler.config.prediction_type}")
            b, num, dim = pred.shape
            logits = pred[:, reg_token_nums:]
            

            start_loss_action_chunk_idx = 0

            loss_mask = loss_mask[...,start_loss_action_chunk_idx:]
            loss = F.mse_loss(logits[..., start_loss_action_chunk_idx:,:], target[..., start_loss_action_chunk_idx:,:, ], reduction='none')
            affordance_loss = 0.
        
            loss = loss[loss_mask]
            from einops import rearrange, reduce
            loss = reduce(loss, 'b ... -> b (...)', 'mean')
            running_loss = loss.detach()
            loss = loss.mean()
            loss_scaler(loss, optimizer)

            loss_rota = running_loss[...,3:gripper_idx].mean()
            loss_world_vector = running_loss[...,:3].mean()
            loss_grip_close = running_loss[...,gripper_idx:gripper_idx+1].mean()
            loss_terminate = running_loss[...,8:].mean()
            
            running_loss = running_loss.mean()


            running_loss = reduce_and_average(running_loss)

            if "use_adjust_scheduler" in cfg and cfg.use_adjust_scheduler:
                scheduler.step()
            else:
                scheduler.step_update(epoch * len(train_dataloader) + i)

            iter_end_time = time.time()
            iter_time = iter_end_time - iter_start_time
            train_time = iter_end_time - train_start_time
            data_time = iter_time - train_time

            if rank == 0:
                if i % 10 == 0:
                    print("[epoch {}, iter {}, iter_time {}] lr: {} loss: {},  world_vector:{}, rota:{}, grip:{}, terminate:{}, afford: {}".
                          format(epoch + 1, i + 1, iter_time, optimizer.param_groups[0]["lr"], running_loss, loss_world_vector, loss_rota, loss_grip_close, loss_terminate, affordance_loss), flush=True)

                
                if writer is not None:
                    writer.add_scalar("MSE_loss", running_loss, total_iter_num)
                    writer.add_scalar("MSE_loss_rota", loss_rota, total_iter_num)
                    writer.add_scalar("MSE_loss_world_vector", loss_world_vector, total_iter_num)
                    writer.add_scalar("MSE_loss_grip_close", loss_grip_close, total_iter_num)
                    writer.add_scalar("MSE_loss_terminate", loss_terminate, total_iter_num)
                    writer.add_scalar('MSE_loss_affordance_loss', affordance_loss, total_iter_num)
                    
            # running_loss = 0.0
            sys.stdout.flush()

            if (
                rank == 0
                and total_iter_num != 0
                and (total_iter_num % 5000 == 0)
            ):
                checkpoint = {
                    "parameter": network_module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "iter": i,
                    "total_iter_num": total_iter_num,
                    "loss": running_loss,
                    "loss_scaler": loss_scaler.state_dict(),
                }
                print("save checkpoint!", os.path.join(checkpoint_path, f"ckpt_{total_iter_num}.pth"))
                try:
                    torch.save(checkpoint, os.path.join(checkpoint_path, f"ckpt_{total_iter_num}.pth"))
                    for item in sorted(os.listdir(checkpoint_path), key=lambda x: os.path.getmtime(os.path.join(checkpoint_path, x)))[:-1]:
                        os.system('rm {}'.format(os.path.join(checkpoint_path, item)))
                except Exception as e:
                    print(e)
                    pass

            # *******************************************************#
            # EVAL PART

            # if total_iter_num % 5000 == 0 and total_iter_num != 0 and cfg.dataset.split_type != "overfit":

            #     try:
            #         evaluate_act(network, eval_dataloader, DEVICE, None, 
            #                  None, criterion, writer, args, total_iter_num,
            #                  network_module.noise_scheduler, None, cfg.num_pred_action, cfg, clipmodel=clipmodel, clip_tokenizer=clip_tokenizer)
            #     except Exception as e:
            #         import traceback
            #         traceback.print_exc()
            #         print(e)
            #     network.train()

            if total_iter_num % cfg.close_loop_eval.eval_iters == 0 and total_iter_num != 0 and cfg.use_close_loop_eval : 
            # if total_iter_num % 5000 == 0 and cfg.use_close_loop_eval : 
                close_loop_eval_ddp(cfg, args, rank, eval_dataset, DEVICE, network, total_iter_num, run_dir, writer, )

            total_iter_num += 1
            data_start_time = time.time()

def close_loop_eval_ddp(cfg, args, rank, eval_dataset, DEVICE, network, total_iter_num, run_dir, writer,):
    print(args)
    try:
        if cfg.dataname in ['calvin', 'calvin_mc']:
            close_loop_eval = getattr(importlib.import_module("close_loop_eval_diffusion_calvin"), "close_loop_eval_calvin")
        else:                        
            close_loop_eval = getattr(importlib.import_module("close_loop_eval_diffusion"), "close_loop_eval_v2")
        close_loop_eval_start_time = time.time()
        logging.info(f"begin eval: {args.rank}")
        with torch.no_grad():
            success_num, _, total_success_rate = close_loop_eval(
                            model=network,
                            test_episodes_num=cfg.close_loop_eval.test_episodes_num,
                            eval_data_list=cfg.dataset.close_loop_eval_data_list,
                            args=args,
                            rand_seed=total_iter_num,
                            stride=cfg.dataset.stride,
                            camera_coord = not cfg.dataset.use_baseframe_action,
                            root_folder = os.path.join(HydraConfig.get().runtime.cwd, run_dir, "close_loop_videos", f"{total_iter_num}_iters"),
                            data_root_path = cfg.dataset.data_path,
                            cfg = cfg,
                            eval_dataset= eval_dataset,
                        )
                    
        print('begin sync', cfg.n_action_steps)
        print(total_success_rate)
        print(success_num)
        close_loop_eval_end_time = time.time()
        total_success_rate = reduce_and_average(torch.tensor(total_success_rate, device=DEVICE))
        for k in success_num:
            success_num[k] = reduce_and_sum(torch.tensor(success_num[k], device = DEVICE))
        print('rate:', total_success_rate, success_num, cfg.close_loop_eval.eval_num)                    
        if rank == 0:
            print(f"close_loop_eval success_rate: {total_success_rate}, used_time: {close_loop_eval_end_time - close_loop_eval_start_time}", flush=True)
            logging.info(f"close_loop_eval success_rate: {total_success_rate}")
                    
            if writer is not None:
                writer.add_scalar("Close_loop_eval_success_rate", total_success_rate, total_iter_num)

            for k in success_num.keys():
                if k in cfg.close_loop_eval.eval_num:
                    print(f"{k} task success rate: {success_num[k] / cfg.close_loop_eval.eval_num[k]}", flush = True)
                    logging.info(f"{k} task success rate: {success_num[k]/ cfg.close_loop_eval.eval_num[k]}")
                    if writer is not None:
                        writer.add_scalar(f"{k} task success rate", success_num[k] / cfg.close_loop_eval.eval_num[k], total_iter_num)
    except Exception as e:
        import traceback
        traceback.print_exc()
        tb = traceback.format_exc()
        f1 = open('debug1'+str(rank)+'.out', 'a')
        f1.write(str(rank)+str(e)+'wrong\n')
        f1.write(str(rank)+str(tb)+'\n')
        f1.flush()

        f1.close()
        print(e)

def get_gt_trajectory(cfg, act_new):
    if cfg.dataname == 'maniskill':
        act_new['world_vector'] = (act_new['world_vector'] - (-0.0768)) / 0.0768 - 1.
        act_new['rotation_delta'] = (act_new['rotation_delta'] - (-0.0768)) / 0.0768 - 1.
    if cfg.dataname in ['calvin', 'calvin_mc']:
        trajectory = torch.cat([act_new['action'], act_new['gripper_closedness_action']],dim=-1)
    elif cfg.dataset.use_euler:
        trajectory = torch.cat([act_new['world_vector'], act_new['rotation_delta'], act_new['gripper_closedness_action']],dim=-1)
    else:
        trajectory = torch.cat([act_new['world_vector'], act_new['rotation_delta'], act_new['gripper_closedness_action'], act_new['terminate_episode']],dim=-1)
        
    return trajectory


if __name__ == "__main__":
    
    train()
