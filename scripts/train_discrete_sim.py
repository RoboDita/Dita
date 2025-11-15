import argparse
import datetime
import importlib
import logging
import os
os.environ["TORCH_DISTRIBUTED_DEBUG"] = 'INFO'
import subprocess
import sys
import time

import hydra
import torch
from Dataset_Sim.SimDataset_discrete import SimDataset
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MS2_RENDERER_LOG_LEVEL"] = "error"
os.environ["MS2_ASSET_DIR"] = "/xxx/xxx/share_data/Anonymous/maniskill2/assets"
import numpy as np
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import (
    Transform3d,
    matrix_to_quaternion,
    quaternion_to_matrix,
)
from scipy.spatial.transform import Rotation as R
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from timm.utils import NativeScaler
from torch.utils.tensorboard import SummaryWriter
import math
from functools import partial


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


def dict_to_gpu(dict, DEVICE):

    gpu_dict = {}
    for k in dict:
        if k == "camera_extrinsic_cv" :
            continue
        if k == "natural_language_instruction":
            gpu_dict[k] = dict[k]
            continue
        # if k == 'segmentation':
        #     gpu_dict[k] = []
        #     for i in range(len(dict['segmentation'])):
        #         temp_dict = {}
        #         for kk in dict['segmentation'][i].keys():
        #             if kk == 'boxes' or kk == 'class_labels' or kk == 'detect_labels':
        #                 temp_dict[kk] = dict['segmentation'][i][kk].to(DEVICE)
        #             else:
        #                 temp_dict[kk] = dict['segmentation'][i][kk]
        #         gpu_dict[k].append(temp_dict)
        #     continue
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
            port = 22113

            print(f"MASTER_PORT = {port}")
            os.environ["MASTER_PORT"] = str(port)

            time.sleep(3)

        node_list = os.environ["SLURM_STEP_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # addr = subprocess.getoutput("ifconfig bond0 | grep 'inet ' | awk '{print $2}'")
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
    print(f"local_rank: {local_rank}", flush = True)
    args.dist_backend = "nccl"
    print("| distributed init (rank {})".format(rank), flush=True)
    dist_backend = "nccl"
    init_method = os.path.join(HydraConfig.get().runtime.cwd, HydraConfig.get().run.dir, "initial_method.txt")
    torch.distributed.init_process_group(
        backend=dist_backend,  # init_method=args.dist_url,
        # init_method=f"file://{init_method}",
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
    # return
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


def evaluate(network, eval_dataloader, DEVICE, tokens_per_context_image, tokens_per_step, writer, args, total_iter_num, cfg):

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        network.eval()
        eval_loss_total = 0.0
        eval_loss_per_iter = 0.0

        acc_total = 0.0
        loss_l2_total = 0.0
        TP_total = 0.0
        TP_and_FN_total = 0.0

        acc_world_vector_total = 0.0
        acc_rotation_delta_total = 0.0
        acc_gripper_closedness_total = 0.0
        acc_terminate_total = 0.0

        loss_wv_l2_total = 0.0
        loss_rota_delta_l2_total = 0.0
        loss_grip_close_l2_total = 0.0
        loss_term_l2_total = 0.0

        for i, batch in enumerate(eval_dataloader):
            if i % 50 == 0:
                print(f"{i}/{len(eval_dataloader)}", flush = True)

            obs_dict = batch["observation"]
            act_dict = batch["action"]
            obs_new = dict_to_gpu(obs_dict, DEVICE)
            act_new = dict_to_gpu(act_dict, DEVICE)
            camera_extrinsic_cv = batch["observation"]["camera_extrinsic_cv"].flatten(0, 2).to(DEVICE, non_blocking=True)
            if cfg.dataset.use_baseframe_action:
                camera_extrinsic_cv = torch.eye(4).to(DEVICE).unsqueeze(0).repeat(camera_extrinsic_cv.shape[0], 1, 1)

            labels = network.module.action_tokenizer.tokenize(act_new)
            with torch.cuda.amp.autocast():
                output_tokens = network(obs_new, act_new, act_tokens=labels)
            b, num, dim = output_tokens.shape
            logits = output_tokens.reshape(b, cfg.dataset.traj_length, tokens_per_step - tokens_per_context_image, dim)  # bs, seq_length, token_per_step, dim


            logits = logits.permute(0, 3, 1, 2).contiguous()
            labels = labels[..., 1:]
            logits = logits[..., :-1]
            loss_mask = None
            loss_mask = (torch.sum(act_new['world_vector'], dim = [2]) + torch.sum(act_new['rotation_delta'], dim = [2]) \
                        + torch.sum( act_new['terminate_episode'][:,:,0].unsqueeze(-1), dim = [2]) + torch.sum(obs_new['image'], dim = [2,3,4]) ) != 0
            if loss_mask is None:
                loss = criterion(logits, labels)
            else:
                loss = F.cross_entropy(logits, labels, reduction = 'none')
                loss = loss[loss_mask].mean()

            
            detokenize_output, logits_arg, _ = network.module.action_tokenizer.detokenize(logits.permute(0, 2, 3, 1).contiguous().float())

            terminate_recall_TP, terminate_recall_TP_and_FN = calc_terminate_recall(labels, logits_arg)

            acc = (logits_arg == labels).sum() / (labels.shape[0] * labels.shape[1] * labels.shape[2])

            loss_l2, loss_wv_l2, loss_rota_delta_l2, loss_grip_close_l2, loss_term_l2 = calc_l2_loss(act_new, detokenize_output, camera_extrinsic_cv)

            eval_loss_per_iter = loss.detach()

            loss_l2 = reduce_and_average(loss_l2)
            loss_wv_l2 = reduce_and_average(loss_wv_l2)
            loss_rota_delta_l2 = reduce_and_average(loss_rota_delta_l2)
            loss_grip_close_l2 = reduce_and_average(loss_grip_close_l2)
            loss_term_l2 = reduce_and_average(loss_term_l2)
            eval_loss_per_iter = reduce_and_average(eval_loss_per_iter)
            acc = reduce_and_average(acc)

            
            torch.distributed.all_reduce(terminate_recall_TP)
            torch.distributed.all_reduce(terminate_recall_TP_and_FN)

            acc_world_vector = calc_acc_and_reduce(logits_arg[:, :, :3], labels[:, :, :3])
            acc_rotation_delta = calc_acc_and_reduce(logits_arg[:, :, 3:7], labels[:, :, 3:7])
            acc_gripper_closedness = calc_acc_and_reduce(logits_arg[:, :, -2], labels[:, :, -2])
            acc_terminate = calc_acc_and_reduce(logits_arg[:, :, -1], labels[:, :, -1])

            eval_loss_total += eval_loss_per_iter
            acc_total += acc
            loss_l2_total += loss_l2
            TP_total += terminate_recall_TP
            TP_and_FN_total += terminate_recall_TP_and_FN

            acc_world_vector_total += acc_world_vector
            acc_rotation_delta_total += acc_rotation_delta
            acc_gripper_closedness_total += acc_gripper_closedness
            acc_terminate_total += acc_terminate

            loss_wv_l2_total += loss_wv_l2
            loss_grip_close_l2_total += loss_grip_close_l2
            loss_rota_delta_l2_total += loss_rota_delta_l2
            loss_term_l2_total += loss_term_l2

            eval_loss_per_iter = 0

        eval_loss_total /= i + 1
        acc_total /= i + 1
        loss_l2_total /= i + 1
        TP_total /= i + 1
        TP_and_FN_total /= i + 1

        acc_world_vector_total /= i + 1
        acc_rotation_delta_total /= i + 1
        acc_gripper_closedness_total /= i + 1
        acc_terminate_total /= i + 1

        loss_wv_l2_total /= i + 1
        loss_grip_close_l2_total /= i + 1
        loss_rota_delta_l2_total /= i + 1
        loss_term_l2_total /= i + 1

        if args.rank == 0:

            print(
                "EVAL: Acc: {}, loss: {}, L2_loss: {}, Terminate_recall: {}".format(
                    acc_total, eval_loss_total, loss_l2_total, TP_total / TP_and_FN_total
                )
            )
            logging.info(
                "EVAL: Acc: {}, loss: {}, L2_loss: {}, Terminate_recall: {}".format(
                    acc_total, eval_loss_total, loss_l2_total, TP_total / TP_and_FN_total
                )
            )
            if writer is not None:
                writer.add_scalar("EVAL_ACC", acc_total, total_iter_num / cfg.gradient_accumultation_step)
                writer.add_scalar("EVAL_MSELoss", loss_l2, total_iter_num / cfg.gradient_accumultation_step)
                writer.add_scalar("EVAL_CrossEntropyLoss", eval_loss_total, total_iter_num / cfg.gradient_accumultation_step)
                writer.add_scalar("EVAL_Acc_WorldVector", acc_world_vector_total, total_iter_num / cfg.gradient_accumultation_step)
                writer.add_scalar("EVAL_Acc_RotationDelta", acc_rotation_delta_total, total_iter_num / cfg.gradient_accumultation_step)
                writer.add_scalar("EVAL_Acc_GripperClosed", acc_gripper_closedness_total, total_iter_num / cfg.gradient_accumultation_step)
                writer.add_scalar("EVAL_Acc_Terminate", acc_terminate_total, total_iter_num / cfg.gradient_accumultation_step)
                if TP_and_FN_total != 0:
                    writer.add_scalar("EVAL_terminate_Recall", TP_total / TP_and_FN_total, total_iter_num / cfg.gradient_accumultation_step)

                writer.add_scalar("EVAL_MSELoss_WorldVector", loss_wv_l2_total, total_iter_num / cfg.gradient_accumultation_step)
                writer.add_scalar("EVAL_MSELoss_RotationDelta", loss_rota_delta_l2_total, total_iter_num / cfg.gradient_accumultation_step)
                writer.add_scalar("EVAL_MSELoss_GripperClosed", loss_grip_close_l2_total, total_iter_num / cfg.gradient_accumultation_step)
                writer.add_scalar("EVAL_MSELoss_Terminate", loss_term_l2_total, total_iter_num / cfg.gradient_accumultation_step)


def remove_prev_ckpt(path):

    rank = int(os.environ["RANK"])
    files = [os.path.join(path, f) for f in os.listdir(path)]
    files.sort(key=os.path.getmtime, reverse=False)
    if rank == 0:
        for i in range(len(files) - 2):
            os.remove(files[i])


@hydra.main(config_path=os.path.join(os.path.dirname(__file__), "..", "config"), config_name="config_discrete_sim")

def train(cfg: DictConfig):

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # parser = get_args_parser()
    # args = parser.parse_args()
    args = argparse.Namespace()
    # print(args)
    init_distributed_mode(args, cfg)
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])

    train_dataset = SimDataset(
        data_path=cfg.dataset.data_path,
        language_embedding_path=cfg.dataset.language_embedding_path,
        dataset_type=0,
        use_baseframe_action=cfg.dataset.use_baseframe_action,
        split_type=cfg.dataset.split_type,
        traj_per_episode=cfg.dataset.traj_per_episode,
        traj_length=cfg.dataset.traj_length,
        data_cam_list=cfg.dataset.train_data_cam_list,
        stride=cfg.dataset.stride,
        num_given_observation=cfg.dataset.num_given_observation,
        include_target=cfg.dataset.include_target,
    )  # dataset_type 0 for train and 1 for eval
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        drop_last=True,
        num_workers=8,
        prefetch_factor=6,
        pin_memory=True,
        persistent_workers=True,
    )

    eval_dataset = SimDataset(
        data_path=cfg.dataset.data_path,
        language_embedding_path=cfg.dataset.language_embedding_path,
        dataset_type=1,
        use_baseframe_action=cfg.dataset.use_baseframe_action,
        split_type=cfg.dataset.split_type,
        traj_per_episode=2,
        traj_length=cfg.dataset.traj_length,
        data_cam_list=cfg.dataset.eval_data_cam_list,
        stride=cfg.dataset.stride,
        num_given_observation=cfg.dataset.num_given_observation,
        include_target=cfg.dataset.include_target,
    )  # dataset_type 0 for train and 1 for eval
    eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset)
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

    DEVICE = "cuda:" + str(os.environ["LOCAL_RANK"]) if torch.cuda.is_available() else "cpu"
    # DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    action_spec = get_action_spec(cfg.action_spec, DEVICE)

    net = getattr(importlib.import_module(cfg.model.name), "llama_discrete")
    network = net(
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
        input_size = cfg.model.input_size if cfg.model.input_size is not None else None,
    )

    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, network.parameters()), lr=1e-4, eps=1e-7)
    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, network.parameters()), betas=(0.9, 0.98), lr=1e-4, weight_decay=0.01)
    params = param_groups_weight_decay(network, lr=cfg.lr, weight_decay=0.05, lr_mult=0.1, pretrained_weight_list=("image_tokenizer.tokenizer",))

    # import ipdb; ipdb.set_trace()
    if cfg.optimizer.name == "adamw":
        optimizer = create_optimizer_v2(
            params, opt=cfg.optimizer.name, lr=cfg.lr, weight_decay=cfg.optimizer.weight_decay, betas=(cfg.optimizer.betas_0, cfg.optimizer.betas_1)
        )

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 10000)
    # scheduler, _ = create_scheduler_v2(optimizer, num_epochs=10, updates_per_epoch=len(train_dataloader), step_on_epochs=False)
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
        # import ipdb;ipdb.set_trace()
        print("use adjust scheduler!!!!!", flush = True)
        lr = cfg.lr

        iter_per_epoch = int(3121.0 * 8 / (cfg.batch_size / 64)  / int(os.environ["SLURM_NTASKS"])) 
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
    loss_scaler = NativeScaler()

    if "ckpt_path" in cfg and cfg.ckpt_path != "None":
        ckpt = torch.load(cfg.ckpt_path, map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()))
        network.load_state_dict(ckpt["parameter"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        loss_scaler.load_state_dict(ckpt["loss_scaler"])

        start_epoch = ckpt["epoch"]
        total_iter_num = ckpt["total_iter_num"] + 1

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(DEVICE)

    # network = network.to(DEVICE)
    network = torch.nn.parallel.DistributedDataParallel(network.cuda(local_rank), device_ids=[local_rank], find_unused_parameters=False)
    #***********************************debug
    # close_loop_eval = getattr(importlib.import_module("close_loop_eval"), "close_loop_eval_v2")
    # close_loop_eval_start_time = time.time()
    # success_num, total_success_rate = close_loop_eval(
    #     model=network,
    #     test_episodes_num=cfg.close_loop_eval.test_episodes_num,
    #     eval_data_list=cfg.dataset.close_loop_eval_data_list,
    #     args=args,
    #     rand_seed=total_iter_num,
    #     stride=cfg.dataset.stride,
    #     camera_coord = not cfg.dataset.use_baseframe_action,
    #     root_folder = os.path.join(HydraConfig.get().runtime.cwd, HydraConfig.get().run.dir, "close_loop_videos", f"{total_iter_num}_iters"),
    #     data_root_path = cfg.dataset.data_path,
    #     exec_steps = cfg.close_loop_eval.exec_steps,
    #     # use_language_instruction = cfg.dataset.use_language_instruction,
    # )
    # total_success_rate = reduce_and_average(torch.tensor(total_success_rate, dtype = torch.float32, device = DEVICE))
    # for k in success_num:
    #     success_num[k] = reduce_and_average(torch.tensor(success_num[k], dtype = torch.float32, device = DEVICE))
    # print("FINAL: ",success_num, flush = True)
    # exit()
    # #***********************************debug
    tokens_per_action = network.module.tokens_per_action
    tokens_per_context_image = network.module.tokens_per_context_image
    tokens_per_step = tokens_per_action + tokens_per_context_image

    criterion = torch.nn.CrossEntropyLoss()
    L2_loss = torch.nn.MSELoss()
    # loss_scaler = NativeScaler()

    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    tensorboard_output_path = os.path.join(cfg.tensorboard_output_dir, cfg.task_name + "_" + current_time)
    checkpoint_path = os.path.join(HydraConfig.get().runtime.cwd, HydraConfig.get().run.dir, "checkpoints")
    tensorboard_path = os.path.join(tensorboard_output_path, "tensorboard")
    log_path = "./output.log"
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

        train_sampler.set_epoch(epoch)

        running_loss = 0.0
        data_start_time = time.time()
        # torch.autograd.set_detect_anomaly(True)
        for i, batch in enumerate(train_dataloader):
            # torch.cuda.synchronize()
            iter_start_time = time.time()
            # import ipdb;ipdb.set_trace()
            if total_iter_num % cfg.gradient_accumultation_step == 1 or cfg.gradient_accumultation_step == 1:
                optimizer.zero_grad()
            obs_dict = batch["observation"]
            act_dict = batch["action"]
            camera_extrinsic_cv = batch["observation"]["camera_extrinsic_cv"].flatten(0, 2).to(DEVICE, non_blocking=True)
            if cfg.dataset.use_baseframe_action:
                camera_extrinsic_cv = torch.eye(4).to(DEVICE).unsqueeze(0).repeat(camera_extrinsic_cv.shape[0], 1, 1)
            obs_new = dict_to_gpu(obs_dict, DEVICE)
            act_new = dict_to_gpu(act_dict, DEVICE)

            # torch.cuda.synchronize()
            train_start_time = time.time()
            labels = network.module.action_tokenizer.tokenize(act_new)

            # act_new_gt = torch.cat(
            #     [act_new["world_vector"], act_new["rotation_delta"], act_new["gripper_closedness_action"], act_new["terminate_episode"][..., [0]]],
            #     dim=-1,
            # )
            # act_new_gt = act_new_gt.reshape(-1)

            # import ipdb; ipdb.set_trace()

            with torch.cuda.amp.autocast():
                if "dino_supervision" in cfg and cfg.dino_supervision is not None and cfg.dino_supervision:
                    output_tokens, seg_loss = network(obs_new, act_new, act_tokens=labels, seg_labels = obs_new['segmentation'])
                else:
                    seg_loss = None
                    output_tokens = network(obs_new, act_new, act_tokens=labels)

            b, num, dim = output_tokens.shape
            logits = output_tokens.reshape(b, cfg.dataset.traj_length, tokens_per_action, dim)  # bs, seq_length, token_per_step, dim

            
            labels = labels[..., 1:].contiguous()
            logits = logits[..., :-1, :].contiguous()
            loss_mask = None
            # import ipdb;ipdb.set_trace()
            img_mask = torch.sum(obs_new['image'], dim = [2,3,4])
            img_mask = torch.cat([img_mask, torch.zeros((img_mask.shape[0], act_new['world_vector'].shape[1] - img_mask.shape[1])).to(img_mask.device)], dim = 1 )
            # loss_mask = (torch.sum(act_new['world_vector'], dim = [2]) + torch.sum(act_new['rotation_delta'], dim = [2]) \
            #             + torch.sum( act_new['terminate_episode'][:,:,0].unsqueeze(-1), dim = [2]) + torch.sum(obs_new['image'], dim = [2,3,4]) ) != 0
            # loss_mask = (torch.sum(act_new['world_vector'], dim = [2]) + torch.sum(act_new['rotation_delta'], dim = [2]) \
            #             + torch.sum( act_new['terminate_episode'][:,:,0].unsqueeze(-1), dim = [2]) ) != 0
            loss_mask = (torch.sum(act_new['world_vector'], dim = [2]) + torch.sum(act_new['rotation_delta'], dim = [2]) \
                        + torch.sum( act_new['terminate_episode'][:,:,0].unsqueeze(-1), dim = [2]) + img_mask) != 0
            if loss_mask is None:
                loss = criterion(logits.reshape(-1, dim), labels.reshape(-1))
            else:
                loss = F.cross_entropy(logits.permute(0, 3, 1, 2).contiguous(), labels, reduction = 'none')
                loss = loss[loss_mask].mean()
            
            if seg_loss is not None:
                loss = loss + cfg.dino_supervision_ratio * seg_loss

            # import ipdb; ipdb.set_trace()
            # logits = torch.diagonal(logits, dim1=-2, dim2=-1)
            # loss = F.mse_loss(logits.reshape(-1, 1).squeeze(-1), act_new_gt).float() * 1e5

            # logits = logits.squeeze(-1)
            # detokenize_output = {}
            # detokenize_output["world_vector"] = logits[..., :3]
            # detokenize_output["rotation_delta"] = logits[..., 3:7]
            # detokenize_output["gripper_closedness_action"] = logits[..., 7:8]

            # detokenize_output["terminate_episode"] = torch.nn.functional.one_hot((logits[..., 8] < 0.5).long(), num_classes=3)

            detokenize_output, logits_arg, _ = network.module.action_tokenizer.detokenize(logits.float())
            # logits_arg = torch.max(logits, dim=1)[1]
            # Check_Wrong_tokens(labels, logits_arg)
            terminate_recall_TP, terminate_recall_TP_and_FN = calc_terminate_recall(labels, logits_arg)

            acc = (logits_arg == labels).sum() / (labels.numel())
            gripper_close_acc = (logits_arg == labels)[act_new["loss_weight"] > 1].sum() / (
                (act_new["loss_weight"][act_new["loss_weight"] > 1]).numel() + 1e-5
            )

            # import ipdb; ipdb.set_trace()

            # detokenize_gt = network.module.action_tokenizer.detokenize(labels)
            # loss.backward()
            # optimizer.step()
            if total_iter_num % cfg.gradient_accumultation_step == 0:
                need_update = True
            else:
                need_update = False

            loss_scaler(loss / cfg.gradient_accumultation_step, optimizer, need_update = need_update)

               

            loss_l2, loss_wv_l2, loss_rota_delta_l2, loss_grip_close_l2, loss_term_l2 = calc_l2_loss(act_new, detokenize_output, camera_extrinsic_cv)

            running_loss = loss.detach()

            loss_l2 = reduce_and_average(loss_l2)
            loss_wv_l2 = reduce_and_average(loss_wv_l2)
            loss_rota_delta_l2 = reduce_and_average(loss_rota_delta_l2)
            loss_grip_close_l2 = reduce_and_average(loss_grip_close_l2)
            loss_term_l2 = reduce_and_average(loss_term_l2)
            running_loss = reduce_and_average(running_loss)
            # acc = 0
            acc = reduce_and_average(acc)
            gripper_close_acc = reduce_and_average(gripper_close_acc)

            torch.distributed.all_reduce(terminate_recall_TP)
            torch.distributed.all_reduce(terminate_recall_TP_and_FN)

            acc_world_vector = calc_acc_and_reduce(logits_arg[:, :, :3], labels[:, :, :3])
            acc_rotation_delta = calc_acc_and_reduce(logits_arg[:, :, 3:7], labels[:, :, 3:7])
            acc_gripper_closedness = calc_acc_and_reduce(logits_arg[:, :, -2], labels[:, :, -2])
            acc_terminate = calc_acc_and_reduce(logits_arg[:, :, -1], labels[:, :, -1])

            if need_update:
                if "use_adjust_scheduler" in cfg and cfg.use_adjust_scheduler:
                    scheduler.step()
                else:
                    scheduler.step_update(epoch * len(train_dataloader) + i / cfg.gradient_accumultation_step)

            iter_end_time = time.time()
            iter_time = iter_end_time - iter_start_time
            train_time = iter_end_time - train_start_time
            data_time = iter_time - train_time

            if rank == 0:
                if i % 10 == 0:
                    print("[epoch {}, iter {}] loss: {}, l2_loss: {}, acc: {}".format(epoch + 1, i + 1, running_loss, loss_l2, acc), flush=True)
                    print("iter_time : ", iter_time, " ", "train_time : ", train_time, " ", "data_time : ", data_time, flush=True)
                    print("Real_Data_time: ", iter_start_time - data_start_time, flush=True)
                    logging.info("[epoch {}, iter {}] loss: {}, l2_loss: {}, acc: {}".format(epoch + 1, i + 1, running_loss, loss_l2, acc))
                    logging.info(
                        "lr: {}, terminate_recall: {}".format(optimizer.param_groups[0]["lr"], terminate_recall_TP / terminate_recall_TP_and_FN)
                    )
                    logging.info(
                        "acc(world_vector, rotation_delta, gripper_closed, teminate) = ({}, {}, {}, {})".format(
                            acc_world_vector, acc_rotation_delta, acc_gripper_closedness, acc_terminate
                        )
                    )
                if writer is not None and need_update:
                    writer.add_scalar("CrossEntropyLoss", running_loss, total_iter_num / cfg.gradient_accumultation_step)
                    writer.add_scalar("MSELoss", loss_l2, total_iter_num / cfg.gradient_accumultation_step)
                    writer.add_scalar("Acc", acc, total_iter_num / cfg.gradient_accumultation_step)
                    writer.add_scalar("lr", optimizer.param_groups[0]["lr"], total_iter_num / cfg.gradient_accumultation_step)
                    writer.add_scalar("Acc_WorldVector", acc_world_vector, total_iter_num / cfg.gradient_accumultation_step)
                    writer.add_scalar("Acc_RotationDelta", acc_rotation_delta, total_iter_num / cfg.gradient_accumultation_step)
                    writer.add_scalar("Acc_GripperClosed", acc_gripper_closedness, total_iter_num / cfg.gradient_accumultation_step)
                    writer.add_scalar("Acc_GripperChange", gripper_close_acc, total_iter_num / cfg.gradient_accumultation_step)
                    writer.add_scalar("Acc_Terminate", acc_terminate, total_iter_num / cfg.gradient_accumultation_step)
                    if terminate_recall_TP_and_FN != 0:
                        writer.add_scalar("terminate_Recall", terminate_recall_TP / terminate_recall_TP_and_FN, total_iter_num / cfg.gradient_accumultation_step)

                    writer.add_scalar("MSELoss_WorldVector", loss_wv_l2, total_iter_num / cfg.gradient_accumultation_step)
                    writer.add_scalar("MSELoss_RotationDelta", loss_rota_delta_l2, total_iter_num / cfg.gradient_accumultation_step)
                    writer.add_scalar("MSELoss_GripperClosed", loss_grip_close_l2, total_iter_num / cfg.gradient_accumultation_step)
                    writer.add_scalar("MSELoss_Terminate", loss_term_l2, total_iter_num / cfg.gradient_accumultation_step)
            running_loss = 0.0
            sys.stdout.flush()

            if (
                rank == 0
                and total_iter_num != 0
                and (total_iter_num % 100 == 0 or (total_iter_num % 100 == 0 and acc == 1 and cfg.dataset.split_type == "overfit"))
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
                # remove_prev_ckpt(checkpoint_path)
                for item in sorted(os.listdir(checkpoint_path), key=lambda x: os.path.getmtime(os.path.join(checkpoint_path, x)))[:-1]:
                    if '0000' in item:
                        continue
                    os.system('rm {}'.format(os.path.join(checkpoint_path, item)))

            # *******************************************************#
            # EVAL PART

            # if total_iter_num % 1000 == 0 and cfg.dataset.split_type != "overfit":
            
            if total_iter_num % (cfg.eval_per_iter * cfg.gradient_accumultation_step) == 0 and   total_iter_num !=0 and cfg.dataset.split_type != "overfit" :

                # evaluate(network, eval_dataloader, DEVICE, tokens_per_context_image, tokens_per_step, writer, args, total_iter_num, cfg)

                # network.train()
                pass

            #*******************************************************#
            # Close Loop Eval Part
            if total_iter_num % (cfg.eval_per_iter * cfg.gradient_accumultation_step) == 0 and  total_iter_num!=0 and cfg.use_close_loop_eval or total_iter_num == 10000: 
            # if total_iter_num % 5000 == 0 and cfg.use_close_loop_eval : 
                close_loop_eval = getattr(importlib.import_module("close_loop_eval"), "close_loop_eval_v2")
                close_loop_eval_start_time = time.time()
                success_num, total_success_rate = close_loop_eval(
                    model=network,
                    test_episodes_num=cfg.close_loop_eval.test_episodes_num,
                    eval_data_list=cfg.dataset.close_loop_eval_data_list,
                    args=args,
                    rand_seed=total_iter_num,
                    stride=cfg.dataset.stride,
                    camera_coord = not cfg.dataset.use_baseframe_action,
                    root_folder = os.path.join(HydraConfig.get().runtime.cwd, HydraConfig.get().run.dir, "close_loop_videos", f"{total_iter_num}_iters"),
                    data_root_path = cfg.dataset.data_path,
                    exec_steps = cfg.close_loop_eval.exec_steps,
                    # use_language_instruction = cfg.dataset.use_language_instruction,
                )
                close_loop_eval_end_time = time.time()
                total_success_rate = reduce_and_average(torch.tensor(total_success_rate, dtype = torch.float32, device = DEVICE))
                for k in success_num:
                    success_num[k] = reduce_and_average(torch.tensor(success_num[k], dtype = torch.float32, device = DEVICE))
                
                if rank == 0:
                    print(f"close_loop_eval success_rate: {total_success_rate}, used_time: {close_loop_eval_end_time - close_loop_eval_start_time}", flush=True)
                    logging.info(f"close_loop_eval success_rate: {total_success_rate}")
                
                    if writer is not None:
                        writer.add_scalar("Close_loop_eval_success_rate", total_success_rate, total_iter_num / cfg.gradient_accumultation_step)

                    for k in success_num.keys():
                        print(f"{k} task success rate: {success_num[k] * args.world_size / cfg.close_loop_eval.eval_num[k]}", flush = True)
                        logging.info(f"{k} task success rate: {success_num[k] * args.world_size/ cfg.close_loop_eval.eval_num[k]}")
                        if writer is not None:
                            writer.add_scalar(f"{k} task success rate", success_num[k] *args.world_size / cfg.close_loop_eval.eval_num[k], total_iter_num / cfg.gradient_accumultation_step)


            total_iter_num += 1
            data_start_time = time.time()


if __name__ == "__main__":

    train()
