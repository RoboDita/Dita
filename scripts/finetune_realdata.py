import os


from scripts.finetune_utils import get_training_data
from scripts.llama_dp import RobotTransformerNet
from utils import resume_or_load_checkpoint, ExponentialMovingAverage, EMA



os.environ['LD_PRELOAD']='/xxx/xxx/Anonymous/lib/libtcmalloc_minimal.so.4.5.8'
import argparse
from dataclasses import dataclass

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
sys.path.append(os.path.join(current_path, "../Dita/utils/"))
sys.path.append(os.path.join(current_path, "../Dita/scripts"))
sys.path.append(os.path.join(current_path, "../Dita/openvla"))
sys.path.append(os.path.join(current_path, "openvla/"))


from openvla.prismatic.util import set_global_seed
from utils.ddp_utils import init_distributed_mode
import hydra
import torch
from Dataset_Sim.SimDataset_discrete import get_action_spec


from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig



os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["AWS_ACCESS_KEY_ID"] = "Your Key"
os.environ["AWS_SECRET_ACCESS_KEY"] = "Your Key"

os.environ["S3_ENDPOINT"] = "Your EndPoint"
os.environ["S3_USE_HTTPS"] = "0"
os.environ["S3_VERIFY_SSL"] = "0"
import tensorflow_io as tfio
import numpy as np
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F


from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2

from torch.utils.tensorboard import SummaryWriter

def set_seed(seed=42):  

    

    random.seed(seed)  

    np.random.seed(seed)  

    torch.manual_seed(seed)  

   

    if torch.cuda.is_available():  

        torch.cuda.manual_seed_all(seed)  

      

        torch.backends.cudnn.benchmark = False  

        torch.backends.cudnn.deterministic = True  




def dict_to_gpu(dict, DEVICE):

    gpu_dict = {}
    for k in dict:
        if k == "camera_extrinsic_cv":
            continue
        b, sample_per_episode = dict[k].shape[:2]
        gpu_dict[k] = dict[k].reshape(b * sample_per_episode, *dict[k].shape[2:]).to(DEVICE, non_blocking=True)
       
    return gpu_dict

def unnormalize(x):
    x = x.clone()
    for i in range(3):
        x[..., i] = x[..., i] * IMAGENET_DEFAULT_STD[i] + IMAGENET_DEFAULT_MEAN[i]

    return x



def get_args_parser():

    parser = argparse.ArgumentParser()
    return parser


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
            print('freeze', name)
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


@hydra.main(version_base=None, config_path=os.path.join(os.path.dirname(__file__), "..", "config"), config_name="config_diffusion_openx_finetune")
def train(cfg: DictConfig):
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # set_seed()
    # parser = get_args_parser()
    # args = parser.parse_args()
    # print(args)
    # args = None
    args = argparse.Namespace()
    print(args)
    init_distributed_mode(args, cfg)
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    print(cfg.dataset.data_path)

    # cls = RLDSDataset if not episodic else EpisodicRLDSDataset
    data_path = 's3://openx'
    # if cfg.use_droid == 1:
    #     shuffle_buffer_size = 32000
    # else:
    shuffle_buffer_size = cfg.shuffle_buffer_size
    shuffle_buffer_size = 32000
    # cfg.batch_size= 32

    


    train_dataloader, train_sampler =  get_training_data(cfg, is_training=True)
    # eval_cfg = cfg.clone()
    import copy
    eval_cfg = copy.deepcopy(cfg)
    if eval_cfg.euler_delta != 5:
        eval_cfg.euler_delta = 4
    eval_dataloader, _ =  get_training_data(eval_cfg, is_training=False)
    print('eval_dataloader', len(eval_dataloader))
    #print('begin')
    #for item in train_dataloader:
    #    print('read')

    DEVICE = "cuda:" + str(os.environ["LOCAL_RANK"]) if torch.cuda.is_available() else "cpu"
    # DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    action_spec = get_action_spec(DEVICE)
    
    network = RobotTransformerNet(
        output_tensor_spec=action_spec,
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
        # token_embedding_size=cfg.model.token_embedding_size,
        # qformer_depth=cfg.model.qformer_depth,
        # intermediate_size=cfg.model.intermediate_size,
    ) 
    
    # use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    # lora_rank: int = 32                                             # Rank of LoRA weight matrix
    # lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights

    # conf = OmegaConf.create({"a": {"aa": 10, "bb": 20}})
    # OmegaConf.set_struct(conf, True)
    run_dir = HydraConfig.get().run.dir
    checkpoint_path = os.path.join(HydraConfig.get().runtime.cwd, run_dir, "checkpoints")
    tensorboard_output_path = os.path.join(HydraConfig.get().runtime.cwd, run_dir,)
    tensorboard_path = os.path.join(tensorboard_output_path, "tensorboard")
    total_iter_num = 0
    start_epoch = 0
    tokens_per_action = network.tokens_per_action
    tokens_per_context_image = network.tokens_per_context_image
    tokens_per_step = tokens_per_action + tokens_per_context_image
    log_path = os.path.join(HydraConfig.get().runtime.cwd, run_dir, "output.log") 
    if 'pretrained_path' in cfg and cfg.pretrained_path != "None" and cfg.ckpt_path == "None":
        start_epoch, total_iter_num, checkpoint_path, tensorboard_path, log_path, run_dir = resume_or_load_checkpoint(cfg, network, None, None)      
    if cfg.use_lora:
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

        target_modules = [n for n, m in network.named_modules() if isinstance(m, torch.nn.Linear)]
        lora_config = LoraConfig(r=cfg.lora_rank,lora_alpha=cfg.lora_rank*2,lora_dropout=cfg.lora_dropout,target_modules=target_modules,init_lora_weights="gaussian",)
        network = get_peft_model(network, lora_config)
        network.print_trainable_parameters()

            

    from transformers import AutoTokenizer, CLIPModel
    clip_tokenizer = AutoTokenizer.from_pretrained(
        "openai/clip-vit-large-patch14", use_fast=False
    )
    clipmodel = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
    writer = None



    params = param_groups_weight_decay(network, lr=cfg.lr, weight_decay=0.05, lr_mult=0.1, pretrained_weight_list=("image_tokenizer.tokenizer",))

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
        updates_per_epoch=1 if len(train_dataloader) == 0 else len(train_dataloader),
        step_on_epochs=cfg.scheduler.step_on_epochs,
    )



                
    network = torch.nn.parallel.DistributedDataParallel(network.cuda(local_rank), device_ids=[local_rank], find_unused_parameters=True if cfg.use_lora else False)

    if rank == 0:
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path, exist_ok=True)
        if cfg.task_name != "test":
            if not os.path.exists(tensorboard_path):
                os.makedirs(tensorboard_path, exist_ok=True)
            writer = SummaryWriter(tensorboard_path)
        logging.basicConfig(filename=log_path, level=logging.INFO, format="%(asctime)s : %(message)s")
        print("Training!", flush=True)

    L2_loss = torch.nn.MSELoss()
    


    network.train()
    import signal, sys

    inputs = clip_tokenizer(text=[""], return_tensors="pt", max_length=77, padding="max_length", truncation=True)
    for key in inputs:
        inputs[key] = inputs[key].to(DEVICE)

    none_ccontext = clipmodel.text_model(**inputs)[0].squeeze(0).detach()


    for epoch in range(start_epoch, cfg.epoch):

        running_loss = 0.0
        data_start_time = time.time()

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        iter_start_time = time.time()
        if total_iter_num > cfg.max_iters:
            break
        try:
            
            for i, batch in enumerate(train_dataloader):
            # for i in range(1000):
                if total_iter_num > cfg.max_iters:
                    break
                optimizer.zero_grad()

                obs_new = {}
                obs_new['language_instruction'] = []

                obs_new['image'] = batch['pixel_values'].to(DEVICE).clone()
                actions = batch["action"].to(DEVICE).clone()
                loss_mask = batch['loss_weight'].to(DEVICE).clone()
                if len(batch['pixel_values'].shape) == 6:
                    obs_new['image'] = obs_new['image'].flatten(0, 1)
                    actions = actions.flatten(0, 1)
                    loss_mask = loss_mask.flatten(0, 1)

                    obs_new['language_instruction'] = np.asarray([batch['language_instruction'] for iii in range(batch['pixel_values'].shape[1])]).transpose(1, 0).flatten().tolist()
                        # 32 2
                else:
                    obs_new['language_instruction'] += batch['language_instruction']
                
                train_start_time = time.time()
                trajectory = actions

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

                pred = network(obs_new, None, noisy_action_tokens=noisy_trajectory,timesteps=timesteps, num_pred_action=cfg.num_pred_action,)

                if network.module.noise_scheduler.config.prediction_type == 'epsilon':
                    target = noise
                elif network.module.noise_scheduler.config.prediction_type == 'sample':
                    target = trajectory
                    pass
                elif network.module.noise_scheduler.config.prediction_type == 'v_prediction':
                    target = network.module.noise_scheduler.get_velocity(trajectory, noise, timesteps)
                    pass
                else:
                    raise ValueError(f"Unsupported prediction type {network.module.noise_scheduler.config.prediction_type}")
                b, num, dim = pred.shape
                #import ipdb;ipdb.set_trace()
                logits = pred
                loss_a = 0

                start_loss_action_chunk_idx = 0

                if 'no_cur_act_loss' in cfg:
                    start_loss_action_chunk_idx = 1

                # print(target, loss_mask)
                loss_mask = loss_mask[...,start_loss_action_chunk_idx:]

                loss = F.mse_loss(logits[...,start_loss_action_chunk_idx:,:], target[..., start_loss_action_chunk_idx:,:], reduction='none')
                loss_a = F.l1_loss(logits[...,start_loss_action_chunk_idx:,:], target[..., start_loss_action_chunk_idx:,:],)
                orig_loss = loss

                
                running_loss = loss.detach()
                from einops import rearrange, reduce
                
                loss = loss[loss_mask]
                loss = reduce(loss, 'b ... -> b (...)', 'mean')
                extra_loss = reduce(orig_loss[:, :-cfg.dataset.traj_length], 'b ... -> b (...)', 'mean').detach()
                
                loss = loss.mean()

                loss.backward()
                optimizer.step()
                
               
                loss_rota = running_loss[...,3:6].mean()
                loss_world_vector = running_loss[...,:3].mean()
                loss_grip_close = running_loss[...,6:7].mean()
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
                        print("[epoch {}, iter {}, iter_time {}, train_time {}, ] lr: {} loss: {}, ext_vec: {}, extra_rota: {}, world_vector:{}, rota:{}, grip:{}, mse: {}".
                            format(epoch + 1, i + 1, iter_time,train_time, optimizer.param_groups[0]["lr"], running_loss, extra_loss_trans, 
                                   extra_loss_rota, loss_world_vector, loss_rota, loss_grip_close, loss_a), flush=True)
                #        loss_rota = running_loss[...,3:7].mean()
                # loss_world_vector = running_loss[...,:3].mean()
                # loss_grip_close = running_loss[...,7:8].mean()
                # loss_terminate = running_loss[...,8:].mean()
                    
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
                ) and not 'DEBUG' in os.environ:
                    checkpoint = {
                        "parameter": network.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": epoch,
                        "iter": i,
                        "total_iter_num": total_iter_num,
                        "loss": running_loss,
                        # "loss_scaler": loss_scaler.state_dict(),
                    }
                    print("save checkpoint!", os.path.join(checkpoint_path, f"ckpt_{total_iter_num}.pth"))

                    
                    torch.save(checkpoint, os.path.join(checkpoint_path, f"ckpt_{total_iter_num}.pth"))


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
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "iter": i,
        "total_iter_num": total_iter_num,
        "loss": running_loss,
        # "loss_scaler": loss_scaler.state_dict(),
    }
    print("save checkpoint!", os.path.join(checkpoint_path, f"ckpt_{total_iter_num}.pth"))
    # for item in sorted(os.listdir(checkpoint_path), key=lambda x: os.path.getmtime(os.path.join(checkpoint_path, x)))[:]:
    #     os.system('rm {}'.format(os.path.join(checkpoint_path, item)))
    torch.save(checkpoint, os.path.join(checkpoint_path, f"ckpt_{total_iter_num}.pth"))


if __name__ == "__main__":
    SLURM_STEP_NODELIST = os.environ['SLURM_STEP_NODELIST']
    import subprocess
    output = subprocess.check_output("scontrol show hostname {} | head -n1".format(SLURM_STEP_NODELIST), shell=True)
    os.environ['MASTER_ADDR'] = output.strip().decode('ascii')
    train()
