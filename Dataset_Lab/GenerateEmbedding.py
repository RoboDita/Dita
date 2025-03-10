import os
import sys

import torch
# import a s3 package as Client
from transformers import AutoTokenizer, CLIPModel, CLIPProcessor

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pickle
import time

import numpy as np
import argparse
import subprocess



def init_distributed_mode(args):
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
            port = 22111

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
    torch.distributed.init_process_group(
        backend=dist_backend,  # init_method=args.dist_url,
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
    return  
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        flush = kwargs.pop("flush", True)
        if is_master or force:
            builtin_print(*args, **kwargs, flush=flush)

    __builtin__.print = print





parser = argparse.ArgumentParser()
args = parser.parse_args()
init_distributed_mode(args)

model = CLIPModel.from_pretrained("/xxx/xxx/share_data/Anonymous/clip-vit-large-patch14/")
processor = CLIPProcessor.from_pretrained("/xxx/xxx/share_data/Anonymous/clip-vit-large-patch14/")
clip_tokenizer = AutoTokenizer.from_pretrained(
    "/xxx/xxx/share_data/Anonymous/clip-vit-large-patch14/", use_fast=False
)
DEVICE = "cuda:" + str(os.environ["LOCAL_RANK"]) if torch.cuda.is_available() else "cpu"
local_rank = int(os.environ["LOCAL_RANK"])
model = model.to(DEVICE)
model = torch.nn.parallel.DistributedDataParallel(model.cuda(local_rank), device_ids=[local_rank], find_unused_parameters=False)

# url = "s3://Anonymous/Sim_Data"
# save_url = "s3://Anonymous/Sim_Data_language_embeddings_77token"
url = "/xxx/xxx/share_data/Anonymous/Dataset/LabData__"
save_url = "/xxx/xxx/share_data/Anonymous/Dataset/LabData___embedding"

print("Start Generate!", flush=True)

full_list = os.listdir(url)


if os.path.exists(save_url) == False:
    os.makedirs(save_url)

with torch.no_grad():
    for i in range(len(full_list)):

        data_path = os.path.join(url, full_list[i])
        if data_path.endswith(".pkl") == False:
            continue
        try:
            data = pickle.load(open(data_path, 'rb'))
        except Exception as e:
            print(e)
            print(f"{data_path}", flush = True)
            exit()
        instruction = data['language_instruction']
        if instruction == 'pick up the banana.':
            instruction = 'pick up the banana'
            data['language_instruction'] = instruction
            pickle.dump(data,open(data_path,'wb'))
        if instruction == 'pick up the red cube into the white bowl':
            instruction = 'pick up the red cube and move it into the white bowl'
            data['language_instruction'] = instruction
            pickle.dump(data,open(data_path,'wb'))
        if instruction == 'pick up the banana into the box':
            instruction = 'pick up the banana and move it into the box'
            data['language_instruction'] = instruction
            pickle.dump(data,open(data_path,'wb'))

        # if '20240914' in full_list[i]:
        #     instruction = instruction.replace('red','orange')
        #     data['language_instruction'] = instruction
        #     pickle.dump(data,open(data_path,'wb'))
        
        inputs = clip_tokenizer(text=instruction, return_tensors="pt", max_length=77, padding="max_length")
        for key in inputs:
            inputs[key] = inputs[key].to(DEVICE)
        text_embeddings = model.module.text_model(**inputs)[0].squeeze(0)
        text_embeddings = text_embeddings.detach().cpu().numpy()
        save_path = os.path.join(save_url, full_list[i])
        pickle.dump(text_embeddings, open(save_path, 'wb'))
        print(f"{i} done!", flush = True)