import os
import sys
from networkx import expected_degree_graph

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

client = Client()
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
url = "s3://Anonymous/Sim_Data/ManiSkill2-camerabase3/"
save_url = "s3://Anonymous/Sim_Data_language_embeddings_77token_camerabase3/"


directory_list = ["PickClutterYCB-v0/"]
# directory_list = ['PegInsertionSide-v0-0321', 'PickCube-v0-0321', 'StackCube-v0-0321', 'AssemblingKits-v0-0321']
# directory_list = directory_list + ['PickSingleYCB-v0-0321', 'PickSingleYCB-v0-0321-part2']
# directory_list = directory_list + ['PickSingleYCB-v0-0321-part3', 'PickSingleYCB-v0-0321-part4']
# directory_list = directory_list + ["PickSingleYCB-v0-0321-part5", "PickSingleYCB-v0-0321-part6"]
print("Start Generate!", flush=True)



with torch.no_grad():
    for i in range(len(directory_list)):

        # calc_every_traj = "YCB" in directory_list[i]
        calc_every_traj = True

        data_root_path = os.path.join(url, directory_list[i])
        data_root_path = url
        # data_list = list(client.list(data_root_path))
        # data_list = pickle.load(open(f"/xxx/xxx/share_data/Anonymous/Dataset/Sim_Data/maniskill2_0503_datalist/{directory_list[i].replace('/','')}_full_list.pkl", "rb"))
        data_list = pickle.load(open('/xxx/xxx/share_data/Anonymous/Dataset/Sim_Data/maniskill_fewcam_datalist/stackcube_segmentation_fulllist.pkl','rb'))
        data_list = sorted(data_list)

        last_time = time.time()
        # for ii, data in enumerate(data_list):
        # for ii, data in enumerate(temp_data_list):
        epi_per_gpu = (len(data_list) // args.world_size) + 1
        print(f"total: {len(data_list)}, per_gpu: {epi_per_gpu}", flush =  True)
        for ii in range(epi_per_gpu * args.rank, min(epi_per_gpu * (args.rank + 1), len(data_list))):

            data = data_list[ii].replace(directory_list[i],'')
            data = data_list[ii].replace(data_list[ii].split('/')[0],'')
            data = data_list[ii]

            if ii % 50 == 0:
                now_time = time.time()
                print("{} done!, Time: {}".format(ii, now_time - last_time), flush=True)
                last_time = now_time
            if data.endswith(".pkl") == False:
                continue
            data_path = os.path.join(data_root_path, data)
            # import pdb;pdb.set_trace()
            try:
                pkl_file = pickle.loads(client.get(data_path))
            except:
                print(data_path)
                continue
            save_path = os.path.join(save_url, directory_list[i], data)
            save_path = os.path.join(save_url, data)
            # import pdb;pdb.set_trace()
            if not calc_every_traj and ii != 0:
                client.put(save_path, pickle.dumps(text_embeddings))
                continue
            instruction = pkl_file["step"][0]["observation"]["natural_instruction"]
            inputs = clip_tokenizer(text=instruction, return_tensors="pt", max_length=77, padding="max_length")
            for key in inputs:
                inputs[key] = inputs[key].to(DEVICE)
            text_embeddings = model.module.text_model(**inputs)[0].squeeze(0)
            text_embeddings = text_embeddings.detach().cpu().numpy()
            client.put(save_path, pickle.dumps(text_embeddings))

        print(f"{directory_list[i], args.rank} done!", flush=True)
        # print(text_embeddings.shape)
        # sys.exit()
