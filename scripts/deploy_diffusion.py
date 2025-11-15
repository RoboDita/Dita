"""
deploy.py

Provide a lightweight server/client implementation for deploying OpenVLA models (through the HF AutoClass API) over a
REST API. This script implements *just* the server, with specific dependencies and instructions below.

Note that for the *client*, usage just requires numpy/json-numpy, and requests; example usage below!

Dependencies:
    => Server (runs OpenVLA model on GPU): `pip install uvicorn fastapi json-numpy`
    => Client: `pip install requests json-numpy`

Client (Standalone) Usage (assuming a server running on 0.0.0.0:8000):

```
import requests
import json_numpy
json_numpy.patch()
import numpy as np

action = requests.post(
    "http://0.0.0.0:8000/act",
    json={"image": np.zeros((256, 256, 3), dtype=np.uint8), "instruction": "do something"}
).json()

Note that if your server is not accessible on the open web, you can use ngrok, or forward ports to your client via ssh:
    => `ssh -L 8000:localhost:8000 ssh USER@<SERVER_IP>`
"""
import os
import os.path

import hydra
from omegaconf import DictConfig

from Dataset_Sim.SimDataset_discrete import get_action_spec
import json_numpy
from scripts.close_loop_eval_diffusion import PytorchDiffInference

json_numpy.patch()
import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import draccus
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

# === Utilities ===
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def get_openvla_prompt(instruction: str, openvla_path: Union[str, Path]) -> str:
    if "v01" in openvla_path:
        return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
    else:
        return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


# === Server Interface ===
class DiffusionTransformServer:
    def __init__(self, openvla_path: Union[str, Path], traj_length, num_pred_action, trajectory_dim, use_action_head_diff=0, reg_prediction_nums=0, stride=1, dataset='lab') -> Path:
        """
        A simple server for OpenVLA models; exposes `/act` to predict an action for a given image + instruction.
            => Takes in {"image": np.ndarray, "instruction": str, "unnorm_key": Optional[str]}
            => Returns  {"action": np.ndarray}
        """
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.model = PytorchDiffInference(model=openvla_path, sequence_length = traj_length, use_action_head_diff=use_action_head_diff,
                                num_pred_action=num_pred_action, stride=stride)
        self.trajectory_dim = trajectory_dim
        self.dataname=dataset


    def predict_action(self, payload: Dict[str, Any]) -> str:
        try:
            print('recieve')
            if double_encode := "encoded" in payload:
                # Support cases where `json_numpy` is hard to install, and numpy arrays are "double-encoded" as strings
                assert len(payload.keys() == 1), "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            # Parse payload components
            image, instruction = payload["image"], payload["instruction"]

            self.model.set_natural_instruction(instruction)
            # total_num = 0
            # start_time = time.time()


            # self.model.set_eef_pose(eef_pose(env, obs["camera_param"]["base_camera"]["extrinsic_cv"], camera_coord=camera_coord))

            self.model.set_observation(rgb=image, wrist=None)
            print(instruction)
            model_output = self.model.inference(torch.eye(4), set_pose=True, trajectory_dim=self.trajectory_dim, cfg=0, ret_7=True)
            
            normalized_actions = model_output
            print(normalized_actions)
            if self.dataname=='droid':
                action_low = torch.tensor([-0.03282685, -0.04533654, -0.04281189, -0.10672842, -0.1170417 ,       -0.15017144,  0.     ]).to(torch.float32).cpu().numpy()
                action_high = torch.tensor([0.03991393, 0.04513103, 0.04638495, 0.10861504, 0.10766238, 0.1506115 , 1.  ]).to(torch.float32).cpu().numpy()
#'q01': array([-0.03282685, -0.04533654, -0.04281189, -0.10672842, -0.1170417 ,       -0.15017144,  0.        ]), 'q99': array([0.03991393, 0.04513103, 0.04638495, 0.10861504, 0.10766238, 0.1506115 , 1.        ]),
#                action_low = torch.tensor([-0.04222496, -0.04689935, -0.0431233 , -0.10854903, -0.13731207, -0.11201588, 0]).to(torch.float32).cpu().numpy()
#                action_high = torch.tensor([0.04214072 ,0.04183721, 0.04685052, 0.11397445 ,0.13725548 ,0.11503155, 1]).to(torch.float32).cpu().numpy()
            elif self.dataname=='bridge':
                action_low = torch.tensor([-0.02956029, -0.04225363, -0.02601106, -0.07742271, -0.09248264,-0.12843152,  0.  ]).to(torch.float32).cpu().numpy()
                action_high = torch.tensor([0.02900185, 0.0411192 , 0.04054347, 0.0787284 , 0.07604538,0.1249242 , 1.   ]).to(torch.float32).cpu().numpy()
            elif self.dataname=='fractal20220817_data':
                action_low = torch.tensor([-0.22453528, -0.14820013, -0.23158971, -0.35179949, -0.41930113, -0.43643461,  0.   ]).to(torch.float32).cpu().numpy()
                action_high = torch.tensor([0.17824687, 0.1493838 , 0.21842355, 0.5892666 , 0.35272657,0.44796681, 1. ]).to(torch.float32).cpu().numpy()
                pass
            elif self.dataname=='common':
                action_low = torch.tensor([-0.0633593499660492, -0.09164660893380643, -0.03522557020187378, -0.08247147373855114, -0.09154958546161651, -0.11967177748680115, 0.0 ]).to(torch.float32).cpu().numpy()
                action_high = torch.tensor([0.06707319289445901, 0.09126371890306473, 0.039003190100193263, 0.08525340244174029, 0.08206451445817953, 0.1191132126748563, 1.0 ]).to(torch.float32).cpu().numpy()
            else:
                # action_low = torch.tensor([-0.07335383, -0.07777873, -0.07212001, -0.10891825, -0.23829974, -0.19956847, 0]).to(torch.float32).cpu().numpy()
                # action_high = torch.tensor([0.08313077, 0.09487492, 0.08827358, 0.11910569, 0.18393938, 0.16685072, 1]).to(torch.float32).cpu().numpy()
                action_low = torch.tensor([-0.009082376956939697, -0.02768026292324066, -0.09064042553305625, -0.088255375623703, -0.07572497427463531, -0.10641985386610031, 0]).to(torch.float32).cpu().numpy()
                action_high = torch.tensor([0.049961209297180176, 0.029934369027614594, 0.06721316277980804, 0.06538952142000198, 0.03357397019863129, 0.17205530777573924, 1]).to(torch.float32).cpu().numpy()

            normalized_actions[...,:6] = 0.5 * (normalized_actions[...,:6] + 1) * (action_high[...,:6] - action_low[...,:6]) + action_low[...,:6]
            action = normalized_actions
            if double_encode:
                return JSONResponse(json_numpy.dumps(action))
            else:
                return JSONResponse(action)
        except:  # noqa: E722
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'image': np.ndarray, 'instruction': str}\n"
                "You can optionally an `unnorm_key: str` to specific the dataset statistics you want to use for "
                "de-normalizing the output actions."
            )
            return "error"

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.predict_action)
        uvicorn.run(self.app, host=host, port=port)


# @dataclass
# class DeployConfig:
#     # fmt: off
#     model_path: Union[str, Path] = "openvla/openvla-7b"               # HF Hub Path (or path to local run directory)

#     # Server Configuration
#     host: str = "0.0.0.0"                                               # Host IP Address
#     port: int = 10202                                                    # Host Port

    # fmt: on

@hydra.main(version_base=None, config_path=os.path.join(os.path.dirname(__file__), "..", "config"), config_name="config_diffusion_openx_finetune")
def deploy(cfg: DictConfig) -> None:
    from llama_dp import RobotTransformerNet
    action_spec = get_action_spec('cuda')
    # print(OmegaConf.to_yaml(cfg))
    network = RobotTransformerNet(
        output_tensor_spec=action_spec,
        vocab_size=cfg.model.vocab_size,
        trajectory_dim=cfg.trajectory_dim,
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
        use_action_head_diff=cfg.use_action_head_diff
        # token_embedding_size=cfg.model.token_embedding_size,
        # qformer_depth=cfg.model.qformer_depth,
        # intermediate_size=cfg.model.intermediate_size,
    )
    
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    # for n, p in network.named_parameters():
    #     print(n)
    # for n, m in network.named_modules():
    #     print(n, )
    if cfg.use_lora:
        target_modules = [n for n, m in network.named_modules() if isinstance(m, torch.nn.Linear)]
        lora_config = LoraConfig(r=cfg.lora_rank,lora_alpha=min(cfg.lora_rank, 16),lora_dropout=cfg.lora_dropout,target_modules=target_modules,init_lora_weights="gaussian",)
        network = get_peft_model(network, lora_config)
    else:
        pass
    if "ckpt_path" in cfg and cfg.ckpt_path != "None":
        ckpt_path = cfg.ckpt_path
        
        if os.path.isdir(ckpt_path):
            ckpt_path = os.path.join(ckpt_path, sorted(os.listdir(ckpt_path), key=lambda x: os.path.getmtime(os.path.join(ckpt_path, x)))[-1])
        if os.path.exists(ckpt_path):
            print('load ', ckpt_path)
            ckpt = torch.load(ckpt_path, 'cpu')
            print(network.load_state_dict(ckpt["parameter"]))
    assert cfg.dataset.traj_length == cfg.num_pred_action + 1
    server = DiffusionTransformServer(network,traj_length=cfg.dataset.traj_length, num_pred_action=cfg.dataset.traj_length-1, trajectory_dim=7,use_action_head_diff=cfg.use_action_head_diff,dataset=cfg.dataname)
    SLURM_STEP_NODELIST = os.environ['SLURM_STEP_NODELIST']
    import subprocess
    SLURM_STEP_NODELIST = os.environ['SLURM_STEP_NODELIST']
    import subprocess
    output = subprocess.check_output("scontrol show hostname {} | head -n1".format(SLURM_STEP_NODELIST), shell=True)
    host = "get you host"
    # output = subprocess.check_output("scontrol show hostname {} | head -n1".format(SLURM_STEP_NODELIST), shell=True)
#    output = os.environ['MASTER_ADDR']
    SLURM_STEP_NODELIST = os.environ['SLURM_STEP_NODELIST']
    # import subprocess
    # output = subprocess.check_output("scontrol show hostname {} | head -n1".format(SLURM_STEP_NODELIST), shell=True)
    host = "get you host"
    # output = subprocess.check_output("scontrol show hostname {} | head -n1".format(SLURM_STEP_NODELIST), shell=True)
    print(host)
    port = cfg.port if 'port' in cfg else 10203
    server.run(host, port=port)

def client_example():
    import requests
    import json_numpy
    json_numpy.patch()
    import numpy as np
    import ipdb;ipdb.set_trace()
    requests.post("http://0.0.0.0:10202/act",)
    action = requests.post(
        "http://ip:10202/act",
        json={"image": np.zeros((256, 256, 3), dtype=np.uint8), "instruction": "do something"}
    ).json()
    print(action)

if __name__ == "__main__":
    deploy()
