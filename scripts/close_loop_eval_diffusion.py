import argparse
import logging
import os

from utils.data_utils import get_pose_cam
os.environ["MS2_ASSET_DIR"] = "/xxx/xxx/share_data/Anonymous/maniskill2/assets"
import pickle
import sys
import time

import gymnasium as gym
import numpy as np
import sapien.core as sapien
import torch
import torch.nn as nn
import torchvision
from Dataset_Sim.SimDataset import process_traj_v3
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.utils.io_utils import load_json
from mani_skill2.utils.sapien_utils import look_at
from mani_skill2.utils.visualization.cv2_utils import OpenCVViewer
from mani_skill2.utils.wrappers import RecordEpisode

from openx_utils.robot_utils import cal_action, cal_action_from_pose, eef_pose
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPModel, CLIPProcessor
from transforms3d.quaternions import mat2quat, quat2mat

# import a s3 package as Client
from pytorch3d.transforms import (
                    Transform3d,
                    matrix_to_euler_angles,
                    matrix_to_quaternion,
                    matrix_to_rotation_6d,
                    quaternion_to_matrix,
                    euler_angles_to_matrix
                )
from collections import defaultdict
import random

# param
MAX_EPISODE_STEPS = 300
TARGET_CONTROL_MODE = "pd_ee_delta_pose"  # param can be one of ['pd_ee_delta_pose', 'pd_ee_target_delta_pose']
CAL_DELTA_METHOD = 2  # 0:direct 1:tf 2:model
CAMERA_NAMES = ["hand_camera", "camera_1", "camera_2", "camera_3", "camera_4", "camera_5"]
CAMERA_POSES = {
    "camera_1": look_at([0.3, 0.2, 0.6], [-0.1, 0, 0.1]),
    "camera_2": look_at([0.3, -0.2, 0.6], [-0.1, 0, 0.1]),
    "camera_3": look_at([0.3, 0.2, 0.4], [-0.1, 0, 0.3]),
    "camera_4": look_at([0.5, -0.2, 0.8], [-0.1, 0, 0.1]),
    "camera_5": look_at([0.5, 0.3, 0.6], [-0.2, 0, 0.1]),
}
CAMERA_W = 224
CAMERA_H = 224

NATURAL_INSTRUCTIONS = {
    "PickCube-v0": "pick up the red cube and move it to the green point",
    "StackCube-v0": "stack the red cube on the green cube",
    "PickSingleYCB-v0": "pick up the ",
    # "PickSingleEGAD-v0": "Pick up an EGAD object and move it to a goal position",
    "PegInsertionSide-v0": "insert the peg into the horizontal hole in the box",
    # "PlugCharger-v0": "Plug a charger into a wall receptacle",
    "AssemblingKits-v0": "insert the objects into the corresponding holes on the plate",
    # "TurnFaucet-v0": "Turn on a faucet by rotating its handle",
    # "PandaAvoidObstacles-v0": "Navigate the (Panda) robot arm through a region of dense obstacles and move the end-effector to a goal pose",
    # "PickClutterYCB-v0": "Pick up an object from a clutter of 4-8 YCB objects",
}
CAMERA_POOL_FILE = "/xxx/xxx/share_data/Anonymous/maniskill2/camera_pool_300k.npz"
camera_pool = np.load(CAMERA_POOL_FILE)["cameras"]

class PytorchDiffInference(nn.Module):
    def __init__(self, model, prediction_type='epsilon',sequence_length = 15, 
                 use_wrist_img=False, device="cuda", 
                 stride=1, num_pred_action=4,
                 use_action_head_diff=0, use_euler=0):
        super().__init__()

        self.device = torch.device(device)

        # use_wrist_img = use_wrist_img
        use_depth_img = False
        self.use_euler = use_euler

        self.use_wrist_img = use_wrist_img
        self.use_depth_img = use_depth_img

        self.sequence_length = sequence_length
        self.num_pred_action = num_pred_action
        self.use_action_head_diff = use_action_head_diff

        self.stride = stride

        self.model = model.to(device)

        print('sequence_length:', self.sequence_length)
        try:
            self.use_wrist_img = self.model.module.use_wrist_img
            self.use_depth_img = self.model.module.use_depth_img
        except:
            self.use_wrist_img = False
            self.use_depth_img = False

        self.model.eval()
        self.model_input = []
        self.observation = []
        self.model_input_wrist = []
        self.model_input_depth = []
        self.instruction = ""
        self.stride = stride
        self.data_transform = torchvision.transforms.Compose(
            [
                # torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ]
        )


        self.data_transform_eval = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    # v2.RandomResizedCrop(size=(224, 224), antialias=True),
                    
                    torchvision.transforms.CenterCrop(size=(480, 480)),
                    torchvision.transforms.Resize((224, 224), antialias=True),
                    
                    torchvision.transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                    # torchvision.transforms.Pad(padding = (80,1,80,0)),
                    # torchvision.transforms.Resize((224,224), antialias=True)
                ]
            )
        
        self.data_transform_eval1 = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    # v2.RandomResizedCrop(size=(224, 224), antialias=True),
                    
                    torchvision.transforms.CenterCrop(size=(480, 480)),
                    torchvision.transforms.Resize((224, 224), antialias=True),
                    
                    # torchvision.transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                    # torchvision.transforms.Pad(padding = (80,1,80,0)),
                    # torchvision.transforms.Resize((224,224), antialias=True)
                ]
            )

        self.clip_tokenizer = AutoTokenizer.from_pretrained(
            "/xxx/xxx/share_data/Anonymous/clip-vit-large-patch14/", use_fast=False
        )
        self.clipmodel = CLIPModel.from_pretrained(
            "/xxx/xxx/share_data/Anonymous/clip-vit-large-patch14/"
        ).to(self.device)
        
    # clip_tokenizer = AutoTokenizer.from_pretrained(
    #     "/xxx/xxx/share_data/Anonymous/clip-vit-large-patch14/", use_fast=False
    # )
    # clipmodel = CLIPModel.from_pretrained("/xxx/xxx/share_data/Anonymous/clip-vit-large-patch14/").to(DEVICE)

        # self.to(self.device)
        self.frame = 0
        
        # self.base_episode = pkl.load(open("/xxx/xxx/Anonymous/project/embodied_foundation/PickCube-v0_traj_0_camera_0.pkl", "rb"))
        # self.episode = pkl.load(open("/xxx/xxx/Anonymous/project/embodied_foundation/PickCube-v0_traj_0_camera_1.pkl", "rb"))
        self.eef_pose = None
        self.empty_instruction = None
        # model_output: dx dy dz dqw dqx dqy dqz terminate

    def set_natural_instruction(self, instruction: str):
        inputs = self.clip_tokenizer(text=instruction, return_tensors="pt", max_length=77, padding="max_length",  truncation=True)
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        with torch.no_grad():
            text_embeddings = self.clipmodel.text_model(**inputs)[0].squeeze(0).detach()
        self.instruction = text_embeddings

    def set_eef_pose(self, eef_pose):
        self.eef_pose = eef_pose

    def set_observation(self, rgb, depth=None, wrist=None, is_real_eval=False):
        assert (rgb >= 0).all() and (rgb <= 255).all()
        self.observation.append(rgb)
        if self.model_input == []:
            # import ipdb;ipdb.set_trace()
            if is_real_eval == 1:
                rgb_data = self.data_transform_eval(rgb).to(self.device, non_blocking=True)

                # rgb_data = (rgb_data.permute(0, 2,3,1)*255).cpu().to(torch.uint8)
                # rgb_data = (rgb_data / 255. - torch.tensor(IMAGENET_DEFAULT_MEAN)) / torch.tensor(IMAGENET_DEFAULT_STD)
                # rgb_data = rgb_data.permute(0,3, 1, 2)
                print(rgb_data)
            elif is_real_eval == 2:
                # rgb_data = self.data_transform_eval1(rgb).to(self.device, non_blocking=True)

                # rgb_data = (rgb_data.permute(0, 2,3,1)*255).cpu().to(torch.uint8)
                rgb_data = (torch.tensor(rgb) / 255. - torch.tensor(IMAGENET_DEFAULT_MEAN)) / torch.tensor(IMAGENET_DEFAULT_STD)
                rgb_data = rgb_data.permute(2, 0, 1).to(self.device, non_blocking=True)
                
            else:
                rgb = torch.tensor(rgb).to(self.device, non_blocking=True)
                rgb_data = self.data_transform((rgb / 255.0).permute(2, 0, 1).contiguous())
            self.model_input = rgb_data.unsqueeze(0)
            if len(self.model_input) < self.sequence_length:
                self.model_input = self.model_input.repeat(self.sequence_length, 1, 1, 1)
        else:
            if is_real_eval == 1:
                rgb_data = self.data_transform_eval(rgb).to(self.device, non_blocking=True)
            elif is_real_eval == 2:
                # rgb_data = self.data_transform_eval1(rgb).to(self.device, non_blocking=True)
                print(rgb.shape)
                # rgb_data = (rgb_data.permute(0, 2,3,1)*255).cpu().to(torch.uint8)
                rgb_data = (torch.tensor(rgb)/ 255. - torch.tensor(IMAGENET_DEFAULT_MEAN)) / torch.tensor(IMAGENET_DEFAULT_STD)
                rgb_data = rgb_data.permute(2,0,1).to(self.device, non_blocking=True) 
            else:
                rgb = torch.tensor(rgb).to(self.device, non_blocking=True)
                rgb_data = self.data_transform((rgb / 255.0).permute(2, 0, 1).contiguous())
            self.model_input = torch.cat((self.model_input, rgb_data.unsqueeze(0)), dim=0)
            self.model_input = self.model_input[-self.sequence_length :]

        if wrist is not None and self.use_wrist_img:
            if self.model_input_wrist == []:

                wrist_data = torch.tensor(wrist).to(self.device, non_blocking=True)

                wrist_data = self.data_transform((wrist_data / 255.0).permute(2, 0, 1).contiguous())

                self.model_input_wrist = wrist_data.unsqueeze(0)
                if len(self.model_input_wrist) < self.sequence_length:
                    self.model_input_wrist = self.model_input_wrist.repeat(self.sequence_length, 1, 1, 1)
            else:

                wrist_data = torch.tensor(wrist).to(self.device, non_blocking=True)

                wrist_data = self.data_transform((wrist_data / 255.0).permute(2, 0, 1).contiguous())

                self.model_input_wrist = torch.cat((self.model_input_wrist, wrist_data.unsqueeze(0)), dim=0)
                self.model_input_wrist = self.model_input_wrist[-self.sequence_length :]
            wrist = (
                nn.functional.interpolate(torch.tensor(wrist).permute(2, 0, 1).unsqueeze(0), size=(224, 224), mode="nearest")
                .squeeze()
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            self.observation[-1] = np.concatenate([self.observation[-1], wrist], axis=1)
        if depth is not None and self.use_depth_img:
            if self.model_input_depth == []:

                depth_data = torch.tensor(depth / 10).to(self.device, non_blocking=True)
                self.model_input_depth = depth_data.unsqueeze(0)
            else:
                depth_data = torch.tensor(depth / 10).to(self.device, non_blocking=True)
                self.model_input_depth = torch.cat((self.model_input_depth, depth_data.unsqueeze(0)), dim=0)
                self.model_input_depth = self.model_input_depth[-self.sequence_length :]
            depth = torch.tensor(depth / 10 * 255).repeat(1, 1, 3).byte().cpu().numpy()
            self.observation[-1] = np.concatenate([self.observation[-1], depth], axis=1)

    def reset_observation(self):
        self.model_input = []
        self.observation = []
        self.model_input_wrist = []
        self.model_input_depth = []
        self.frame = 0

                
    def save_video(self, fpath):
        # height, width, _ = self.observation[0].shape
        # fourcc = cv2.VideoWriter_fourcc(*"AVC1")
        # if os.path.exists(fpath):
        #     os.remove(fpath)
        # out = cv2.VideoWriter(fpath, fourcc, 10.0, (width, height))
        # for image in self.observation:
        #     out.write(image)  # Write out frame to video
        # out.release()
        from moviepy.editor import ImageSequenceClip
        clip = ImageSequenceClip(self.observation, fps=10 / self.stride)
        clip.write_videofile(fpath, codec="libx264", audio=False, logger=None)  # Use 'libx264' for the H.264 codec

    def calc_act(self, base_episode, camera_extrinsic_cv, current_frame_idx):
        try:
            pose1 = torch.tensor(base_episode["step"][current_frame_idx]["prev_ee_pose"]).clone()
            # pose2 = torch.tensor(base_episode["step"][current_frame_idx]["target_ee_pose"]).clone()
            pose2 = torch.tensor(base_episode["step"][current_frame_idx + self.stride]["prev_ee_pose"]).clone()
        except:
            current_frame_idx = min(current_frame_idx, len(base_episode["step"]) - 1)
            pose1 = torch.tensor(base_episode["step"][current_frame_idx]["prev_ee_pose"]).clone()
            # pose2 = torch.tensor(base_episode["step"][current_frame_idx]["target_ee_pose"]).clone()
            pose2 = torch.tensor(base_episode["step"][-1]["prev_ee_pose"]).clone()

        pose1[0] -= 0.615  # base to world
        pose2[0] -= 0.615  # base to world
        action = {}
        action["world_vector"], action["rotation_delta"] = process_traj_v3(
            (camera_extrinsic_cv),
            pose1,
            pose2,
        )

        if base_episode["step"][current_frame_idx]["is_terminal"] == True:
            action["terminate_episode"] = torch.tensor([1, 0, 0], dtype=torch.int32)
        else:
            action["terminate_episode"] = torch.tensor([0, 1, 0], dtype=torch.int32)
        action["gripper_closedness_action"] = torch.tensor(
            base_episode["step"][current_frame_idx]["action"][-1],
            dtype=torch.float32,
        ).unsqueeze(-1)

        return action

    def get_target_pose(self, delta_pos, delta_rot):
        target_ee_pose_at_camera = sapien.Pose(p=self.eef_pose.p + delta_pos)
        r_prev = quat2mat(self.eef_pose.q)
        r_diff = quat2mat(delta_rot)
        r_target = r_diff @ r_prev
        target_ee_pose_at_camera.set_q(mat2quat(r_target))

        return target_ee_pose_at_camera
 
    def inference(self, extrinsics=None, abs_pose=0, abs_seq_pose=0, horizon=-1, set_pose=False, trajectory_dim=11, reg_prediction_nums=0, pad_diff_nums=0, obs_pose=None, cfg=0, ret_7=False, dim=0):
        # import ipdb;ipdb.set_trace()

        obs = {"image": self.model_input[- (self.sequence_length - self.num_pred_action + 1):].unsqueeze(0)}
        if self.use_wrist_img:
            obs["wrist_image"] = self.model_input_wrist[-self.sequence_length :].unsqueeze(0)
        if self.use_depth_img:
            obs["depth_image"] = self.model_input_depth[-self.sequence_length :].unsqueeze(0)
        obs["natural_language_embedding"] = self.instruction[None, None, ...].repeat(1, obs["image"].shape[1], 1, 1)

        if cfg != 0:
            # classifier free guidance
            if self.empty_instruction is None:
                inputs = self.clip_tokenizer(text='', return_tensors="pt", max_length=77, padding="max_length")
                for key in inputs:
                    inputs[key] = inputs[key].to(self.device)
                with torch.no_grad():
                    self.empty_instruction = self.clipmodel.text_model(**inputs)[0].squeeze(0).detach()
                self.empty_instruction = self.empty_instruction[None, None, ...].repeat(1, obs["image"].shape[1], 1, 1)
            obs['natural_language_embedding'] = torch.cat([self.empty_instruction, obs['natural_language_embedding'], ], dim=0)
            obs["image"] = torch.cat([obs["image"], obs["image"]], dim=0)

        if obs_pose is not None:
            obs['poses'] = obs_pose.to(obs["natural_language_embedding"].device)  # B 1 11
# Image.fromarray(((unnormalize(obs['image'][0, 1].permute(1,2,0)))*255).cpu().numpy().astype(np.uint8)).save('temp_tttt_{}_r.png'.format(0))
        # with torch.cuda.amp.autocast():
        with torch.no_grad():
            # 1 x T x L x C
            if 'DEBUG' in os.environ: import ipdb;ipdb.set_trace()
            if self.use_action_head_diff in [2, 3, 4, 5]:
                if hasattr(self.model, 'module'):
                    model_output = self.model.module.inference_withfeats(obs, num_pred_action=self.num_pred_action, abs_pose=abs_pose, horizon=horizon, reg_prediction_nums=reg_prediction_nums, pad_diff_nums=pad_diff_nums, cfg=cfg)
                else:
                    model_output = self.model.inference_withfeats(obs, num_pred_action=self.num_pred_action, abs_pose=abs_pose, horizon=horizon, reg_prediction_nums=reg_prediction_nums, pad_diff_nums=pad_diff_nums, cfg=cfg)
            else:
                if hasattr(self.model, 'module'):
                    model_output = self.model.module.inference(obs, num_pred_action=self.num_pred_action, abs_pose=abs_pose, horizon=horizon, reg_prediction_nums=reg_prediction_nums, pad_diff_nums=pad_diff_nums, cfg=cfg)
                else:
                    model_output = self.model.inference(obs, num_pred_action=self.num_pred_action, abs_pose=abs_pose, horizon=horizon, reg_prediction_nums=reg_prediction_nums, pad_diff_nums=pad_diff_nums, cfg=cfg)
            # model_output = self.model.module.inference1(obs, num_pred_action=self.num_pred_action, abs_pose=abs_pose, horizon=horizon)
            # 
        if 'DEBUG' in os.environ: import ipdb;ipdb.set_trace()
        if ret_7:
            # this is for openx data
            output = torch.cat(
                [
                    model_output["world_vector"].cpu(),
                    model_output["rotation_delta"].cpu(),
                    # torch.tensor(np.stack([tp.p for tp in target_pose]))[None,...],
                    # torch.tensor(np.stack([tp.q for tp in target_pose]))[None,...],
                    model_output["gripper_closedness_action"].cpu(),
                ], dim=-1
            )[0]
            self.frame += self.stride
            return output.cpu().numpy()
        elif trajectory_dim == 7:
            if self.use_euler:
                model_output['world_vector'] = (model_output['world_vector'] + 1)*0.0768 - 0.0768
                model_output['rotation_delta'] = (model_output['rotation_delta'] + 1)*0.0768 - 0.0768
                pass
            assert set_pose
            # this is for openx data
            rot_delta = model_output["rotation_delta"][0]
            def euler_to_quaternion(eulers):
                import scipy.spatial.transform as st
                quaternion = st.Rotation.from_euler('xyz', eulers).as_quat()
                return torch.tensor([quaternion[-1], quaternion[0], quaternion[1], quaternion[2]])
            quat_list = torch.stack([euler_to_quaternion(rot_delta[i].cpu().numpy()) for i in range(len(rot_delta))])[None,...]
            import numpy as np
            output = torch.cat(
                [
                    model_output["world_vector"].cpu(),
                    quat_list.cpu(),
                    # torch.tensor(np.stack([tp.p for tp in target_pose]))[None,...],
                    # torch.tensor(np.stack([tp.q for tp in target_pose]))[None,...],
                    model_output["gripper_closedness_action"].cpu(),
                    model_output["terminate_episode"][:,:, [0]].cpu()
                ], dim=-1
            )[0]
            self.frame += self.stride
            return output.cpu().numpy()
            pass
        elif abs_seq_pose:
            # convert to quat

            model_output["rotation_delta"][0,:, 0] += 1
            # print(model_output["world_vector"], model_output["rotation_delta"])
            target_pose = [self.get_target_pose(model_output["world_vector"][0,tp].cpu().numpy(), model_output["rotation_delta"][0, tp].cpu().numpy()) 
                           for tp in range(model_output["world_vector"].shape[1])]
            import numpy as np
            output = torch.cat(
                [
                    # model_output["world_vector"],
                    # model_output["rotation_delta"],
                    torch.tensor(np.stack([tp.p for tp in target_pose]))[None,...],
                    torch.tensor(np.stack([tp.q for tp in target_pose]))[None,...],
                    model_output["gripper_closedness_action"].cpu(),
                    model_output["terminate_episode"][:,:, [0]].cpu()
                ], dim=-1
            )[0]
            pass
        elif not abs_pose:
            if not set_pose:
                model_output["rotation_delta"][0,:, 0] += 1
                # print(model_output["world_vector"], model_output["rotation_delta"])
                target_pose = [self.get_target_pose(model_output["world_vector"][0,tp].cpu().numpy(), model_output["rotation_delta"][0, tp].cpu().numpy()) 
                            for tp in range(model_output["world_vector"].shape[1])]
                import numpy as np
                output = torch.cat(
                    [
                        # model_output["world_vector"],
                        # model_output["rotation_delta"],
                        torch.tensor(np.stack([tp.p for tp in target_pose]))[None,...],
                        torch.tensor(np.stack([tp.q for tp in target_pose]))[None,...],
                        model_output["gripper_closedness_action"].cpu(),
                        model_output["terminate_episode"][:,:, [0]].cpu()
                    ], dim=-1
                )[0]
            else:
                model_output["rotation_delta"][0,:, 0] += 1
                # print(model_output["world_vector"], model_output["rotation_delta"])
                # target_pose = [self.get_target_pose(model_output["world_vector"][0,tp].cpu().numpy(), model_output["rotation_delta"][0, tp].cpu().numpy()) 
                            # for tp in range(model_output["world_vector"].shape[1])]
                import numpy as np
                output = torch.cat(
                    [
                        model_output["world_vector"].cpu(),
                        model_output["rotation_delta"].cpu(),
                        # torch.tensor(np.stack([tp.p for tp in target_pose]))[None,...],
                        # torch.tensor(np.stack([tp.q for tp in target_pose]))[None,...],
                        model_output["gripper_closedness_action"].cpu(),
                        model_output["terminate_episode"][:,:, [0]].cpu()
                    ], dim=-1
                )[0]
            # output = torch.cat(
            #     [
            #         model_output["world_vector"].cpu(),
            #         model_output["rotation_delta"].cpu(),
            #         model_output["gripper_closedness_action"].cpu(),
            #         model_output["terminate_episode"][:,:, [0]].cpu()
            #     ], dim=-1
            # )[0]
            # print(output[...,:7], output[...,:7].min(), output[...,:7].max(),"output")
        else:
            output = torch.cat(
                [
                    model_output["world_vector"].cpu(),
                    model_output["rotation_delta"].cpu(),
                    model_output["gripper_closedness_action"].cpu(),
                    model_output["terminate_episode"][:,:, [0]].cpu()
                ], dim=-1
            )[0]
            pass

        # import ipdb; ipdb.set_trace()
        # output = torch.diagonal(model_output, dim1=1, dim2=2).squeeze(0)

        # add 1 to quat
        output[..., -2] = (output[...,-2] > 0.0).float() * 2 - 1
        # output[..., 3] += 1
        # print('traj term:', output[..., -1])
        # import ipdb;ipdb.set_trace()
        output[..., -1] = output[..., -1] > 0.5
        # output[..., -1] = torch.logical_and(torch.logical_and(output[..., -1] > 0.5, model_output["terminate_episode"][0,:, 1].cpu() < 0.5), model_output["terminate_episode"][0,:, 2].cpu() < 0.5).float()
        # print(output[..., -1].shape)
        # gt_output = self.calc_act(self.base_episode, torch.tensor(self.episode["camera_extrinsic_cv"]), self.frame)
        # gt_output = self.calc_act(self.base_episode, torch.eye(4), self.frame)

        # gt_output = torch.cat(
        #     [
        #         gt_output["world_vector"],
        #         gt_output["rotation_delta"],
        #         gt_output["gripper_closedness_action"],
        #         gt_output["terminate_episode"][[0]],
        #     ]
        # )

        # print(list(output.cpu().numpy() - gt_output.cpu().numpy()))
        # print(f"eval frame {self.frame}", flush=True)
        # Image.fromarray((unnormalize(self.model_input[-1].permute(1, 2, 0)).clamp(0, 1) * 255).byte().cpu().numpy()).save(f"obs_0.png")
        
        # self.frame += 1
        self.frame += self.stride

        # import ipdb

        # ipdb.set_trace()

        return output.cpu().numpy()

def analyze_traj_str(traj_str):

    env_id = traj_str.split("-")[0] + "-v0"
    select_camera = f"camera_{traj_str[-5]}"
    seed_start_pos = traj_str.find("traj") + 5
    seed = int(traj_str[seed_start_pos:-13])
    return env_id, select_camera, seed


def close_loop_eval_v2(
    obs_mode="rgbd",
    reward_mode=None,
    control_mode=TARGET_CONTROL_MODE,
    render_mode="cameras",
    record_dir=None,
    render_goal_point=True,
    test_episodes_num=100,
    model=None,
    eval_data_list=None,
    args=None,
    rand_seed=0,
    json_repo="/xxx/xxx/share_data/Anonymous/maniskill2/demos/v0/rigid_body/",
    camera_coord=True,
    stride=1,
    root_folder = None,
    data_root_path = None,
    cfg=None,
    eval_dataset=None,
):
    if 'noseed' not in cfg:
        print('fix seed-----------------------------------')
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
    else:
        print('not fix seed')

    if 'input_size' in cfg:
        global CAMERA_W, CAMERA_H
        import ast
        # img_size = ast.literal_eval(cfg.input_size)
        CAMERA_W = int(cfg.input_size)
        CAMERA_H = int(cfg.input_size)

    assert cfg is not None
    print('begin ....', root_folder)
    # import pdb;pdb.set_trace()
    client = Client()
    np.set_printoptions(suppress=True, precision=3)
    # print(eval_data_list)
    eval_traj_list = pickle.load(open(eval_data_list, "rb"))
    if len(eval_traj_list) > 500:
        eval_traj_list = [item for item in eval_traj_list if not item.__contains__('PegInsertionSide-v0')]
        eval_traj_list = eval_traj_list[:-88]
    # eval_traj_list = [item for item in eval_traj_list if not item.__contains__('AssemblingKits-v0')]

    # data_url = os.path.join(self.data_path, self.data_cam_list[index])
    np.random.seed(0 % 9973)
    eval_traj_index = np.random.permutation(len(eval_traj_list))[: args.world_size * test_episodes_num]
    eval_traj_index = eval_traj_index[args.rank * test_episodes_num : (args.rank + 1) * test_episodes_num]
    eval_traj_index = sorted(eval_traj_index)
    # import ipdb;ipdb.set_trace()

    extrinsics_pool = np.asarray([[[-4.4721359e-01, 8.9442706e-01,-1.4901161e-08, -4.4721358e-02], [ 6.6666663e-01, 3.3333331e-01,-6.6666663e-01, 1.3333333e-01], 
                                   [-5.9628463e-01,-2.9814237e-01,-7.4535596e-01, 6.8572754e-01], [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00,]],
                                   [[ 4.4721359e-01, 8.9442706e-01, 1.4901161e-08,  4.4721358e-02], [ 6.6666663e-01,-3.3333331e-01,-6.6666663e-01, 1.3333333e-01],
                                     [-5.9628463e-01, 2.9814237e-01,-7.4535596e-01, 6.8572754e-01],[ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00,]],
                                     [[-4.4721359e-01, 8.9442718e-01, 3.7252903e-09, -4.4721343e-02],[ 1.9518001e-01, 9.7590014e-02, -9.7590005e-01, 3.1228808e-01],
                                      [-8.7287164e-01,-4.3643576e-01,-2.1821789e-01, 4.3643579e-01],[ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00,]],
                                      [[ 3.1622776e-01, 9.4868326e-01,-7.4505806e-09,  3.1622782e-02],[ 7.0392162e-01,-2.3464054e-01,-6.7040157e-01, 1.3743240e-01],
                                       [-6.3599873e-01, 2.1199957e-01,-7.4199849e-01, 9.5399815e-01],[ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],
                                       [[-3.9391929e-01, 9.1914511e-01, 7.4505806e-09, -7.8783855e-02],[ 5.0444633e-01, 2.1619129e-01,-8.3593971e-01, 1.8448329e-01],
                                        [-7.6834989e-01,-3.2929277e-01,-5.4882127e-01, 8.1225550e-01],[ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]]])
    success_num = {"PickCube-v0" : 0, "PickSingleYCB-v0" : 0, "StackCube-v0" : 0}
    success_num = {"PickCube-v0" : 0.0, "PickSingleYCB-v0" : 0.0, "StackCube-v0" : 0.0, "PickClutterYCB-v0": 0.0,
                   "AssemblingKits-v0" : 0.0, "PickSingleEGAD-v0": 0.0}
    # success_num = {'StackCube', 'PickSingleYCB', 'PickClutterYCB', 'AssemblingKits', 'PegInsertionSide', 'PickSingleEGAD', 'PickCube'}
    # total_num = {"PickCube-v0" : 0, "PickSingleYCB-v0" : 0, "StackCube-v0" : 0}
    success_list = []
    i = 0
    print('begin 2....')
    print(cfg)
    model = PytorchDiffInference(model=model, sequence_length = cfg.dataset.traj_length, 
                                 num_pred_action=cfg.num_pred_action, stride=stride, use_action_head_diff=cfg.use_action_head_diff, use_euler=cfg.dataset.use_euler if 'use_euler' in cfg.dataset else 0)
    print('begin init model....')
    print('A:', i, args.rank, len(eval_traj_index), len(eval_traj_list))
    print('--------------------')
    print(eval_traj_index)
    traj_str = eval_traj_list[eval_traj_index[i]]
    env_id, select_camera, seed, reset_kwargs, instruction = analyze_traj_str_v2(client, traj_str, json_repo, data_root_path)
    data_tmp_ = pickle.loads(client.get(os.path.join(data_root_path, traj_str)))

    def get_changed_pose(eval_dataset, data_path_, end_chaged_pose=False):
        # data_path_ = 'PickClutterYCB-v0/PickClutterYCB-v0_traj_3472_camera_11/data.pkl'
        data_url = os.path.join(eval_dataset.data_path, data_path_)
        data_pkl = pickle.loads(eval_dataset.client.get(data_url))
        
        # for i in range(100):
        trajs = eval_dataset.construct_traj(data_pkl, data_path_)
        #     print(i, trajs["action"]['gripper_change_pose'][0])
        if end_chaged_pose:
            target_position_pose = torch.tensor(data_pkl['step'][-1]['prev_ee_pose']).clone()

            target_position_pose[0] -= 0.615
            
            # data_pkl['step'][0]['prev_ee_pose']
            camera_extrinsic_cv = torch.tensor(data_pkl['step'][0]["camera_extrinsic_cv"])
            gripper_change_pose = get_pose_cam(camera_extrinsic_cv if cfg.dataset.use_baseframe_action == False else torch.eye(4), target_position_pose)
            
            t_gripper_position = torch.tensor(data_pkl['step'][-1]["action"][-1], dtype=torch.float32,).unsqueeze(-1)
            t_terminate_episode = torch.tensor([1, 0, 0], dtype=torch.int32)
            gripper_change_pose = torch.cat([gripper_change_pose, t_gripper_position, t_terminate_episode], dim=-1)
            
            return gripper_change_pose
        # print(trajs['action']['world_vector'], trajs["action"]['gripper_closedness_action'], trajs["action"]['gripper_change_pose'][0])
        return trajs["action"]['gripper_change_pose'][0]
        pass

    obs_pose = None
    if 'obs_poses' in cfg and cfg.obs_poses in [111]:
        # zeros inference for large gaussian noise
        obs_pose = get_changed_pose(eval_dataset, traj_str)[None, None,]
        obs_pose = torch.randn_like(obs_pose)

    if 'obs_poses' in cfg and cfg.obs_poses in [2, 3, 5, 8]:
        obs_pose = get_changed_pose(eval_dataset, traj_str)[None, None,]
        if cfg.obs_poses == 3:
            obs_pose = torch.zeros_like(obs_pose)
        elif cfg.obs_poses == 5:
            obs_pose[:, :, 7:] = torch.zeros_like(obs_pose[:, :, 7:])
        elif cfg.obs_poses == 8:
            obs_pose[:, :, 3:] = torch.zeros_like(obs_pose[:, :, 3:])
    # env_id = 'PickCube-v0'; select_camera = 1; seed = 982; reset_kwargs = {}
    fix_camera_pool = {}
    fix_camera_pool["camera_1"] = look_at([0.3, 0.2, 0.6], [-0.1, 0, 0.1])
    fix_camera_pool["camera_2" ] = look_at([0.3, -0.2, 0.6], [-0.1, 0, 0.1])
    fix_camera_pool["camera_3"] = look_at([0.3, 0.2, 0.4], [-0.1, 0, 0.3])
    fix_camera_pool["camera_4"] = look_at([0.5, -0.2, 0.8], [-0.1, 0, 0.1])
    fix_camera_pool["camera_5"] = look_at([0.5, 0.3, 0.6], [-0.2, 0, 0.1])  
    
    
    if 'fix_camera' in cfg and cfg.fix_camera:
        for ii in range(0, 5):
            if (extrinsics_pool[ii].astype(np.float32) == data_tmp_['step'][0]['camera_extrinsic_cv']).all():
                select_camera = ii + 1
                break
        # select_camera = traj_str.split('camera_')[1][:1]
        camera_pose = fix_camera_pool["camera_{}".format(select_camera)]
    else:
        camera_pose = look_at(camera_pool[select_camera][:3], camera_pool[select_camera][3:6], camera_pool[select_camera][6:9])
    # import pdb;pdb.set_trace()
    env: BaseEnv = gym.make(
        env_id,
        renderer_kwargs={"offscreen_only": True, "device": f"cuda:{args.local_rank}"},
        obs_mode=obs_mode,
        reward_mode=reward_mode,
        control_mode=control_mode,
        render_mode=render_mode,
        render_camera_cfgs=dict(width=2 * CAMERA_W, height=2 * CAMERA_H),
        camera_cfgs=dict(
            base_camera=dict(p=camera_pose.p, q=camera_pose.q, width=CAMERA_W, height=CAMERA_H),
            hand_camera=dict(width=128, height=128),
        ),
        max_episode_steps=MAX_EPISODE_STEPS * 100,
    )
    print('begin gym....')

    if record_dir != None:
        record_dir = record_dir.format(env_id=env_id)
        env = RecordEpisode(env, record_dir, render_mode=render_mode)
    if render_goal_point and hasattr(env, "goal_site"):
        env.goal_site.unhide_visual()

    current_pose_queue = []
    obs, _ = env.reset(seed=seed, options=reset_kwargs)
    if 'obs_poses' in cfg and cfg.obs_poses in [1, 4, 7, 11]:
        cur_ee_pose = eef_pose(env, extrinsic_cv=obs["camera_param"]["base_camera"]["extrinsic_cv"], camera_coord=camera_coord)
        cur_eef_pose_7d = np.concatenate([cur_ee_pose.p, cur_ee_pose.q], axis=-1)
        cur_eef_pose_7d = torch.tensor(cur_eef_pose_7d)
        cur_eef_pose_7d[0] -= 0.615
        if 'abs_pose' in cfg and cfg.abs_pose == 2:
            tmp_m = quaternion_to_matrix(torch.tensor(cur_ee_pose.q))
            euler_rot = matrix_to_euler_angles(tmp_m, convention='XYZ').cpu().numpy()
            cur_eef_pose_7d = np.concatenate([cur_ee_pose.p, euler_rot], axis=-1)
            cur_eef_pose_7d = torch.tensor(cur_eef_pose_7d)
            pass
        # import ipdb;ipdb.set_trace()
        #  if cfg.dataset.use_baseframe_action == False else torch.eye(4)
        t_ = torch.tensor([0, 1, 0], dtype=torch.int32)
        cur_abs_pose = torch.cat([cur_eef_pose_7d, torch.tensor([1], dtype=torch.float32), t_],dim=-1)[None, None,]
        current_pose_queue.append(cur_abs_pose)
        current_pose_queue.append(cur_abs_pose)
        if cfg.obs_poses == 4:
            obs_pose = cur_abs_pose
        elif cfg.obs_poses in [1, 7]:
            obs_pose = get_changed_pose(eval_dataset, traj_str)[None, None,]
            obs_pose[:, :, 7:] = torch.zeros_like(obs_pose[:, :, 7:])
            obs_pose = torch.cat([cur_abs_pose, obs_pose], dim=1)
        elif cfg.obs_poses == 11:
            obs_pose = torch.cat([current_pose_queue[-2], current_pose_queue[-1]], dim=1)


    print('begin reset....')
    model.set_natural_instruction(instruction)
    # total_num = 0
    # start_time = time.time()

    if root_folder != None:
        os.makedirs(root_folder, exist_ok = True) 

    print('begin tqdm....')
    import tqdm
    pbar = tqdm.tqdm(total=test_episodes_num)
    print('begin....')
    model.set_eef_pose(eef_pose(env, obs["camera_param"]["base_camera"]["extrinsic_cv"] , camera_coord=camera_coord))
    model.set_observation(rgb=obs["image"]["base_camera"]["rgb"], wrist=obs["image"]["hand_camera"]["rgb"])
    # o o o o o o
    #   a a a a
    #   o o o o o o
    #     a a a a

    # o o o
    #   a
    # o o o o
    #     a
    # o o o o o
    #       a

    # o o n n n n
    #   a a a a
    #         o o
    #           a a a a

    moving_action = False
    moving_model_output = None

    pad_diff_nums = 0
    reg_prediction_nums = 1 if cfg.abs_sup in [3, 4] else 0
    if cfg.abs_sup in [3, 4] and ('no_abs_diff' not in cfg or not cfg.no_abs_diff):
        pad_diff_nums = 1
    if cfg.abs_sup and 'obs_poses' in cfg and cfg.obs_poses and 'no_abs_diff' in cfg and cfg.no_abs_diff:
        reg_prediction_nums = 1

    is_initial = True # the first step 
    step_i = 0
    while i < test_episodes_num and i < len(eval_traj_index):
        # total_num += 1
        # if total_num >500:
        #     break
        is_episode_end = False
        model_output = model.inference(obs["camera_param"]["base_camera"]["extrinsic_cv"] if cfg.dataset.use_baseframe_action == False else torch.eye(4), abs_pose= cfg.abs_pose if 'abs_pose' in cfg else 0, set_pose=True, 
                                       trajectory_dim=cfg.trajectory_dim, reg_prediction_nums=reg_prediction_nums, pad_diff_nums=pad_diff_nums, obs_pose=obs_pose, cfg = cfg.cfg if 'cfg' in cfg else 0)

        if moving_model_output is None:
            moving_model_output = model_output.copy()
        else:
            lambda_1 = (np.arange(len(moving_model_output), 0, -1)/ len(moving_model_output) * 0.5)
            for pi in range(len(moving_model_output)):
                moving_model_output[pi][:7] = moving_model_output[pi][:7] * lambda_1[pi] + model_output[pi][:7] * (1 - lambda_1[pi])


        for iiii in range(len(model_output[:cfg.n_action_steps])):
            
            model_output_new = model_output[iiii]
            
            # print(model_output_new.shape)
            if not ('abs_pose' in cfg and cfg.abs_pose):
                # torch.tensor(np.stack([tp.p for tp in target_pose]))[None,...],
                target_pose = model.get_target_pose(model_output_new[:3], model_output_new[3:7]) 
                model_output_new[:3] = target_pose.p
                model_output_new[3:7] = target_pose.q
                pass
            else:
                model_output_new[0] += 0.615
                pass
            if 'abs_pose' in cfg and cfg.abs_pose == 2:
                # euler to quat
                # numpy
                
                r_tmp = matrix_to_quaternion(euler_angles_to_matrix(torch.tensor(model_output_new[3:6]), convention='XYZ'))
                model_output_new = np.concatenate([model_output_new[:3], r_tmp, model_output_new[6:]], axis=-1)
                
                

            delta, loop = np.array([1, 1, 1, 1, 1, 1, 1], dtype=float), 1
            step_i += 1
            
            while np.max(np.abs(delta[:3])) > 1e-4 and loop > 0:
                loop -= 1
                delta = cal_action_from_pose(env, model_output_new[:8], obs["camera_param"]["base_camera"]["extrinsic_cv"], camera_coord=camera_coord)
                ee_action = tuple(delta[:6])
                gripper_action = delta[-1]
                action_dict = dict(arm=ee_action, gripper=gripper_action)
                action = env.agent.controller.from_action_dict(action_dict)

                if render_goal_point and hasattr(env, "goal_site"):
                    env.goal_site.unhide_visual()
                obs, reward, terminated, truncated, info = env.step(action)
            if 'obs_poses' in cfg and cfg.obs_poses in [2, 8] and step_i > 4:
                cur_ee_pose = eef_pose(env, extrinsic_cv=obs["camera_param"]["base_camera"]["extrinsic_cv"], camera_coord=camera_coord)

                should_change_obs_pose = np.linalg.norm(cur_ee_pose.p - obs_pose[0, 0, :3].cpu().numpy(), ord=1) < 1e-1 
                    # \
                        # and np.linalg.norm(cur_ee_pose.q - obs_pose[3:7].cpu().numpy(), ord=1) < 1e-3 
                # import ipdb;ipdb.set_trace()
                # print(step_i, model_output_new[:8], should_change_obs_pose, cur_ee_pose.p, obs_pose[0, 0, :3], camera_coord, np.linalg.norm(cur_ee_pose.p - obs_pose[0,0,:3].cpu().numpy(), ord=1))
                # import ipdb;ipdb.set_trace()
                if model_output_new[7] == -1: # TODO it is better to change the fixed value.
                    # if should_change_obs_pose:
                    obs_pose = get_changed_pose(eval_dataset, traj_str, True)[None, None,]
                    # print('change obs pose', step_i, obs_pose)
                else:
                    # 1
                    obs_pose = get_changed_pose(eval_dataset, traj_str)[None, None,]
                if cfg.obs_poses == 8:
                    obs_pose[:, :, 3:] = torch.zeros_like(obs_pose[:, :, 3:])
                    # print('change obs pose', step_i, obs_pose)
            if 'obs_poses' in cfg and cfg.obs_poses in [1, 4, 7, 11]:
                cur_ee_pose = eef_pose(env, extrinsic_cv=obs["camera_param"]["base_camera"]["extrinsic_cv"], camera_coord=camera_coord)
                cur_eef_pose_7d = np.concatenate([cur_ee_pose.p, cur_ee_pose.q], axis=-1)
                cur_eef_pose_7d = torch.tensor(cur_eef_pose_7d)
                cur_eef_pose_7d[0] -= 0.615
                if 'abs_pose' in cfg and cfg.abs_pose == 2:
                    tmp_m = quaternion_to_matrix(torch.tensor(cur_ee_pose.q))
                    euler_rot = matrix_to_euler_angles(tmp_m, convention='XYZ').cpu().numpy()
                    cur_eef_pose_7d = np.concatenate([cur_ee_pose.p, euler_rot], axis=-1)
                    cur_eef_pose_7d = torch.tensor(cur_eef_pose_7d)
                    pass                
                t_ = torch.tensor([0, 1, 0], dtype=torch.int32)
                cur_abs_pose = torch.cat([cur_eef_pose_7d, torch.tensor([model_output_new[7]], dtype=torch.float32), t_],dim=-1)[None, None,]
                current_pose_queue.append(cur_abs_pose)
                if cfg.obs_poses == 4:
                    obs_pose = cur_abs_pose
                elif cfg.obs_poses == 7:
                    obs_pose = get_changed_pose(eval_dataset, traj_str)[None, None,]
                    # obs_pose[:, :, 7:] = torch.zeros_like(obs_pose[:, :, 7:])
                    obs_pose = torch.cat([cur_abs_pose, obs_pose], dim=1)
                elif cfg.obs_poses == 11:
                    obs_pose = torch.cat([current_pose_queue[-2], current_pose_queue[-1]], dim=1)
                elif cfg.obs_poses == 1:
                    if model_output_new[7] == -1: # TODO it is better to change the fixed value.
                        # if should_change_obs_pose:
                        obs_pose = get_changed_pose(eval_dataset, traj_str, True)[None, None,]
                        # print('change obs pose', step_i, obs_pose)
                    else:
                        # 1
                        obs_pose = get_changed_pose(eval_dataset, traj_str)[None, None,]
                    obs_pose[:, :, 7:] = torch.zeros_like(obs_pose[:, :, 7:])
                    obs_pose = torch.cat([cur_abs_pose, obs_pose], dim=1)
            truncated = model.frame >= (MAX_EPISODE_STEPS * 4 / cfg.n_action_steps)

            if 'DEBUG_STEPS' in os.environ:
                truncated = model.frame >= MAX_EPISODE_STEPS
            # TODO if the model gets stuck, we should try to reset it. e.g., the occlusion.
            # beside, we should expand the action steps when the model suffer from this issue.

            if terminated or truncated:
                success_list.append(info["success"])
                if env_id not in success_num:
                    success_num[env_id] = 0
                success_num[env_id ] += info["success"]
                # total_num[env_id] += 1
                print(i, traj_str, info["success"], flush = True)
                if root_folder != None:
                    model.save_video(os.path.join(root_folder, f'{(i+args.rank * test_episodes_num):04d}_{instruction}_{env_id}_{select_camera}_{info["success"]}.mp4'))
            
                i += 1
                pbar.update(1)
                if i >= test_episodes_num or i >= len(eval_traj_index):
                    is_episode_end = True
                    break

                traj_str = eval_traj_list[eval_traj_index[i]]
                obs_pose = None
                if 'obs_poses' in cfg and cfg.obs_poses in [2, 5, 8]:
                    obs_pose = get_changed_pose(eval_dataset, traj_str)[None, None,]  
                    if cfg.obs_poses == 5:
                        obs_pose[:, :, 7:] = torch.zeros_like(obs_pose[:, :, 7:])
                    elif cfg.obs_poses == 8:
                        obs_pose[:, :, 3:] = torch.zeros_like(obs_pose[:, :, 3:])            
                is_initial = True
                step_i = 0
                env_id_new, select_camera_new, seed_new, reset_kwargs_new, instruction_new = analyze_traj_str_v2(client, traj_str, json_repo, data_root_path)
                data_tmp_ = pickle.loads(client.get(os.path.join(data_root_path, traj_str)))
                if env_id != env_id_new or select_camera != select_camera_new or instruction_new != instruction or reset_kwargs_new != reset_kwargs:
                    env_id = env_id_new
                    select_camera = select_camera_new
                    seed = seed_new
                    reset_kwargs = reset_kwargs_new
                    instruction = instruction_new
                    if 'fix_camera' in cfg and cfg.fix_camera:
                        for ii in range(0, 5):
                            if (extrinsics_pool[ii].astype(np.float32) == data_tmp_['step'][0]['camera_extrinsic_cv']).all():
                                select_camera = ii + 1
                                break
                        # select_camera = traj_str.split('camera_')[1][:1]
                        # select_camera = traj_str.split('camera_')[1][:1]
                        # print(select_camera)
                        camera_pose = fix_camera_pool["camera_{}".format(select_camera)]
                    else:
                        camera_pose = look_at(camera_pool[select_camera][:3], camera_pool[select_camera][3:6], camera_pool[select_camera][6:9])

                    env: BaseEnv = gym.make(
                        env_id,
                        renderer_kwargs={"offscreen_only": True, "device": f"cuda:{args.local_rank}"},
                        obs_mode=obs_mode,
                        reward_mode=reward_mode,
                        control_mode=control_mode,
                        render_mode=render_mode,
                        render_camera_cfgs=dict(width=2 * CAMERA_W, height=2 * CAMERA_H),
                        camera_cfgs=dict(
                            base_camera=dict(p=camera_pose.p, q=camera_pose.q, width=CAMERA_W, height=CAMERA_H),
                            hand_camera=dict(width=128, height=128),
                        ),
                        max_episode_steps=MAX_EPISODE_STEPS * 100,
                    )
                    if record_dir != None:
                        record_dir = record_dir.format(env_id=env_id)
                        env = RecordEpisode(env, record_dir, render_mode=render_mode)
                    # instruction = NATURAL_INSTRUCTIONS[env_id]
                    model.set_natural_instruction(instruction)
                env_id = env_id_new
                select_camera = select_camera_new
                seed = seed_new
                reset_kwargs = reset_kwargs_new
                if render_goal_point and hasattr(env, "goal_site"):
                    env.goal_site.unhide_visual()
                obs, _ = env.reset(seed=seed, options=reset_kwargs)
                model.reset_observation()

                if 'obs_poses' in cfg and cfg.obs_poses in [1, 4, 7, 11]:
                    cur_ee_pose = eef_pose(env, extrinsic_cv=obs["camera_param"]["base_camera"]["extrinsic_cv"], camera_coord=camera_coord)
                    cur_eef_pose_7d = np.concatenate([cur_ee_pose.p, cur_ee_pose.q], axis=-1)
                    cur_eef_pose_7d = torch.tensor(cur_eef_pose_7d)
                    cur_eef_pose_7d[0] -= 0.615
                    if 'abs_pose' in cfg and cfg.abs_pose == 2:
                        tmp_m = quaternion_to_matrix(torch.tensor(cur_ee_pose.q))
                        euler_rot = matrix_to_euler_angles(tmp_m, convention='XYZ').cpu().numpy()
                        cur_eef_pose_7d = np.concatenate([cur_ee_pose.p, euler_rot], axis=-1)
                        cur_eef_pose_7d = torch.tensor(cur_eef_pose_7d)
                        pass                    
                    t_ = torch.tensor([0, 1, 0], dtype=torch.int32)
                    cur_abs_pose = torch.cat([cur_eef_pose_7d, torch.tensor([1], dtype=torch.float32), t_],dim=-1)[None, None,]
                    current_pose_queue = []
                    current_pose_queue.append(cur_abs_pose)
                    current_pose_queue.append(cur_abs_pose)
                    if cfg.obs_poses == 4:
                        obs_pose = cur_abs_pose
                    elif cfg.obs_poses in [1, 7]:
                        obs_pose = get_changed_pose(eval_dataset, traj_str)[None, None,]
                        obs_pose[:, :, 7:] = torch.zeros_like(obs_pose[:, :, 7:])
                        obs_pose = torch.cat([cur_abs_pose, obs_pose], dim=1)
                    elif cfg.obs_poses == 11:
                        obs_pose = torch.cat([current_pose_queue[-2], current_pose_queue[-1]], dim=1)                
                moving_model_output = None
                model.set_eef_pose(eef_pose(env, obs["camera_param"]["base_camera"]["extrinsic_cv"], camera_coord=camera_coord))
                model.set_observation(rgb=obs["image"]["base_camera"]["rgb"], wrist=obs["image"]["hand_camera"]["rgb"])

                break
            model.set_eef_pose(eef_pose(env, obs["camera_param"]["base_camera"]["extrinsic_cv"], camera_coord=camera_coord))
            model.set_observation(rgb=obs["image"]["base_camera"]["rgb"], wrist=obs["image"]["hand_camera"]["rgb"])

        if is_episode_end:
            break
        # model.set_eef_pose(eef_pose(env, obs["camera_param"]["base_camera"]["extrinsic_cv"], camera_coord=camera_coord))
        # model.set_observation(rgb=obs["image"]["base_camera"]["rgb"], wrist=obs["image"]["hand_camera"]["rgb"])
    
    total_success_rate = np.mean(success_list)
    env.close()
    del env
    # end_time = time.time()
    # print(f"time: {end_time - start_time}, i: {i}, total_num: {total_num}")
    return success_num, None, total_success_rate


def close_loop_eval_metaworld(
    obs_mode="rgbd",
    reward_mode=None,
    control_mode=TARGET_CONTROL_MODE,
    render_mode="cameras",
    record_dir=None,
    render_goal_point=True,
    test_episodes_num=100,
    model=None,
    eval_data_list=None,
    args=None,
    rand_seed=0,
    json_repo="/xxx/xxx/share_data/Anonymous/maniskill2/demos/v0/rigid_body/",
    camera_coord=True,
    stride=1,
    root_folder = None,
    data_root_path = None,
    cfg=None,
    eval_dataset=None,
):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    if 'input_size' in cfg:
        global CAMERA_W, CAMERA_H
        import ast
        # img_size = ast.literal_eval(cfg.input_size)
        CAMERA_W = int(cfg.input_size)
        CAMERA_H = int(cfg.input_size)

    assert cfg is not None
    print('begin ....', root_folder)
    # import pdb;pdb.set_trace()
    client = Client()
    np.set_printoptions(suppress=True, precision=3)
    print(eval_data_list)
    eval_traj_list = pickle.load(open(eval_data_list, "rb"))
    if len(eval_traj_list) > 500:
        eval_traj_list = [item for item in eval_traj_list if not item.__contains__('PegInsertionSide-v0')]
        eval_traj_list = eval_traj_list[:-88]
    # eval_traj_list = [item for item in eval_traj_list if not item.__contains__('AssemblingKits-v0')]

    # data_url = os.path.join(self.data_path, self.data_cam_list[index])
    np.random.seed(0 % 9973)
    eval_traj_index = np.random.permutation(len(eval_traj_list))[: args.world_size * test_episodes_num]
    eval_traj_index = eval_traj_index[args.rank * test_episodes_num : (args.rank + 1) * test_episodes_num]
    eval_traj_index = sorted(eval_traj_index)
    # import ipdb;ipdb.set_trace()

    extrinsics_pool = np.asarray([[[-4.4721359e-01, 8.9442706e-01,-1.4901161e-08, -4.4721358e-02], [ 6.6666663e-01, 3.3333331e-01,-6.6666663e-01, 1.3333333e-01], 
                                   [-5.9628463e-01,-2.9814237e-01,-7.4535596e-01, 6.8572754e-01], [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00,]],
                                   [[ 4.4721359e-01, 8.9442706e-01, 1.4901161e-08,  4.4721358e-02], [ 6.6666663e-01,-3.3333331e-01,-6.6666663e-01, 1.3333333e-01],
                                     [-5.9628463e-01, 2.9814237e-01,-7.4535596e-01, 6.8572754e-01],[ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00,]],
                                     [[-4.4721359e-01, 8.9442718e-01, 3.7252903e-09, -4.4721343e-02],[ 1.9518001e-01, 9.7590014e-02, -9.7590005e-01, 3.1228808e-01],
                                      [-8.7287164e-01,-4.3643576e-01,-2.1821789e-01, 4.3643579e-01],[ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00,]],
                                      [[ 3.1622776e-01, 9.4868326e-01,-7.4505806e-09,  3.1622782e-02],[ 7.0392162e-01,-2.3464054e-01,-6.7040157e-01, 1.3743240e-01],
                                       [-6.3599873e-01, 2.1199957e-01,-7.4199849e-01, 9.5399815e-01],[ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],
                                       [[-3.9391929e-01, 9.1914511e-01, 7.4505806e-09, -7.8783855e-02],[ 5.0444633e-01, 2.1619129e-01,-8.3593971e-01, 1.8448329e-01],
                                        [-7.6834989e-01,-3.2929277e-01,-5.4882127e-01, 8.1225550e-01],[ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]]])
    success_num = {"PickCube-v0" : 0, "PickSingleYCB-v0" : 0, "StackCube-v0" : 0}
    success_num = {"PickCube-v0" : 0.0, "PickSingleYCB-v0" : 0.0, "StackCube-v0" : 0.0, "PickClutterYCB-v0": 0.0,
                   "AssemblingKits-v0" : 0.0, "PickSingleEGAD-v0": 0.0}
    # success_num = {'StackCube', 'PickSingleYCB', 'PickClutterYCB', 'AssemblingKits', 'PegInsertionSide', 'PickSingleEGAD', 'PickCube'}
    # total_num = {"PickCube-v0" : 0, "PickSingleYCB-v0" : 0, "StackCube-v0" : 0}
    success_list = []
    i = 0
    print('begin 2....')
    print(cfg)
    model = PytorchDiffInference(model=model, sequence_length = cfg.dataset.traj_length, 
                                 num_pred_action=cfg.num_pred_action, stride=stride, use_action_head_diff=cfg.use_action_head_diff)
    print('begin init model....')
    print('A:', i, args.rank, len(eval_traj_index), len(eval_traj_list))
    print('--------------------')
    print(eval_traj_index)
    traj_str = eval_traj_list[eval_traj_index[i]]
    env_id, select_camera, seed, reset_kwargs, instruction = analyze_traj_str_v2(client, traj_str, json_repo, data_root_path)
    data_tmp_ = pickle.loads(client.get(os.path.join(data_root_path, traj_str)))

    def get_changed_pose(eval_dataset, data_path_, end_chaged_pose=False):
        # data_path_ = 'PickClutterYCB-v0/PickClutterYCB-v0_traj_3472_camera_11/data.pkl'
        data_url = os.path.join(eval_dataset.data_path, data_path_)
        data_pkl = pickle.loads(eval_dataset.client.get(data_url))
        
        # for i in range(100):
        trajs = eval_dataset.construct_traj(data_pkl, data_path_)
        #     print(i, trajs["action"]['gripper_change_pose'][0])
        if end_chaged_pose:
            target_position_pose = torch.tensor(data_pkl['step'][-1]['prev_ee_pose']).clone()

            target_position_pose[0] -= 0.615
            
            # data_pkl['step'][0]['prev_ee_pose']
            camera_extrinsic_cv = torch.tensor(data_pkl['step'][0]["camera_extrinsic_cv"])
            gripper_change_pose = get_pose_cam(camera_extrinsic_cv if cfg.dataset.use_baseframe_action == False else torch.eye(4), target_position_pose)
            
            t_gripper_position = torch.tensor(data_pkl['step'][-1]["action"][-1], dtype=torch.float32,).unsqueeze(-1)
            t_terminate_episode = torch.tensor([1, 0, 0], dtype=torch.int32)
            gripper_change_pose = torch.cat([gripper_change_pose, t_gripper_position, t_terminate_episode], dim=-1)
            
            return gripper_change_pose
        # print(trajs['action']['world_vector'], trajs["action"]['gripper_closedness_action'], trajs["action"]['gripper_change_pose'][0])
        return trajs["action"]['gripper_change_pose'][0]
        pass

    obs_pose = None
    if 'obs_poses' in cfg and cfg.obs_poses in [111]:
        # zeros inference for large gaussian noise
        obs_pose = get_changed_pose(eval_dataset, traj_str)[None, None,]
        obs_pose = torch.randn_like(obs_pose)

    if 'obs_poses' in cfg and cfg.obs_poses in [2, 3, 5, 8]:
        obs_pose = get_changed_pose(eval_dataset, traj_str)[None, None,]
        if cfg.obs_poses == 3:
            obs_pose = torch.zeros_like(obs_pose)
        elif cfg.obs_poses == 5:
            obs_pose[:, :, 7:] = torch.zeros_like(obs_pose[:, :, 7:])
        elif cfg.obs_poses == 8:
            obs_pose[:, :, 3:] = torch.zeros_like(obs_pose[:, :, 3:])
    # env_id = 'PickCube-v0'; select_camera = 1; seed = 982; reset_kwargs = {}
    fix_camera_pool = {}
    fix_camera_pool["camera_1"] = look_at([0.3, 0.2, 0.6], [-0.1, 0, 0.1])
    fix_camera_pool["camera_2" ] = look_at([0.3, -0.2, 0.6], [-0.1, 0, 0.1])
    fix_camera_pool["camera_3"] = look_at([0.3, 0.2, 0.4], [-0.1, 0, 0.3])
    fix_camera_pool["camera_4"] = look_at([0.5, -0.2, 0.8], [-0.1, 0, 0.1])
    fix_camera_pool["camera_5"] = look_at([0.5, 0.3, 0.6], [-0.2, 0, 0.1])  
    
    
    if 'fix_camera' in cfg and cfg.fix_camera:
        for ii in range(0, 5):
            if (extrinsics_pool[ii].astype(np.float32) == data_tmp_['step'][0]['camera_extrinsic_cv']).all():
                select_camera = ii + 1
                break
        # select_camera = traj_str.split('camera_')[1][:1]
        camera_pose = fix_camera_pool["camera_{}".format(select_camera)]
    else:
        camera_pose = look_at(camera_pool[select_camera][:3], camera_pool[select_camera][3:6], camera_pool[select_camera][6:9])
    # import pdb;pdb.set_trace()
    # env: BaseEnv = gym.make(
    #     env_id,
    #     renderer_kwargs={"offscreen_only": True, "device": f"cuda:{args.local_rank}"},
    #     obs_mode=obs_mode,
    #     reward_mode=reward_mode,
    #     control_mode=control_mode,
    #     render_mode=render_mode,
    #     render_camera_cfgs=dict(width=2 * CAMERA_W, height=2 * CAMERA_H),
    #     camera_cfgs=dict(
    #         base_camera=dict(p=camera_pose.p, q=camera_pose.q, width=CAMERA_W, height=CAMERA_H),
    #         hand_camera=dict(width=128, height=128),
    #     ),
    #     max_episode_steps=MAX_EPISODE_STEPS * 100,
    # )
    from diffusion_policy_3d.env_runner.metaworld_runner import MetaworldRunner
    metaworldrunner = MetaworldRunner(
        eval_episodes = 20,
        n_obs_steps = 2,
        n_action_steps = 1,
        fps = 10,
        # n_envs: null
        # n_train: null
        # n_test: null
        task_name = 'assembly',
        device = 'cuda',
        use_point_crop = False
        )

    from diffusion_policy_3d.policy.base_policy import BasePolicy
    class Custom(BasePolicy):

        def __init__(self, infer_model):
            super().__init__()
            self.model = infer_model
            pass

        def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            """
            obs_dict:
                str: B,To,*
            return: B,Ta,Da
            """
            self.model.set_natural_instruction(goal)
            self.model.set_observation(rgb=obs['rgb_obs']['rgb_static'])
            model_output = self.model.inference(torch.eye(4), abs_pose= 0, set_pose=True, 
                                    trajectory_dim=self.cfg.trajectory_dim, 
                                    pad_diff_nums=0, obs_pose=None, ret_7=True)
            
            model_output[...,-1] = np.where(model_output[...,-1] > 0, np.ones_like(model_output[...,-1])*(-1), np.ones_like(model_output[...,-1]))
            return model_output[0]            

        # reset state for stateful policies
        def reset(self):
            pass

        # ========== training ===========
        # no standard training interface except setting normalizer
        def set_normalizer(self, normalizer):
            pass
            # raise NotImplementedError()        

    runner_log = metaworldrunner.run(policy)
    import metaworld
    import metaworld
    import random

    ml10 = metaworld.MT10() # Construct the benchmark, sampling tasks

    testing_envs = []
    for name, env_cls in ml10.test_classes.items():
        env = env_cls(render_mode='rgb_array')
        task = random.choice([task for task in ml10.test_tasks
                                if task.env_name == name])
        env.set_task(task)
        testing_envs.append(env)





    print('begin gym....')

    # if record_dir != None:
    #     record_dir = record_dir.format(env_id=env_id)
    #     env = RecordEpisode(env, record_dir, render_mode=render_mode)
    # if render_goal_point and hasattr(env, "goal_site"):
    #     env.goal_site.unhide_visual()

    current_pose_queue = []
    env_i = 0
    env = testing_envs[env_i]
    obs = env.reset()

    if 'obs_poses' in cfg and cfg.obs_poses in [1, 4, 7, 11]:
        cur_ee_pose = eef_pose(env, extrinsic_cv=obs["camera_param"]["base_camera"]["extrinsic_cv"], camera_coord=camera_coord)
        cur_eef_pose_7d = np.concatenate([cur_ee_pose.p, cur_ee_pose.q], axis=-1)
        cur_eef_pose_7d = torch.tensor(cur_eef_pose_7d)
        cur_eef_pose_7d[0] -= 0.615
        if 'abs_pose' in cfg and cfg.abs_pose == 2:
            tmp_m = quaternion_to_matrix(torch.tensor(cur_ee_pose.q))
            euler_rot = matrix_to_euler_angles(tmp_m, convention='XYZ').cpu().numpy()
            cur_eef_pose_7d = np.concatenate([cur_ee_pose.p, euler_rot], axis=-1)
            cur_eef_pose_7d = torch.tensor(cur_eef_pose_7d)
            pass
        # import ipdb;ipdb.set_trace()
        #  if cfg.dataset.use_baseframe_action == False else torch.eye(4)
        t_ = torch.tensor([0, 1, 0], dtype=torch.int32)
        cur_abs_pose = torch.cat([cur_eef_pose_7d, torch.tensor([1], dtype=torch.float32), t_],dim=-1)[None, None,]
        current_pose_queue.append(cur_abs_pose)
        current_pose_queue.append(cur_abs_pose)
        if cfg.obs_poses == 4:
            obs_pose = cur_abs_pose
        elif cfg.obs_poses in [1, 7]:
            obs_pose = get_changed_pose(eval_dataset, traj_str)[None, None,]
            obs_pose[:, :, 7:] = torch.zeros_like(obs_pose[:, :, 7:])
            obs_pose = torch.cat([cur_abs_pose, obs_pose], dim=1)
        elif cfg.obs_poses == 11:
            obs_pose = torch.cat([current_pose_queue[-2], current_pose_queue[-1]], dim=1)


    print('begin reset....')
    model.set_natural_instruction(instruction)
    # total_num = 0
    # start_time = time.time()

    if root_folder != None:
        os.makedirs(root_folder, exist_ok = True) 

    print('begin tqdm....')
    import tqdm
    pbar = tqdm.tqdm(total=test_episodes_num)
    print('begin....')
    # model.set_eef_pose(eef_pose(env, obs["camera_param"]["base_camera"]["extrinsic_cv"] , camera_coord=camera_coord))
    model.set_observation(rgb=obs["image"]["base_camera"]["rgb"], wrist=obs["image"]["hand_camera"]["rgb"])
    # o o o o o o
    #   a a a a
    #   o o o o o o
    #     a a a a

    # o o o
    #   a
    # o o o o
    #     a
    # o o o o o
    #       a

    # o o n n n n
    #   a a a a
    #         o o
    #           a a a a

    moving_action = False
    moving_model_output = None

    pad_diff_nums = 0
    reg_prediction_nums = 1 if cfg.abs_sup in [3, 4] else 0
    if cfg.abs_sup in [3, 4] and ('no_abs_diff' not in cfg or not cfg.no_abs_diff):
        pad_diff_nums = 1
    if cfg.abs_sup and 'obs_poses' in cfg and cfg.obs_poses and 'no_abs_diff' in cfg and cfg.no_abs_diff:
        reg_prediction_nums = 1

    is_initial = True # the first step 
    step_i = 0
    while i < test_episodes_num and i < len(eval_traj_index):
        # total_num += 1
        # if total_num >500:
        #     break
        is_episode_end = False
        model_output = model.inference(obs["camera_param"]["base_camera"]["extrinsic_cv"] if cfg.dataset.use_baseframe_action == False else torch.eye(4), abs_pose= cfg.abs_pose if 'abs_pose' in cfg else 0, set_pose=True, 
                                       trajectory_dim=cfg.trajectory_dim, reg_prediction_nums=reg_prediction_nums, pad_diff_nums=pad_diff_nums, obs_pose=obs_pose, cfg = cfg.cfg if 'cfg' in cfg else 0)

        if moving_model_output is None:
            moving_model_output = model_output.copy()
        else:
            lambda_1 = (np.arange(len(moving_model_output), 0, -1)/ len(moving_model_output) * 0.5)
            for pi in range(len(moving_model_output)):
                moving_model_output[pi][:7] = moving_model_output[pi][:7] * lambda_1[pi] + model_output[pi][:7] * (1 - lambda_1[pi])


        for iiii in range(len(model_output[:cfg.n_action_steps])):
            
            model_output_new = model_output[iiii]
            # output[..., -2] = (output[...,-2] > 0.0).float() * 2 - 1
            action = np.concatenate([model_output_new[:3], (model_output_new[-2:-1] +  1 / 2)], axis=0)
            step_i += 1
            obs, reward, terminated, truncated, info = env.step(action)

            if 'obs_poses' in cfg and cfg.obs_poses in [2, 8] and step_i > 4:
                cur_ee_pose = eef_pose(env, extrinsic_cv=obs["camera_param"]["base_camera"]["extrinsic_cv"], camera_coord=camera_coord)

                should_change_obs_pose = np.linalg.norm(cur_ee_pose.p - obs_pose[0, 0, :3].cpu().numpy(), ord=1) < 1e-1 
                    # \
                        # and np.linalg.norm(cur_ee_pose.q - obs_pose[3:7].cpu().numpy(), ord=1) < 1e-3 
                # import ipdb;ipdb.set_trace()
                # print(step_i, model_output_new[:8], should_change_obs_pose, cur_ee_pose.p, obs_pose[0, 0, :3], camera_coord, np.linalg.norm(cur_ee_pose.p - obs_pose[0,0,:3].cpu().numpy(), ord=1))
                # import ipdb;ipdb.set_trace()
                if model_output_new[7] == -1: # TODO it is better to change the fixed value.
                    # if should_change_obs_pose:
                    obs_pose = get_changed_pose(eval_dataset, traj_str, True)[None, None,]
                    # print('change obs pose', step_i, obs_pose)
                else:
                    # 1
                    obs_pose = get_changed_pose(eval_dataset, traj_str)[None, None,]
                if cfg.obs_poses == 8:
                    obs_pose[:, :, 3:] = torch.zeros_like(obs_pose[:, :, 3:])
                    # print('change obs pose', step_i, obs_pose)
            if 'obs_poses' in cfg and cfg.obs_poses in [1, 4, 7, 11]:
                cur_ee_pose = eef_pose(env, extrinsic_cv=obs["camera_param"]["base_camera"]["extrinsic_cv"], camera_coord=camera_coord)
                cur_eef_pose_7d = np.concatenate([cur_ee_pose.p, cur_ee_pose.q], axis=-1)
                cur_eef_pose_7d = torch.tensor(cur_eef_pose_7d)
                cur_eef_pose_7d[0] -= 0.615
                if 'abs_pose' in cfg and cfg.abs_pose == 2:
                    tmp_m = quaternion_to_matrix(torch.tensor(cur_ee_pose.q))
                    euler_rot = matrix_to_euler_angles(tmp_m, convention='XYZ').cpu().numpy()
                    cur_eef_pose_7d = np.concatenate([cur_ee_pose.p, euler_rot], axis=-1)
                    cur_eef_pose_7d = torch.tensor(cur_eef_pose_7d)
                    pass                
                t_ = torch.tensor([0, 1, 0], dtype=torch.int32)
                cur_abs_pose = torch.cat([cur_eef_pose_7d, torch.tensor([model_output_new[7]], dtype=torch.float32), t_],dim=-1)[None, None,]
                current_pose_queue.append(cur_abs_pose)
                if cfg.obs_poses == 4:
                    obs_pose = cur_abs_pose
                elif cfg.obs_poses == 7:
                    obs_pose = get_changed_pose(eval_dataset, traj_str)[None, None,]
                    # obs_pose[:, :, 7:] = torch.zeros_like(obs_pose[:, :, 7:])
                    obs_pose = torch.cat([cur_abs_pose, obs_pose], dim=1)
                elif cfg.obs_poses == 11:
                    obs_pose = torch.cat([current_pose_queue[-2], current_pose_queue[-1]], dim=1)
                elif cfg.obs_poses == 1:
                    if model_output_new[7] == -1: # TODO it is better to change the fixed value.
                        # if should_change_obs_pose:
                        obs_pose = get_changed_pose(eval_dataset, traj_str, True)[None, None,]
                        # print('change obs pose', step_i, obs_pose)
                    else:
                        # 1
                        obs_pose = get_changed_pose(eval_dataset, traj_str)[None, None,]
                    obs_pose[:, :, 7:] = torch.zeros_like(obs_pose[:, :, 7:])
                    obs_pose = torch.cat([cur_abs_pose, obs_pose], dim=1)
            truncated = model.frame >= (MAX_EPISODE_STEPS * 4 / cfg.n_action_steps)
            # TODO if the model gets stuck, we should try to reset it. e.g., the occlusion.
            # beside, we should expand the action steps when the model suffer from this issue.

            if terminated or truncated:
                success_list.append(info["success"])
                if env_id not in success_num:
                    success_num[env_id] = 0
                success_num[env_id ] += info["success"]
                # total_num[env_id] += 1
                print(i, traj_str, info["success"], flush = True)
                if root_folder != None:
                    model.save_video(os.path.join(root_folder, f'{(i+args.rank * test_episodes_num):04d}_{instruction}_{env_id}_{select_camera}_{info["success"]}.mp4'))
            
                i += 1
                pbar.update(1)
                if i >= test_episodes_num or i >= len(eval_traj_index):
                    is_episode_end = True
                    break

                traj_str = eval_traj_list[eval_traj_index[i]]
                obs_pose = None
                if 'obs_poses' in cfg and cfg.obs_poses in [2, 5, 8]:
                    obs_pose = get_changed_pose(eval_dataset, traj_str)[None, None,]  
                    if cfg.obs_poses == 5:
                        obs_pose[:, :, 7:] = torch.zeros_like(obs_pose[:, :, 7:])
                    elif cfg.obs_poses == 8:
                        obs_pose[:, :, 3:] = torch.zeros_like(obs_pose[:, :, 3:])            
                is_initial = True
                step_i = 0
                
                model.set_natural_instruction(instruction)
                env_i += 1
                env = testing_envs[env_i]
                obs = env.reset()
                env_id = env_id_new
                select_camera = select_camera_new
                seed = seed_new
                reset_kwargs = reset_kwargs_new
                # if render_goal_point and hasattr(env, "goal_site"):
                #     env.goal_site.unhide_visual()
                # obs, _ = env.reset(seed=seed, options=reset_kwargs)
                model.reset_observation()

                if 'obs_poses' in cfg and cfg.obs_poses in [1, 4, 7, 11]:
                    cur_ee_pose = eef_pose(env, extrinsic_cv=obs["camera_param"]["base_camera"]["extrinsic_cv"], camera_coord=camera_coord)
                    cur_eef_pose_7d = np.concatenate([cur_ee_pose.p, cur_ee_pose.q], axis=-1)
                    cur_eef_pose_7d = torch.tensor(cur_eef_pose_7d)
                    cur_eef_pose_7d[0] -= 0.615
                    if 'abs_pose' in cfg and cfg.abs_pose == 2:
                        tmp_m = quaternion_to_matrix(torch.tensor(cur_ee_pose.q))
                        euler_rot = matrix_to_euler_angles(tmp_m, convention='XYZ').cpu().numpy()
                        cur_eef_pose_7d = np.concatenate([cur_ee_pose.p, euler_rot], axis=-1)
                        cur_eef_pose_7d = torch.tensor(cur_eef_pose_7d)
                        pass                    
                    t_ = torch.tensor([0, 1, 0], dtype=torch.int32)
                    cur_abs_pose = torch.cat([cur_eef_pose_7d, torch.tensor([1], dtype=torch.float32), t_],dim=-1)[None, None,]
                    current_pose_queue = []
                    current_pose_queue.append(cur_abs_pose)
                    current_pose_queue.append(cur_abs_pose)
                    if cfg.obs_poses == 4:
                        obs_pose = cur_abs_pose
                    elif cfg.obs_poses in [1, 7]:
                        obs_pose = get_changed_pose(eval_dataset, traj_str)[None, None,]
                        obs_pose[:, :, 7:] = torch.zeros_like(obs_pose[:, :, 7:])
                        obs_pose = torch.cat([cur_abs_pose, obs_pose], dim=1)
                    elif cfg.obs_poses == 11:
                        obs_pose = torch.cat([current_pose_queue[-2], current_pose_queue[-1]], dim=1)                
                moving_model_output = None
                model.set_eef_pose(eef_pose(env, obs["camera_param"]["base_camera"]["extrinsic_cv"], camera_coord=camera_coord))
                model.set_observation(rgb=obs["image"]["base_camera"]["rgb"], wrist=obs["image"]["hand_camera"]["rgb"])

                break
            model.set_eef_pose(eef_pose(env, obs["camera_param"]["base_camera"]["extrinsic_cv"], camera_coord=camera_coord))
            model.set_observation(rgb=obs["image"]["base_camera"]["rgb"], wrist=obs["image"]["hand_camera"]["rgb"])

        if is_episode_end:
            break
        # model.set_eef_pose(eef_pose(env, obs["camera_param"]["base_camera"]["extrinsic_cv"], camera_coord=camera_coord))
        # model.set_observation(rgb=obs["image"]["base_camera"]["rgb"], wrist=obs["image"]["hand_camera"]["rgb"])
    
    total_success_rate = np.mean(success_list)
    env.close()
    del env
    # end_time = time.time()
    # print(f"time: {end_time - start_time}, i: {i}, total_num: {total_num}")
    return success_num, None, total_success_rate


def close_loop_eval_calvin(
    obs_mode="rgbd",
    reward_mode=None,
    control_mode=TARGET_CONTROL_MODE,
    render_mode="cameras",
    record_dir=None,
    render_goal_point=True,
    test_episodes_num=100,
    model=None,
    eval_data_list=None,
    args=None,
    rand_seed=0,
    json_repo="/xxx/xxx/share_data/Anonymous/maniskill2/demos/v0/rigid_body/",
    camera_coord=True,
    stride=1,
    root_folder = None,
    data_root_path = None,
    cfg=None,
    eval_dataset=None,
):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    if 'input_size' in cfg:
        global CAMERA_W, CAMERA_H
        import ast
        # img_size = ast.literal_eval(cfg.input_size)
        CAMERA_W = int(cfg.input_size)
        CAMERA_H = int(cfg.input_size)

    assert cfg is not None
    print('begin ....', root_folder)
    # import pdb;pdb.set_trace()
    client = Client()
    np.set_printoptions(suppress=True, precision=3)
    print(eval_data_list)
    eval_traj_list = pickle.load(open(eval_data_list, "rb"))
    if len(eval_traj_list) > 500:
        eval_traj_list = [item for item in eval_traj_list if not item.__contains__('PegInsertionSide-v0')]
        eval_traj_list = eval_traj_list[:-88]
    # eval_traj_list = [item for item in eval_traj_list if not item.__contains__('AssemblingKits-v0')]

    # data_url = os.path.join(self.data_path, self.data_cam_list[index])
    np.random.seed(0 % 9973)
    eval_traj_index = np.random.permutation(len(eval_traj_list))[: args.world_size * test_episodes_num]
    eval_traj_index = eval_traj_index[args.rank * test_episodes_num : (args.rank + 1) * test_episodes_num]
    eval_traj_index = sorted(eval_traj_index)
    # import ipdb;ipdb.set_trace()

    success_num = {"PickCube-v0" : 0, "PickSingleYCB-v0" : 0, "StackCube-v0" : 0}
    success_num = {"PickCube-v0" : 0.0, "PickSingleYCB-v0" : 0.0, "StackCube-v0" : 0.0, "PickClutterYCB-v0": 0.0,
                   "AssemblingKits-v0" : 0.0, "PickSingleEGAD-v0": 0.0}
    # success_num = {'StackCube', 'PickSingleYCB', 'PickClutterYCB', 'AssemblingKits', 'PegInsertionSide', 'PickSingleEGAD', 'PickCube'}
    # total_num = {"PickCube-v0" : 0, "PickSingleYCB-v0" : 0, "StackCube-v0" : 0}
    success_list = []
    i = 0
    print('begin 2....')
    print(cfg)
    model = PytorchDiffInference(model=model, sequence_length = cfg.dataset.traj_length, 
                                 num_pred_action=cfg.num_pred_action, stride=stride, use_action_head_diff=cfg.use_action_head_diff)
    print('begin init model....')
    print('A:', i, args.rank, len(eval_traj_index), len(eval_traj_list))
    print('--------------------')
    print(eval_traj_index)
    traj_str = eval_traj_list[eval_traj_index[i]]
    env_id, select_camera, seed, reset_kwargs, instruction = analyze_traj_str_v2(client, traj_str, json_repo, data_root_path)
    data_tmp_ = pickle.loads(client.get(os.path.join(data_root_path, traj_str)))

    def get_changed_pose(eval_dataset, data_path_, end_chaged_pose=False):
        # data_path_ = 'PickClutterYCB-v0/PickClutterYCB-v0_traj_3472_camera_11/data.pkl'
        data_url = os.path.join(eval_dataset.data_path, data_path_)
        data_pkl = pickle.loads(eval_dataset.client.get(data_url))
        
        # for i in range(100):
        trajs = eval_dataset.construct_traj(data_pkl, data_path_)
        #     print(i, trajs["action"]['gripper_change_pose'][0])
        if end_chaged_pose:
            target_position_pose = torch.tensor(data_pkl['step'][-1]['prev_ee_pose']).clone()

            target_position_pose[0] -= 0.615
            
            # data_pkl['step'][0]['prev_ee_pose']
            camera_extrinsic_cv = torch.tensor(data_pkl['step'][0]["camera_extrinsic_cv"])
            gripper_change_pose = get_pose_cam(camera_extrinsic_cv if cfg.dataset.use_baseframe_action == False else torch.eye(4), target_position_pose)
            
            t_gripper_position = torch.tensor(data_pkl['step'][-1]["action"][-1], dtype=torch.float32,).unsqueeze(-1)
            t_terminate_episode = torch.tensor([1, 0, 0], dtype=torch.int32)
            gripper_change_pose = torch.cat([gripper_change_pose, t_gripper_position, t_terminate_episode], dim=-1)
            
            return gripper_change_pose
        # print(trajs['action']['world_vector'], trajs["action"]['gripper_closedness_action'], trajs["action"]['gripper_change_pose'][0])
        return trajs["action"]['gripper_change_pose'][0]
        pass

    obs_pose = None
    if 'obs_poses' in cfg and cfg.obs_poses in [111]:
        # zeros inference for large gaussian noise
        obs_pose = get_changed_pose(eval_dataset, traj_str)[None, None,]
        obs_pose = torch.randn_like(obs_pose)

    if 'obs_poses' in cfg and cfg.obs_poses in [2, 3, 5, 8]:
        obs_pose = get_changed_pose(eval_dataset, traj_str)[None, None,]
        if cfg.obs_poses == 3:
            obs_pose = torch.zeros_like(obs_pose)
        elif cfg.obs_poses == 5:
            obs_pose[:, :, 7:] = torch.zeros_like(obs_pose[:, :, 7:])
        elif cfg.obs_poses == 8:
            obs_pose[:, :, 3:] = torch.zeros_like(obs_pose[:, :, 3:])
    # env_id = 'PickCube-v0'; select_camera = 1; seed = 982; reset_kwargs = {}
    fix_camera_pool = {}
    fix_camera_pool["camera_1"] = look_at([0.3, 0.2, 0.6], [-0.1, 0, 0.1])
    fix_camera_pool["camera_2" ] = look_at([0.3, -0.2, 0.6], [-0.1, 0, 0.1])
    fix_camera_pool["camera_3"] = look_at([0.3, 0.2, 0.4], [-0.1, 0, 0.3])
    fix_camera_pool["camera_4"] = look_at([0.5, -0.2, 0.8], [-0.1, 0, 0.1])
    fix_camera_pool["camera_5"] = look_at([0.5, 0.3, 0.6], [-0.2, 0, 0.1])  
    camera_pose = fix_camera_pool["camera_1"]
    
    import random

    print('begin gym....')

    # if record_dir != None:
    #     record_dir = record_dir.format(env_id=env_id)
    #     env = RecordEpisode(env, record_dir, render_mode=render_mode)
    # if render_goal_point and hasattr(env, "goal_site"):
    #     env.goal_site.unhide_visual()

    current_pose_queue = []
    env_i = 0
    env = testing_envs[env_i]
    obs = env.reset()

    if 'obs_poses' in cfg and cfg.obs_poses in [1, 4, 7, 11]:
        cur_ee_pose = eef_pose(env, extrinsic_cv=obs["camera_param"]["base_camera"]["extrinsic_cv"], camera_coord=camera_coord)
        cur_eef_pose_7d = np.concatenate([cur_ee_pose.p, cur_ee_pose.q], axis=-1)
        cur_eef_pose_7d = torch.tensor(cur_eef_pose_7d)
        cur_eef_pose_7d[0] -= 0.615
        if 'abs_pose' in cfg and cfg.abs_pose == 2:
            tmp_m = quaternion_to_matrix(torch.tensor(cur_ee_pose.q))
            euler_rot = matrix_to_euler_angles(tmp_m, convention='XYZ').cpu().numpy()
            cur_eef_pose_7d = np.concatenate([cur_ee_pose.p, euler_rot], axis=-1)
            cur_eef_pose_7d = torch.tensor(cur_eef_pose_7d)
            pass
        # import ipdb;ipdb.set_trace()
        #  if cfg.dataset.use_baseframe_action == False else torch.eye(4)
        t_ = torch.tensor([0, 1, 0], dtype=torch.int32)
        cur_abs_pose = torch.cat([cur_eef_pose_7d, torch.tensor([1], dtype=torch.float32), t_],dim=-1)[None, None,]
        current_pose_queue.append(cur_abs_pose)
        current_pose_queue.append(cur_abs_pose)
        if cfg.obs_poses == 4:
            obs_pose = cur_abs_pose
        elif cfg.obs_poses in [1, 7]:
            obs_pose = get_changed_pose(eval_dataset, traj_str)[None, None,]
            obs_pose[:, :, 7:] = torch.zeros_like(obs_pose[:, :, 7:])
            obs_pose = torch.cat([cur_abs_pose, obs_pose], dim=1)
        elif cfg.obs_poses == 11:
            obs_pose = torch.cat([current_pose_queue[-2], current_pose_queue[-1]], dim=1)


    print('begin reset....')
    model.set_natural_instruction(instruction)
    # total_num = 0
    # start_time = time.time()

    if root_folder != None:
        os.makedirs(root_folder, exist_ok = True) 

    print('begin tqdm....')
    import tqdm
    pbar = tqdm.tqdm(total=test_episodes_num)
    print('begin....')
    # model.set_eef_pose(eef_pose(env, obs["camera_param"]["base_camera"]["extrinsic_cv"] , camera_coord=camera_coord))
    model.set_observation(rgb=obs["image"]["base_camera"]["rgb"], wrist=obs["image"]["hand_camera"]["rgb"])
    # o o o o o o
    #   a a a a
    #   o o o o o o
    #     a a a a

    # o o o
    #   a
    # o o o o
    #     a
    # o o o o o
    #       a

    # o o n n n n
    #   a a a a
    #         o o
    #           a a a a

    moving_action = False
    moving_model_output = None

    pad_diff_nums = 0
    reg_prediction_nums = 1 if cfg.abs_sup in [3, 4] else 0
    if cfg.abs_sup in [3, 4] and ('no_abs_diff' not in cfg or not cfg.no_abs_diff):
        pad_diff_nums = 1
    if cfg.abs_sup and 'obs_poses' in cfg and cfg.obs_poses and 'no_abs_diff' in cfg and cfg.no_abs_diff:
        reg_prediction_nums = 1

    is_initial = True # the first step 
    step_i = 0
    while i < test_episodes_num and i < len(eval_traj_index):
        # total_num += 1
        # if total_num >500:
        #     break
        is_episode_end = False
        model_output = model.inference(obs["camera_param"]["base_camera"]["extrinsic_cv"] if cfg.dataset.use_baseframe_action == False else torch.eye(4), abs_pose= cfg.abs_pose if 'abs_pose' in cfg else 0, set_pose=True, 
                                       trajectory_dim=cfg.trajectory_dim, reg_prediction_nums=reg_prediction_nums, pad_diff_nums=pad_diff_nums, obs_pose=obs_pose, cfg = cfg.cfg if 'cfg' in cfg else 0)

        if moving_model_output is None:
            moving_model_output = model_output.copy()
        else:
            lambda_1 = (np.arange(len(moving_model_output), 0, -1)/ len(moving_model_output) * 0.5)
            for pi in range(len(moving_model_output)):
                moving_model_output[pi][:7] = moving_model_output[pi][:7] * lambda_1[pi] + model_output[pi][:7] * (1 - lambda_1[pi])


        for iiii in range(len(model_output[:cfg.n_action_steps])):
            
            model_output_new = model_output[iiii]
            # output[..., -2] = (output[...,-2] > 0.0).float() * 2 - 1
            action = np.concatenate([model_output_new[:3], (model_output_new[-2:-1] +  1 / 2)], axis=0)
            step_i += 1
            obs, reward, terminated, truncated, info = env.step(action)

            if 'obs_poses' in cfg and cfg.obs_poses in [2, 8] and step_i > 4:
                cur_ee_pose = eef_pose(env, extrinsic_cv=obs["camera_param"]["base_camera"]["extrinsic_cv"], camera_coord=camera_coord)

                should_change_obs_pose = np.linalg.norm(cur_ee_pose.p - obs_pose[0, 0, :3].cpu().numpy(), ord=1) < 1e-1 
                    # \
                        # and np.linalg.norm(cur_ee_pose.q - obs_pose[3:7].cpu().numpy(), ord=1) < 1e-3 
                # import ipdb;ipdb.set_trace()
                # print(step_i, model_output_new[:8], should_change_obs_pose, cur_ee_pose.p, obs_pose[0, 0, :3], camera_coord, np.linalg.norm(cur_ee_pose.p - obs_pose[0,0,:3].cpu().numpy(), ord=1))
                # import ipdb;ipdb.set_trace()
                if model_output_new[7] == -1: # TODO it is better to change the fixed value.
                    # if should_change_obs_pose:
                    obs_pose = get_changed_pose(eval_dataset, traj_str, True)[None, None,]
                    # print('change obs pose', step_i, obs_pose)
                else:
                    # 1
                    obs_pose = get_changed_pose(eval_dataset, traj_str)[None, None,]
                if cfg.obs_poses == 8:
                    obs_pose[:, :, 3:] = torch.zeros_like(obs_pose[:, :, 3:])
                    # print('change obs pose', step_i, obs_pose)
            if 'obs_poses' in cfg and cfg.obs_poses in [1, 4, 7, 11]:
                cur_ee_pose = eef_pose(env, extrinsic_cv=obs["camera_param"]["base_camera"]["extrinsic_cv"], camera_coord=camera_coord)
                cur_eef_pose_7d = np.concatenate([cur_ee_pose.p, cur_ee_pose.q], axis=-1)
                cur_eef_pose_7d = torch.tensor(cur_eef_pose_7d)
                cur_eef_pose_7d[0] -= 0.615
                if 'abs_pose' in cfg and cfg.abs_pose == 2:
                    tmp_m = quaternion_to_matrix(torch.tensor(cur_ee_pose.q))
                    euler_rot = matrix_to_euler_angles(tmp_m, convention='XYZ').cpu().numpy()
                    cur_eef_pose_7d = np.concatenate([cur_ee_pose.p, euler_rot], axis=-1)
                    cur_eef_pose_7d = torch.tensor(cur_eef_pose_7d)
                    pass                
                t_ = torch.tensor([0, 1, 0], dtype=torch.int32)
                cur_abs_pose = torch.cat([cur_eef_pose_7d, torch.tensor([model_output_new[7]], dtype=torch.float32), t_],dim=-1)[None, None,]
                current_pose_queue.append(cur_abs_pose)
                if cfg.obs_poses == 4:
                    obs_pose = cur_abs_pose
                elif cfg.obs_poses == 7:
                    obs_pose = get_changed_pose(eval_dataset, traj_str)[None, None,]
                    # obs_pose[:, :, 7:] = torch.zeros_like(obs_pose[:, :, 7:])
                    obs_pose = torch.cat([cur_abs_pose, obs_pose], dim=1)
                elif cfg.obs_poses == 11:
                    obs_pose = torch.cat([current_pose_queue[-2], current_pose_queue[-1]], dim=1)
                elif cfg.obs_poses == 1:
                    if model_output_new[7] == -1: # TODO it is better to change the fixed value.
                        # if should_change_obs_pose:
                        obs_pose = get_changed_pose(eval_dataset, traj_str, True)[None, None,]
                        # print('change obs pose', step_i, obs_pose)
                    else:
                        # 1
                        obs_pose = get_changed_pose(eval_dataset, traj_str)[None, None,]
                    obs_pose[:, :, 7:] = torch.zeros_like(obs_pose[:, :, 7:])
                    obs_pose = torch.cat([cur_abs_pose, obs_pose], dim=1)
            truncated = model.frame >= (MAX_EPISODE_STEPS * 4 / cfg.n_action_steps)
            # TODO if the model gets stuck, we should try to reset it. e.g., the occlusion.
            # beside, we should expand the action steps when the model suffer from this issue.

            if terminated or truncated:
                success_list.append(info["success"])
                if env_id not in success_num:
                    success_num[env_id] = 0
                success_num[env_id ] += info["success"]
                # total_num[env_id] += 1
                print(i, traj_str, info["success"], flush = True)
                if root_folder != None:
                    model.save_video(os.path.join(root_folder, f'{(i+args.rank * test_episodes_num):04d}_{instruction}_{env_id}_{select_camera}_{info["success"]}.mp4'))
            
                i += 1
                pbar.update(1)
                if i >= test_episodes_num or i >= len(eval_traj_index):
                    is_episode_end = True
                    break

                traj_str = eval_traj_list[eval_traj_index[i]]
                obs_pose = None
                if 'obs_poses' in cfg and cfg.obs_poses in [2, 5, 8]:
                    obs_pose = get_changed_pose(eval_dataset, traj_str)[None, None,]  
                    if cfg.obs_poses == 5:
                        obs_pose[:, :, 7:] = torch.zeros_like(obs_pose[:, :, 7:])
                    elif cfg.obs_poses == 8:
                        obs_pose[:, :, 3:] = torch.zeros_like(obs_pose[:, :, 3:])            
                is_initial = True
                step_i = 0
                
                model.set_natural_instruction(instruction)
                env_i += 1
                env = testing_envs[env_i]
                obs = env.reset()
                env_id = env_id_new
                select_camera = select_camera_new
                seed = seed_new
                reset_kwargs = reset_kwargs_new
                # if render_goal_point and hasattr(env, "goal_site"):
                #     env.goal_site.unhide_visual()
                # obs, _ = env.reset(seed=seed, options=reset_kwargs)
                model.reset_observation()

                if 'obs_poses' in cfg and cfg.obs_poses in [1, 4, 7, 11]:
                    cur_ee_pose = eef_pose(env, extrinsic_cv=obs["camera_param"]["base_camera"]["extrinsic_cv"], camera_coord=camera_coord)
                    cur_eef_pose_7d = np.concatenate([cur_ee_pose.p, cur_ee_pose.q], axis=-1)
                    cur_eef_pose_7d = torch.tensor(cur_eef_pose_7d)
                    cur_eef_pose_7d[0] -= 0.615
                    if 'abs_pose' in cfg and cfg.abs_pose == 2:
                        tmp_m = quaternion_to_matrix(torch.tensor(cur_ee_pose.q))
                        euler_rot = matrix_to_euler_angles(tmp_m, convention='XYZ').cpu().numpy()
                        cur_eef_pose_7d = np.concatenate([cur_ee_pose.p, euler_rot], axis=-1)
                        cur_eef_pose_7d = torch.tensor(cur_eef_pose_7d)
                        pass                    
                    t_ = torch.tensor([0, 1, 0], dtype=torch.int32)
                    cur_abs_pose = torch.cat([cur_eef_pose_7d, torch.tensor([1], dtype=torch.float32), t_],dim=-1)[None, None,]
                    current_pose_queue = []
                    current_pose_queue.append(cur_abs_pose)
                    current_pose_queue.append(cur_abs_pose)
                    if cfg.obs_poses == 4:
                        obs_pose = cur_abs_pose
                    elif cfg.obs_poses in [1, 7]:
                        obs_pose = get_changed_pose(eval_dataset, traj_str)[None, None,]
                        obs_pose[:, :, 7:] = torch.zeros_like(obs_pose[:, :, 7:])
                        obs_pose = torch.cat([cur_abs_pose, obs_pose], dim=1)
                    elif cfg.obs_poses == 11:
                        obs_pose = torch.cat([current_pose_queue[-2], current_pose_queue[-1]], dim=1)                
                moving_model_output = None
                model.set_eef_pose(eef_pose(env, obs["camera_param"]["base_camera"]["extrinsic_cv"], camera_coord=camera_coord))
                model.set_observation(rgb=obs["image"]["base_camera"]["rgb"], wrist=obs["image"]["hand_camera"]["rgb"])

                break
            model.set_eef_pose(eef_pose(env, obs["camera_param"]["base_camera"]["extrinsic_cv"], camera_coord=camera_coord))
            model.set_observation(rgb=obs["image"]["base_camera"]["rgb"], wrist=obs["image"]["hand_camera"]["rgb"])

        if is_episode_end:
            break
        # model.set_eef_pose(eef_pose(env, obs["camera_param"]["base_camera"]["extrinsic_cv"], camera_coord=camera_coord))
        # model.set_observation(rgb=obs["image"]["base_camera"]["rgb"], wrist=obs["image"]["hand_camera"]["rgb"])
    
    total_success_rate = np.mean(success_list)
    env.close()
    del env
    # end_time = time.time()
    # print(f"time: {end_time - start_time}, i: {i}, total_num: {total_num}")
    return success_num, None, total_success_rate


def analyze_traj_str_v2(client, traj_str, json_repo, data_root_path):

    env_id = traj_str.split("/")[0]
    # select_camera = int(traj_str.split("_")[-1].split(".")[0])
    data = pickle.loads(client.get(os.path.join(data_root_path, traj_str)))
    select_camera = data["camera_index_in_pool"]
    json_root_path = os.path.join(json_repo, env_id)

    if env_id == "PickSingleYCB-v0":
        pkl_str = traj_str.split('/')[1]
        task_name_start_pos = len(env_id) + 1
        task_name_end_pos = pkl_str.find('_traj')
        task_name = pkl_str[task_name_start_pos:task_name_end_pos]
        json_data = load_json(os.path.join(json_root_path, task_name + '.json'))
    else:
        json_data = load_json(os.path.join(json_root_path, "trajectory.json"))
    
    traj_start_pos = traj_str.find("traj")
    index = int(traj_str[traj_start_pos:].split("_")[1])
    reset_kwargs = json_data["episodes"][index]["reset_kwargs"]
    if "seed" in reset_kwargs:
        reset_kwargs["seed"] = json_data["episodes"][index]["episode_seed"]
    seed = reset_kwargs.pop("seed")
   
    instruction = data["step"][0]["observation"]["natural_instruction"]
    return env_id, select_camera, seed, reset_kwargs, instruction
