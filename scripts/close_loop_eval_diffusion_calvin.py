import argparse
import logging
import os
from pathlib import Path

from Dataset_VLA.utils import euler2rotm, rotm2euler
from scripts.evaluation.evaluate_policy import evaluate_policy
from utils.data_utils import get_pose_cam
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
from moviepy.editor import ImageSequenceClip
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPModel, CLIPProcessor
from transforms3d.quaternions import mat2quat, quat2mat
from moviepy.editor import ImageSequenceClip




class PytorchDiffInference(nn.Module):
    def __init__(self, model, prediction_type='epsilon',sequence_length = 15, 
                 use_wrist_img=False, device="cuda", 
                 stride=1, num_pred_action=4,
                 use_action_head_diff=0):
        super().__init__()

        self.device = torch.device(device)

        # use_wrist_img = use_wrist_img
        use_depth_img = False

        self.use_wrist_img = use_wrist_img
        self.use_depth_img = use_depth_img

        self.sequence_length = sequence_length
        self.num_pred_action = num_pred_action
        self.use_action_head_diff = use_action_head_diff

        self.stride = stride

        self.model = model

        print('sequence_length:', self.sequence_length)
        try:
            if hasattr(self.model, 'module'):
                self.use_wrist_img = self.model.module.use_wrist_img
                self.use_depth_img = self.model.module.use_depth_img
            else:
                self.use_wrist_img = self.model.use_wrist_img
                self.use_depth_img = self.model.use_depth_img
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
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((224,224), antialias=True),
                torchvision.transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ]
        )

        self.clip_tokenizer = AutoTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14", use_fast=False
        )
        self.clip_text_encoder = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        ).text_model

        self.to(self.device)
        self.frame = 0
        
        # self.base_episode = pkl.load(open("/xxx/xxx/Anonymous/project/embodied_foundation/PickCube-v0_traj_0_camera_0.pkl", "rb"))
        # self.episode = pkl.load(open("/xxx/xxx/Anonymous/project/embodied_foundation/PickCube-v0_traj_0_camera_1.pkl", "rb"))
        self.eef_pose = None
        self.empty_instruction = None
        # model_output: dx dy dz dqw dqx dqy dqz terminate

    def set_natural_instruction(self, instruction: str):
        inputs = self.clip_tokenizer(text=instruction, return_tensors="pt", max_length=77, padding="max_length")
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        with torch.no_grad():
            text_embeddings = self.clip_text_encoder(**inputs)[0].squeeze(0)
        self.instruction = text_embeddings

    def set_eef_pose(self, eef_pose):
        self.eef_pose = eef_pose

    def set_observation(self, rgb, depth=None, wrist=None):
        assert (rgb >= 0).all() and (rgb <= 255).all()
        self.observation.append(rgb)
        if self.model_input == []:

            # rgb = torch.tensor(rgb).to(self.device, non_blocking=True)
            rgb_data = self.data_transform(rgb).to(self.device, non_blocking=True)
            self.model_input = rgb_data.unsqueeze(0)
            if len(self.model_input) < self.sequence_length:
                self.model_input = self.model_input.repeat(self.sequence_length, 1, 1, 1)
        else:

            # rgb = torch.tensor(rgb).to(self.device, non_blocking=True)
            rgb_data = self.data_transform(rgb).to(self.device, non_blocking=True)
            self.model_input = torch.cat((self.model_input, rgb_data.unsqueeze(0)), dim=0)
            self.model_input = self.model_input[-self.sequence_length :]

        if wrist is not None and self.use_wrist_img:
            if self.model_input_wrist == []:

                # wrist_data = torch.tensor(wrist).to(self.device, non_blocking=True)

                wrist_data = self.data_transform(wrist).to(self.device, non_blocking=True)

                self.model_input_wrist = wrist_data.unsqueeze(0)
                if len(self.model_input_wrist) < self.sequence_length:
                    self.model_input_wrist = self.model_input_wrist.repeat(self.sequence_length, 1, 1, 1)
            else:

                # wrist_data = torch.tensor(wrist).to(self.device, non_blocking=True)

                # wrist_data = self.data_transform((wrist_data / 255.0).permute(2, 0, 1).contiguous())
                wrist_data = self.data_transform(wrist).to(self.device, non_blocking=True)

                self.model_input_wrist = torch.cat((self.model_input_wrist, wrist_data.unsqueeze(0)), dim=0)
                self.model_input_wrist = self.model_input_wrist[-self.sequence_length :]
            # wrist = (
            #     nn.functional.interpolate(torch.tensor(wrist).permute(2, 0, 1).unsqueeze(0), size=(224, 224), mode="nearest")
            #     .squeeze()
            #     .permute(1, 2, 0)
            #     .cpu()
            #     .numpy()
            # )
            # self.observation[-1] = np.concatenate([self.observation[-1], wrist], axis=1)
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
        obs = {"image": self.model_input[-self.sequence_length :].unsqueeze(0)}
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
                    self.empty_instruction = self.clip_text_encoder(**inputs)[0].squeeze(0)
                self.empty_instruction = self.empty_instruction[None, None, ...].repeat(1, obs["image"].shape[1], 1, 1)
            obs['natural_language_embedding'] = torch.cat([self.empty_instruction, obs['natural_language_embedding'], ], dim=0)
            obs["image"] = torch.cat([obs["image"], obs["image"]], dim=0)

        if obs_pose is not None:
            obs['poses'] = obs_pose.to(obs["natural_language_embedding"].device)  # B 1 11
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                # 1 x T x L x C
                if self.use_action_head_diff == 2:
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
        # Image.fromarray((unnormalize(self.model_input[-1].permute(1, 2, 0)).clamp(0, 1) * 255).byte().cpu().numpy()).save(f"obs_{self.frame}.png")

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

from calvin_agent.models.calvin_base_model import CalvinBaseModel
class CustomModel(CalvinBaseModel):
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        # raise NotImplementedError

    def reset(self):
        """
        This is called
        """
        self.model.reset_observation()
        pass
        # raise NotImplementedError

    def step(self, obs, goal):
        """
        Args:
            obs: environment observations
            goal: embedded language goal
        Returns:
            action: predicted action
        """
        self.model.set_natural_instruction(goal)
        self.model.set_observation(rgb=obs['rgb_obs']['rgb_static'])
        model_output = self.model.inference(torch.eye(4), abs_pose= 0, set_pose=True, 
                                trajectory_dim=self.cfg.trajectory_dim, 
                                pad_diff_nums=0, obs_pose=None, ret_7=True)
        
        model_output[...,-1] = np.where(model_output[...,-1] > 0, np.ones_like(model_output[...,-1])*(-1), np.ones_like(model_output[...,-1]))
        print(model_output.shape)
        return model_output

class CustomModel1(CalvinBaseModel):
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        # raise NotImplementedError

    def reset(self):
        """
        This is called
        """
        pass
        # raise NotImplementedError

    def step(self, obs, goal):
        """
        Args:
            obs: environment observations
            goal: embedded language goal
        Returns:
            action: predicted action
        """

        # print(model_output.shape)
        # goal_rgb = goal[0]

        # rgb = obs['rgb_obs']['rgb_static'] # (200, 200, 3)
        # hand_rgb = obs['rgb_obs']['rgb_gripper']

        # goal_rgb = Image.fromarray(goal_rgb)
        # goal_rgb = T.ToTensor()(goal_rgb.convert("RGB"))
        # goal_rgb = self.preprocess(goal_rgb) # (3, 224, 224)

        # rgb = Image.fromarray(rgb)
        # rgb = T.ToTensor()(rgb.convert("RGB"))
        # rgb = self.preprocess(rgb) # (3, 224, 224)
        # self.rgb_list.append(rgb)

        # hand_rgb = Image.fromarray(hand_rgb)
        # hand_rgb = T.ToTensor()(hand_rgb.convert("RGB"))
        # hand_rgb = self.preprocess(hand_rgb)
        # self.hand_rgb_list.append(hand_rgb)

        state = obs['robot_obs'] # (15,)
        # xyz_state = state[:3]
        # rpy_state = state[3:6]
        # rotm_state = euler2rotm(rpy_state)
        # gripper_state = state[-1]
        # state = (xyz_state, rotm_state, gripper_state)
        # self.state_list.append(state)

        # buffer_len = len(self.rgb_list)
        # if buffer_len > self.seq_len:
        #     self.rgb_list.pop(0)
        #     self.hand_rgb_list.pop(0)
        #     self.state_list.pop(0)
        #     assert len(self.rgb_list) == self.seq_len
        #     assert len(self.hand_rgb_list) == self.seq_len
        #     assert len(self.state_list) == self.seq_len
        #     buffer_len = len(self.rgb_list)
        


        # # Static RGB
        # c, h, w = rgb.shape
        # c2,h2,w2=goal_rgb.shape
        # assert c==c2 and h==h2 and w==w2
        # rgb_data = torch.zeros((1, self.seq_len, c, h, w))
        # rgb_tensor = torch.stack(self.rgb_list, dim=0) # (len, c, h, w)
        # rgb_data[0, :buffer_len] = rgb_tensor
        # goal_rgb_data=torch.zeros((1, c, h, w))
        # goal_rgb_data[0]=goal_rgb

        # # Hand RGB
        # c, h, w = hand_rgb.shape
        # hand_rgb_data = torch.zeros((1, self.seq_len, c, h, w))
        # hand_rgb_tensor = torch.stack(self.hand_rgb_list, dim=0) # (len, c, h, w)
        # hand_rgb_data[0, :buffer_len] = hand_rgb_tensor

        # # State
        # arm_state, gripper_state = CustomModel.compute_rel_state(self.state_list)
        # arm_state_data = torch.zeros((1, self.seq_len, 6))
        # arm_state_tensor = torch.from_numpy(arm_state)
        # arm_state_data[0, :buffer_len] = arm_state_tensor
        # gripper_state_tensor = torch.from_numpy(gripper_state)
        # gripper_state_tensor = (gripper_state_tensor + 1.0) / 2.0
        # gripper_state_tensor = gripper_state_tensor.long()
        # gripper_state_data = torch.zeros((1, self.seq_len)).long()
        # gripper_state_data[0, :buffer_len] = gripper_state_tensor
        # gripper_state_data = F.one_hot(gripper_state_data, num_classes=2).type_as(arm_state_data)

        # # Attention mask
        # attention_mask = torch.zeros(1, self.seq_len).long()
        # attention_mask[0, :buffer_len] = 1

        # Action placeholder
        # arm_action_data = torch.zeros((1, self.seq_len, self.configs["policy"]['act_len'], 6))
        # gripper_action_data = torch.zeros(1, self.seq_len, self.configs["policy"]['act_len'])

        #progress_placeholder
        # progress_data=torch.zeros(1, self.seq_len)

        # input_dict = dict()
        # input_dict['rgb'] = rgb_data.to(self.device)
        # input_dict['hand_rgb'] = hand_rgb_data.to(self.device)
        # input_dict["goal_rgb"]=goal_rgb_data.to(self.device)
        # input_dict['arm_state'] = arm_state_data.to(self.device)
        # input_dict['gripper_state'] = gripper_state_data.to(self.device)
        # input_dict['arm_action'] = arm_action_data.to(self.device)
        # input_dict['gripper_action'] = gripper_action_data.to(self.device)
        # input_dict['attention_mask'] = attention_mask.to(self.device)
        # input_dict["text"]=[text]
        # input_dict["progress"]=progress_data
        # Forward pass
        with torch.no_grad():
            # action,action_traj = self.policy.evaluate(input_dict)
            import time
            
            self.model.set_natural_instruction(goal)
            # import ipdb;ipdb.set_trace()
            #print(self.cfg.model.use_wrist_img)
            # hand_rgb = frame['rgb_gripper']
            self.model.set_observation(rgb=obs['rgb_obs']['rgb_static'], wrist=obs['rgb_obs']['rgb_gripper'])
            
            model_output = self.model.inference(torch.eye(4), abs_pose= 0, set_pose=True, 
                                    trajectory_dim=self.cfg.trajectory_dim, 
                                    pad_diff_nums=0, obs_pose=None, ret_7=True)

            model_output[...,-1] = np.where(model_output[...,-1] > 0.5, np.ones_like(model_output[...,-1]), np.ones_like(model_output[...,-1])*(-1))
            action = model_output[0]
            # import ipdb;ipdb.set_trace()

        
        # Action mode: ee_rel_pose_local
        state = obs['robot_obs'] # (15,)
        xyz_state = state[:3]
        rpy_state = state[3:6]
        rotm_state = euler2rotm(rpy_state)
        rel_action = action
        xyz_action = rel_action[:3] / 50 # scale down by 50  
        rpy_action = rel_action[3:6] / 20 # scale down by 20
        gripper_action = rel_action[6]
        rotm_action = euler2rotm(rpy_action)
        xyz_next_state = xyz_state + rotm_state @ xyz_action
        rotm_next_state = rotm_state @ rotm_action
        rpy_next_state = rotm2euler(rotm_next_state)
        action = np.zeros(7)
        action[:3] = (xyz_next_state - xyz_state) * 50  
        action[3:6] = (rpy_next_state - rpy_state) * 20
        action[-1] = gripper_action
        action = torch.from_numpy(action)[None,...].cpu().detach().numpy()
        # self.rollout_step_counter += 1
    
        return action

def close_loop_eval_calvin(
    obs_mode="rgbd",
    reward_mode=None,
    control_mode=None,
    render_mode="cameras",
    record_dir=None,
    render_goal_point=True,
    test_episodes_num=100,
    model=None,
    eval_data_list=None,
    args=None,
    rank=0,
    rand_seed=0,
    json_repo="/xxx/xxx/share_data/Anonymous/maniskill2/demos/v0/rigid_body/",
    camera_coord=True,
    stride=1,
    root_folder = None,
    data_root_path = None,
    cfg=None,
    eval_dataset=None,
):

    assert cfg is not None
    print('begin ....', root_folder)
    # import pdb;pdb.set_trace()

    np.set_printoptions(suppress=True, precision=3)


    np.random.seed(0 % 9973)

    i = 0
    print('begin 2....')
    print(cfg)
    model = PytorchDiffInference(model=model, sequence_length = cfg.dataset.traj_length, use_wrist_img=cfg.model.use_wrist_img,
                                 num_pred_action=cfg.num_pred_action, stride=stride, use_action_head_diff=cfg.use_action_head_diff)
    print('begin init model....')
    print('--------------------')


    import random

    print('begin gym....')

    # if record_dir != None:
    #     record_dir = record_dir.format(env_id=env_id)
    #     env = RecordEpisode(env, record_dir, render_mode=render_mode)
    # if render_goal_point and hasattr(env, "goal_site"):
    #     env.goal_site.unhide_visual()

    from calvin_env.envs.play_table_env import get_env
    def make_env(dataset_path):
        val_folder = Path(dataset_path) / "validation"
        env = get_env(val_folder, show_gui=False)

        # insert your own env wrapper
        # env = Wrapper(env)
        return env
    model = CustomModel1(model,cfg)
    
    env = make_env('calvin_task/')
    evaluate_policy(model, env, epoch=0., eval_log_dir=os.path.join(root_folder, 'tmp_'+str(args.rank)), debug=0, rank=args.rank, each_length=test_episodes_num)

    return 0, None, 0

