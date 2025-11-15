import os
import pickle as pkl

import numpy as np
import sapien.core as sapien
import scipy.ndimage
import torch
import torch.nn as nn
import torchvision

from mani_skill2.utils.sapien_utils import vectorize_pose

# import cv2
from moviepy.editor import ImageSequenceClip
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from transformers import AutoTokenizer, CLIPModel
from transforms3d.quaternions import mat2quat, quat2mat

from Dataset_Sim.SimDataset_discrete import get_action_spec
from Dataset_Sim.SimDataset_discrete import process_traj_v3



def unnormalize(x):
    x = x.clone()
    for i in range(3):
        x[..., i] = x[..., i] * IMAGENET_DEFAULT_STD[i] + IMAGENET_DEFAULT_MEAN[i]

    return x


class PytorchInference(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()

        self.device = torch.device(device)
        action_spec = get_action_spec(self.device)

        vocab_size = 2048
        sequence_length = 15
        num_layers = 12
        use_qformer = True
        use_wrist_img = False
        use_depth_img = False

        self.use_wrist_img = use_wrist_img
        self.use_depth_img = use_depth_img

        self.sequence_length = sequence_length
        self.stride = 1

        
        self.model_input = []
        self.observation = []
        self.model_input_wrist = []
        self.model_input_depth = []
        self.instruction = ""
        self.data_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ]
        )

        self.clip_tokenizer = AutoTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14/", use_fast=False
        )
        self.clip_text_encoder = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14/"
        ).text_model

        self.to(self.device)
        self.frame = 0

        
        self.base_episode = pkl.load(
            open("your data with action in robot base coordinate pkl path", "rb")
        )
        self.episode = pkl.load(open("your data with action in camera base coordinate pkl path", "rb"))
        self.eef_pose = None
        self.eef_pose_base = None

       

    def set_natural_instruction(self, instruction: str):
        inputs = self.clip_tokenizer(text=instruction, return_tensors="pt", max_length=77, padding="max_length")
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        with torch.no_grad():
            text_embeddings = self.clip_text_encoder(**inputs)[0].squeeze(0)
        self.instruction = text_embeddings

    def set_eef_pose_base(self, eef_pose_base):
        self.eef_pose_base = eef_pose_base

    def set_eef_pose(self, eef_pose):
        self.eef_pose = eef_pose

    def set_observation(self, rgb, depth=None, wrist=None):
        assert (rgb >= 0).all() and (rgb <= 255).all()
        self.observation.append(rgb)
        self.model_input.append(self.data_transform(rgb).to(self.device, non_blocking=True))
        if wrist is not None and self.use_wrist_img:
            # self.wrist_observation.append(wrist)
            self.model_input_wrist.append(self.data_transform(wrist).to(self.device, non_blocking=True))
            wrist = (
                nn.functional.interpolate(torch.tensor(wrist).permute(2, 0, 1).unsqueeze(0), size=(224, 224), mode="nearest")
                .squeeze()
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            self.observation[-1] = np.concatenate([self.observation[-1], wrist], axis=1)
        if depth is not None and self.use_depth_img:
            self.model_input_depth.append(torch.tensor(depth / 10).to(self.device, non_blocking=True))
            depth = torch.tensor(depth / 10 * 255).repeat(1, 1, 3).byte().cpu().numpy()
            self.observation[-1] = np.concatenate([self.observation[-1], depth], axis=1)

    def save_video(self, fpath):
        

        clip = ImageSequenceClip(self.observation, fps=10 / self.stride)
        clip.write_videofile(fpath, codec="libx264", audio=False, logger=None)  # Use 'libx264' for the H.264 codec

    def reset_observation(self):
        self.model_input = []
        self.observation = []
        self.model_input_wrist = []
        self.model_input_depth = []
        self.frame = 0

    def calc_act(self, base_episode, camera_extrinsic_cv, current_frame_idx):
        try:
            pose1 = torch.tensor(vectorize_pose(self.eef_pose_base)).clone()
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

    def inference(self, extrinsics=None):
       

        gt_output = self.calc_act(self.base_episode, torch.tensor(extrinsics.astype("float32")), self.frame)
       
        target_pose = self.get_target_pose(gt_output["world_vector"].cpu().numpy(), gt_output["rotation_delta"].cpu().numpy())

        gt_output = torch.cat(
            [
                # gt_output["world_vector"],
                # gt_output["rotation_delta"],
                torch.tensor(target_pose.p),
                torch.tensor(target_pose.q),
                gt_output["gripper_closedness_action"],
                gt_output["terminate_episode"][[0]],
            ]
        )

        # print(list(output.cpu().numpy() - gt_output.cpu().numpy()))
        print(f"eval frame {self.frame}", flush=True)
        Image.fromarray((unnormalize(self.model_input[-1].permute(1, 2, 0)).clamp(0, 1) * 255).byte().cpu().numpy()).save(f"obs_{self.frame}.png")

        self.frame += self.stride

        return gt_output.cpu().numpy()
        return output.cpu().numpy()
