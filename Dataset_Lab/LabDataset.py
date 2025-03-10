import copy
import gzip
import json
import logging
import os
import pickle
import random
import sys
import time
from collections import defaultdict
import matplotlib.pyplot as plt
sys.path.append(os.path.join('/xxx/xxx/Anonymous/Code/embodied_foundation', "scripts/openx_utils/"))
sys.path.append(os.path.join('/xxx/xxx/Anonymous/Code/embodied_foundation', "../embodied_foundation/scripts"))
import numpy as np
from portalocker import TemporaryFileLock
import pytorch3d.transforms as Pose3d
import scipy.spatial.transform as st

import torch
import torchvision.transforms
# import a s3 package as Client
from pytorch3d.transforms import (
    Transform3d,
    euler_angles_to_matrix,
    matrix_to_euler_angles,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_to_matrix, quaternion_to_axis_angle
)
from timm.data.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
)
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset, IterableDataset
import scipy.spatial.transform as st
from PIL import Image
@torch.no_grad()
def get_pose_cam(world2cam, pose1):

    rot_mat1 = quaternion_to_matrix(pose1[3:])
    pose1_mat = torch.eye(4)

    pose1_mat[:3, :3] = rot_mat1
    pose1_mat[:3, 3] = pose1[:3]


    pose1_transform = Transform3d(matrix=pose1_mat.T)

    world2cam_transform = Transform3d(matrix=world2cam.T)
    pose1_cam = pose1_transform.compose(world2cam_transform)
    vector = pose1_cam.get_matrix()[0, -1, :3]
    rotation = matrix_to_quaternion(pose1_cam.get_matrix()[0, :3, :3].T)
    return torch.cat([vector, rotation])

def quaternion_to_euler_radians(w, x, y, z):
    roll = np.arctan2(2 * (w * x + y * z), w**2 + z**2 - (x**2 + y**2))

    sinpitch = 2 * (w * y - z * x)
    pitch = np.arcsin(sinpitch)

    yaw = np.arctan2(2 * (w * z + x * y), w**2 + x**2 - (y**2 + z**2))

    return torch.tensor([roll, pitch, yaw], dtype=torch.float32)

def unnormalize(x):
    x = x.clone()
    for i in range(3):
        x[..., i] = x[..., i] * IMAGENET_DEFAULT_STD[i] + IMAGENET_DEFAULT_MEAN[i]

    return x

@torch.no_grad()
def process_traj(world2cam, pose1, pose2):

    rot_mat1 = quaternion_to_matrix(pose1[3:])
    rot_mat2 = quaternion_to_matrix(pose2[3:])
    pose1_mat, pose2_mat = torch.eye(4), torch.eye(4)

    pose1_mat[:3, :3] = rot_mat1
    pose2_mat[:3, :3] = rot_mat2
    pose1_mat[:3, 3] = pose1[:3]
    pose2_mat[:3, 3] = pose2[:3]

    pose1_transform = Transform3d(matrix=pose1_mat.T)
    pose2_transform = Transform3d(matrix=pose2_mat.T)
    world2cam_transform = Transform3d(matrix=world2cam.T)
    pose1_cam = pose1_transform.compose(world2cam_transform)
    pose2_cam = pose2_transform.compose(world2cam_transform)

    pose1_to_pose2 = pose1_cam.inverse().compose(pose2_cam)

    # translation_delta = pose1_to_pose2.get_matrix()[0, -1, :3]
    translation_delta = (
        pose2_cam.get_matrix()[0, -1, :3] - pose1_cam.get_matrix()[0, -1, :3]
    )

    rotation_delta = matrix_to_quaternion(pose1_to_pose2.get_matrix()[0, :3, :3].T)

    return translation_delta.to(torch.float32), rotation_delta.to(torch.float32)

data_hist_world_vector = np.zeros(11)
data_hist_rotation_delta = np.zeros(11)
def generate_histogram(world_vector, rotation_delta):
    for i in range(world_vector.shape[0]):
        bar_num = int(abs(world_vector[i]) / 0.01)
        bar_num = min(bar_num, 10)
        data_hist_world_vector[bar_num] += 1

    for i in range(rotation_delta.shape[0]):
        bar_num = int(abs(rotation_delta[i]) / 0.01)
        bar_num = min(bar_num, 10)
        data_hist_rotation_delta[bar_num] += 1

def select_gripper(episode):

    episode_step = []
    for i in range(len(episode["steps"])):

        if episode["steps"][i]["observation"]["gripper_position"] != 0:
            episode_step.append(episode["steps"][i])
    
    return episode_step

@torch.no_grad()
def euler_to_quaternion(eulers):
    quaternion = st.Rotation.from_euler('xyz', eulers).as_quat()
    return torch.tensor([quaternion[-1], quaternion[0], quaternion[1], quaternion[2]])



class LabDataset(Dataset):

    def __init__(
        self,
        data_path = "LabData",
        language_embedding_path = "LabData_language_embeddings_77token_v0",
        traj_per_episode = 8,
        traj_length = 15,
        dataset_type = 0,
        use_baseframe_action = False,
        split_type = None,
        stride = 1,
        img_preprocess_type=0,
        data_cam_list = None,
        train_with_droid = False,
        obs_n_frames = 2, # observation frames
        include_target = 0,
        euler_delta=0,
        selected_list = ['left', 'corner', 'right']
    ):
        self.data_path = data_path
        self.traj_per_episode = traj_per_episode
        self.traj_length = traj_length
        self.use_baseframe_action = use_baseframe_action
        self.split_type = split_type
        self.language_embedding_path = language_embedding_path
        self.dataset_type = dataset_type
        self.obs_n_frames = obs_n_frames
        # self.augment_traj = augment_traj                                                                                                                                       
        self.include_target = include_target
        self.stride = stride
        self.img_preprocess_type = img_preprocess_type
        
        self.data_cam_list  = pickle.load(open(data_cam_list, 'rb'))
        
        if data_cam_list.__contains__('fix_cam') and len(self.data_cam_list) < 500:
            aa = []
            for i in range(50):
                aa += self.data_cam_list
            self.data_cam_list = aa
            pass
        self.euler_delta = euler_delta

        from torchvision.transforms import v2
        self.data_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                # v2.RandomResizedCrop(size=(224, 224), antialias=True),
                
                torchvision.transforms.CenterCrop(size=(480, 480)),
                torchvision.transforms.Resize((224,224), antialias=True)
                # torchvision.transforms.RandomHorizontalFlip(p=0.5),
                
                # torchvision.transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                # torchvision.transforms.Pad(padding = (80,1,80,0)),
                # torchvision.transforms.Resize((224,224), antialias=True)
            ]
        )
        self.data_transform1 = torchvision.transforms.Compose(
            [
                # torchvision.transforms.ColorJitter(brightness=0.2, contrast=[0.8, 1.2], saturation=[0.8, 1.2], hue=0.05),
                torchvision.transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                # torchvision.transforms.Pad(padding = (80,1,80,0)),
                # torchvision.transforms.Resize((224,224), antialias=True)
            ]
        )
        
        
        self.selected_list = selected_list 
        

    def __len__(self):

        if self.split_type != "overfit":
            return len(self.data_cam_list)
        else:
            return len(self.data_cam_list) * 1000


    @torch.no_grad()
    def construct_traj(self, episode, episode_path):

        stride = self.stride
        # episode["steps"] = select_gripper(episode)
        gripper_closeness = np.array([episode["steps"][_]["observation"]["gripper_position"] for _ in range(len(episode["steps"]))])
        gripper_change = np.where(gripper_closeness[1:] != gripper_closeness[:-1])[0]

        gripper_change = np.concatenate([gripper_change, gripper_change + 1])
        gripper_change.sort()

        episode_step = []

        start = random.randint(0, stride - 1)

        for i in range(len(gripper_change)):
            episode_step.extend(episode["steps"][start : gripper_change[i] : stride])
            start = gripper_change[i]

        episode_step.extend(episode["steps"][start:-1:stride])
        episode_step.append(episode["steps"][-1])
        episode["steps"] = episode_step

        steps = len(episode["steps"])
        start_frame = np.random.permutation(steps - self.traj_length + 1)[: self.traj_per_episode]
        if self.include_target:            
            start_frame = np.random.permutation(max(steps - 2*self.stride, 1))[: self.traj_per_episode]
            # print(start_frame, max(steps- 2*self.stride, 1))
        if self.include_target == 2: # only for the approaching stage.
            start_frame = np.random.permutation(np.arange(max(steps - 2*self.stride, 1), steps-1, 1))
        if len(start_frame) < self.traj_per_episode:
            start_frame = np.random.choice(start_frame, self.traj_per_episode, replace=True)

        selected_list = self.selected_list
        # selected_camera = selected_list[random.randint(0,len(selected_list))]
        selected_camera = selected_list[0]
        camera_extrinsic_cv = torch.tensor(np.linalg.inv(episode[f"ext_cam_extrinsics_{selected_camera}"]),dtype = torch.float32)

        trajs = {"observation": defaultdict(list), "action": defaultdict(list)}
        language_embedding = pickle.load(open(os.path.join(self.language_embedding_path, episode_path),'rb'))
        
        for i in range(self.traj_per_episode):

            frame_idx = start_frame[i]
            traj = {"observation": defaultdict(list), "action": defaultdict(list)}

            for j in range(self.traj_length):

                current_frame_idx = frame_idx + j
                observation = {}
                action = {}

                observation["natural_language_embedding"] = torch.tensor(language_embedding, dtype = torch.float32)
                observation["camera_extrinsic_cv"] = camera_extrinsic_cv
                
                if j < self.obs_n_frames:
                    if current_frame_idx < steps:
                        
                        if self.img_preprocess_type == 2:
                            # observation["image"] = torch.tensor(episode["steps"][current_frame_idx]["observation"][f"exterior_image_1_{selected_camera}"])
                            observation["image"] = episode["steps"][current_frame_idx]["observation"][f"exterior_image_1_{selected_camera}"]
                        else:
                            # observation["image"] = self.data_transform(
                            #     episode["steps"][current_frame_idx]["observation"][f"exterior_image_1_{selected_camera}"]
                            # )
                            observation["image"] = episode["steps"][current_frame_idx]["observation"][f"exterior_image_1_{selected_camera}"]
                            pass
                    else:
                        observation["image"] = np.zeros_like(traj["observation"]['image'][-1])

                if current_frame_idx == steps - 1:
                    action["terminate_episode"] = torch.tensor([1, 0, 0], dtype=torch.int32)
                    action["gripper_closedness_action"] = torch.tensor(
                        1. if episode["steps"][current_frame_idx]["observation"]["gripper_position"] > 0.2 else 0.0,
                        dtype=torch.float32,
                    ).unsqueeze(-1)
                elif current_frame_idx >  steps - 1:
                    action["terminate_episode"] = torch.tensor([1, 0, 0], dtype=torch.int32)
                    action["gripper_closedness_action"] = torch.tensor(
                        0.0,
                        dtype=torch.float32,
                    ).unsqueeze(-1)
                    
                else:
                    action["terminate_episode"] = torch.tensor([0, 1, 0], dtype=torch.int32)
                    action["gripper_closedness_action"] = torch.tensor(
                        1. if episode["steps"][current_frame_idx + 1]["observation"]["gripper_position"] > 0.2 else 0.0,
                        dtype=torch.float32,
                    ).unsqueeze(-1)


                action["loss_weight"] = torch.ones((9))

                if current_frame_idx < steps - 1:

                    pose1 = torch.tensor(episode["steps"][current_frame_idx]["observation"]['cartesian_position']).clone()
                    pose2 = torch.tensor(episode["steps"][current_frame_idx + 1]["observation"]['cartesian_position']).clone()


                    action["world_vector"], action["rotation_delta"] = process_traj(
                        (camera_extrinsic_cv if self.use_baseframe_action == False else torch.eye(4)),
                        pose1.to(torch.float32),
                        pose2.to(torch.float32),
                    )

                    # action['rotation_delta'] *= torch.sign(action['rotation_delta'][0])
                    if self.euler_delta:
                        action["rotation_delta"] = quaternion_to_euler_radians(action['rotation_delta'][0], action['rotation_delta'][1], action['rotation_delta'][2], action['rotation_delta'][3]) 
                        # matrix_to_euler_angles(quaternion_to_matrix(action["rotation_delta"] ), convention='XYZ')
                    else:
                        action["rotation_delta"][0] -= 1.0
                    
                    action['abs_tar_pose'] = get_pose_cam(camera_extrinsic_cv if self.use_baseframe_action == False else torch.eye(4), pose2)
                    if self.euler_delta:
                        tmp_abs_pose =  action['abs_tar_pose'][:6].clone()
                        tmp_abs_pose[3:] = quaternion_to_euler_radians( action['abs_tar_pose'][3],  action['abs_tar_pose'][4],  action['abs_tar_pose'][5],  action['abs_tar_pose'][6]) 
                        action['abs_tar_pose'] = tmp_abs_pose
                else:
                    action["loss_weight"] = torch.zeros((9))
                    action["world_vector"] = torch.zeros(3)
                    if self.euler_delta:
                        action['rotation_delta'] = torch.zeros(3)
                    else:
                        action['rotation_delta'] = torch.zeros(4)
                    tmp_pose = torch.tensor(episode['steps'][-1]["observation"]['cartesian_position']).clone()
                    action['abs_tar_pose'] =  get_pose_cam(camera_extrinsic_cv if self.use_baseframe_action == False else torch.eye(4), tmp_pose)
                    if self.euler_delta:
                        tmp_abs_pose =  action['abs_tar_pose'][:6].clone()
                        tmp_abs_pose[3:] = quaternion_to_euler_radians( action['abs_tar_pose'][3],  action['abs_tar_pose'][4],  action['abs_tar_pose'][5],  action['abs_tar_pose'][6]) 
                        action['abs_tar_pose'] = tmp_abs_pose
                # generate_histogram(action["world_vector"], action["rotation_delta"])
                if (
                        current_frame_idx > 0 and current_frame_idx < steps
                        and episode["steps"][current_frame_idx]["observation"]["gripper_position"] != episode["steps"][current_frame_idx - 1]["observation"]["gripper_position"]
                    ):
                        action["loss_weight"][7] = 100.0
                if (
                    current_frame_idx > 1 and current_frame_idx < steps
                    and episode["steps"][current_frame_idx]["observation"]["gripper_position"] != episode["steps"][current_frame_idx - 2]["observation"]["gripper_position"]
                ):
                    action["loss_weight"][7] = 100.0

                for k in observation.keys():
                    traj["observation"][k].append(observation[k])
                    if j == self.traj_length - 1 and k != 'image':
                        traj["observation"][k] = torch.stack(traj["observation"][k], dim=0)

                if j == self.obs_n_frames - 1 and 'image' in observation.keys():
                    traj["observation"]['image'] = np.stack(traj["observation"]['image'], axis=0)
                    
                    if self.img_preprocess_type == 2:
                        traj["observation"]['image'] = torch.tensor(traj["observation"]['image'])
                    else:
                        aaa = traj["observation"]['image']
                        tmp_img_inp = np.transpose(aaa, (1,2,0,3)).reshape(aaa.shape[1], aaa.shape[2], aaa.shape[0]*aaa.shape[3])
                        
                        

                        tmp_img_inp = self.data_transform(tmp_img_inp)
                        
                        
                        tmp_img_inp = tmp_img_inp.reshape(aaa.shape[0], aaa.shape[3], tmp_img_inp.shape[1], tmp_img_inp.shape[2])

                        if self.euler_delta in [2,3]:
                            # L C H W
                            t_shape = tmp_img_inp.shape
                            tmp_img_inp = self.data_transform1(tmp_img_inp.permute(1,2,0,3).flatten(2,3))
                            tmp_img_inp = tmp_img_inp.reshape(t_shape[1], t_shape[2], t_shape[0], t_shape[3]).permute(2, 0, 1, 3)
                            
                            if 'DEBUG' in os.environ: 
                                Image.fromarray((unnormalize(tmp_img_inp[0].permute(1,2,0)) * 255).cpu().numpy().astype(np.uint8)).save('temp11_{}_111.png'.format(episode_path.replace('.pkl', '')))
                                Image.fromarray((unnormalize(tmp_img_inp[1].permute(1,2,0)) * 255).cpu().numpy().astype(np.uint8)).save('temp11_{}_221.png'.format(episode_path.replace('.pkl', '')))
                        traj["observation"]['image'] = tmp_img_inp
                for k in action.keys():
                    traj["action"][k].append(action[k])
                    if j == self.traj_length - 1:
                        traj["action"][k] = torch.stack(traj["action"][k], dim=0)


            gripper_change_pose = torch.zeros(11).to(torch.float32) # indicate no target position or we dont know
                
            trajs["action"]['gripper_change_pose'].append(gripper_change_pose)
            if i == self.traj_per_episode - 1:
                trajs["action"]['gripper_change_pose'] = torch.stack(trajs["action"]['gripper_change_pose'], dim=0)
            for k in traj["observation"].keys():
                trajs["observation"][k].append(traj["observation"][k])
                if i == self.traj_per_episode - 1:
                    trajs["observation"][k] = torch.stack(trajs["observation"][k], dim=0)
            for k in traj["action"].keys():
                trajs["action"][k].append(traj["action"][k])
                if i == self.traj_per_episode - 1:
                    trajs["action"][k] = torch.stack(trajs["action"][k], dim=0)
        if trajs is not None:
            trajs['instruction'] = os.path.basename(episode_path).split('_')[1][:-4]
        return trajs

    @torch.no_grad()
    def __getitem__(self, index):

        index = index % (len(self.data_cam_list))
        
        while True:

            try:
                
                data_url = os.path.join(self.data_path, self.data_cam_list[index])
                data_pkl = pickle.load(open(data_url, 'rb'))
                trajs = self.construct_traj(data_pkl, self.data_cam_list[index])
                break

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(e)
                print(f"Fail to load data {self.data_cam_list[index]}", flush = True)
                index = random.randint(0, len(self.data_cam_list)-1)

        return trajs

if __name__ == "__main__":

    print("begin!", flush=True)
    dataset = LabDataset(
        data_path = "LabData",
        language_embedding_path= "LabData___embedding",
        traj_per_episode = 8,
        traj_length = 32,
        dataset_type = 0,
        use_baseframe_action = True,
        split_type = None,
        stride = 4,
        include_target=1,
        obs_n_frames=2,
        euler_delta = 3,
        data_cam_list='fix_cam_lab.pkl',
        selected_list=['left']
    )

    wv_min = torch.ones(3) * 1000
    wv_max = torch.ones(3) * -1000
    rt_min = torch.ones(4) * 1000
    rt_max = torch.ones(4) * -1000
    pose_min = torch.ones(6) * 1000
    pose_max = torch.ones(6) * -1000
    dataloader = DataLoader(dataset, batch_size = 4, shuffle = True)
    

    total_iter_num = 0
    action_list = []
    for ii in range(1):
        for i,batch in enumerate(dataloader):

            action_list.append(torch.cat([batch["action"]["world_vector"].flatten(0, 2), batch["action"]["rotation_delta"].flatten(0, 2)], dim=-1).cpu().numpy())
            wv_min = torch.minimum(
                    wv_min, batch["action"]["world_vector"].amin(dim=(0, 1, 2))
                )
            wv_max = torch.maximum(
                wv_max, batch["action"]["world_vector"].amax(dim=(0, 1, 2))
                )
  
            if total_iter_num % 50 == 0:
                print('low', [str(item) +',' for item in np.quantile(np.concatenate(action_list), 0.01, axis=0).tolist()], flush=True)
                print('high', [str(item) +',' for item in np.quantile(np.concatenate(action_list), 0.99, axis=0).tolist()], flush=True)
                print("wv_min: ", wv_min, flush = True)
                print("wv_max: ", wv_max, flush = True)

            total_iter_num += 1




    print("wv_min: ", wv_min)
    print("wv_max: ", wv_max)
    print("rt_min: ", rt_min)
    print("rt_max: ", rt_max)
    print("finish!", flush=True)
