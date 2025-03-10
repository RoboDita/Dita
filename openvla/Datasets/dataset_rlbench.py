from collections import defaultdict, Counter
import itertools
import math
import os
import random
from pathlib import Path
from time import time

import torch
from torch.utils.data import Dataset
# import a s3 package as Client



# from utils import loader, Resize, TrajectoryInterpolator
from typing import List, Tuple, Optional
import numpy as np
from timm.data.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
)
from pytorch3d.transforms import (
    Transform3d,
    euler_angles_to_matrix,
    matrix_to_euler_angles,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_to_matrix
)
import pickle

def new_path(path):
    return path.replace('/home/Anonymous/Anonymous/3d_diffuser_actor/data/peract/raw/', 's3://Anonymous/rlbench/peract/raw/')


@torch.no_grad()
def process_traj_v3(world2cam, pose1, pose2):

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
    translation_delta = pose2_cam.get_matrix()[0, -1, :3] - pose1_cam.get_matrix()[0, -1, :3]

    rotation_delta = matrix_to_quaternion(pose1_to_pose2.get_matrix()[0, :3, :3].T)

    return translation_delta.to(torch.float32), rotation_delta.to(torch.float32)

class RLBenchDataset(Dataset):
    """RLBench dataset."""

    def __init__(
        self,
        # required
        root,
        instructions=None,
        data_list_pkl = None,
        # dataset specification
        taskvar=[('close_door', 0)],
        max_episode_length=5,
        cache_size=0,
        max_episodes_per_task=100,
        num_iters=None,
        cameras=("wrist", "left_shoulder", "right_shoulder"),
        # for augmentations
        training=True,
        image_rescale=(1.0, 1.0),
        # for trajectories
        return_low_lvl_trajectory=False,
        dense_interpolation=False,
        interpolation_length=100,
        relative_action=False,
        sequence_length=32,
        n_obs= 2,
    ):
        self._cache = {}
        self._cache_size = cache_size
        self._cameras = cameras
        self._max_episode_length = max_episode_length
        self._num_iters = num_iters
        self._training = training
        self._taskvar = taskvar
        self._return_low_lvl_trajectory = return_low_lvl_trajectory
        self.n_obs= n_obs
        self.sequence_length= sequence_length
        if isinstance(root, (Path, str)):
            root = [Path(root)]
        self._root = [Path(r).expanduser() for r in root]
        self._relative_action = relative_action


        self.client = Client()

        
        
        # # For trajectory optimization, initialize interpolation tools
        # if return_low_lvl_trajectory:
        #     assert dense_interpolation
        #     self._interpolate_traj = TrajectoryInterpolator(
        #         use=dense_interpolation,
        #         interpolation_length=interpolation_length
        #     )
        

        # Keep variations and useful instructions
        self._instructions = defaultdict(dict)
        self._num_vars = Counter()  # variations of the same task
        for root, (task, var) in itertools.product(self._root, taskvar):
            data_dir = root / f"{task}+{var}"
            if data_dir.is_dir():
                if instructions is not None:
                    self._instructions[task][var] = instructions[task][var]
                self._num_vars[task] += 1

        # If training, initialize augmentation classes

        # # File-names of episodes per task and variation
        # episodes_by_task = defaultdict(list)  # {task: [(task, var, filepath)]}
        # for root, (task, var) in itertools.product(self._root, taskvar):
        #     data_dir = root / f"{task}+{var}"
        #     if not data_dir.is_dir():
        #         print(f"Can't find dataset folder {data_dir}")
        #         continue
        #     npy_episodes = [(task, var, ep) for ep in data_dir.glob("*.npy")]
        #     dat_episodes = [(task, var, ep) for ep in data_dir.glob("*.dat")]
        #     pkl_episodes = [(task, var, ep) for ep in data_dir.glob("*.pkl")]
        #     episodes = npy_episodes + dat_episodes + pkl_episodes
        #     # Split episodes equally into task variations
        #     if max_episodes_per_task > -1:
        #         episodes = episodes[
        #             :max_episodes_per_task // self._num_vars[task] + 1
        #         ]
        #     if len(episodes) == 0:
        #         print(f"Can't find episodes at folder {data_dir}")
        #         continue
        #     episodes_by_task[task] += episodes

        # Collect and trim all episodes in the dataset
        self._episodes = pickle.load(open(data_list_pkl, 'rb'))
        self._num_episodes = 0

        # for task, eps in episodes_by_task.items():
        #     if len(eps) > max_episodes_per_task and max_episodes_per_task > -1:
        #         eps = random.sample(eps, max_episodes_per_task)
        #     episodes_by_task[task] = sorted(
        #         eps, key=lambda t: int(str(t[2]).split('/')[-1][2:-4])
        #     )
        #     self._episodes += eps
        #     self._num_episodes += len(eps)
        self._num_episodes = len(self._episodes)
        print(f"Created dataset from {root} with {self._num_episodes}")
        # self._episodes_by_task = episodes_by_task
        import torchvision
        self.data_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize(OPENAI_CLIP_MEAN, OPENAI_CLIP_STD)
                torchvision.transforms.Resize((224,224), antialias=True),
                torchvision.transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ]
        )


    @staticmethod
    def _unnormalize_rgb(rgb):
        # (from [-1, 1] to [0, 1]) to feed RGB to pre-trained backbone
        return rgb / 2 + 0.5

    def sample_observations(self, root_path, observations, sequence_length: int = 32, task='') -> dict:  
        if not observations:  
            raise ValueError("Observations list is empty")  
        
        num_samples = 2  # 您需要的样本数量  
        num_obs = 2
        len_trajectory = len(observations)  
        max_start_index = len_trajectory - num_obs  # 最大的起始索引  
        
        if max_start_index < 0:  
            raise ValueError("Observation list is too short to sample sequences of length {}".format(sequence_length))  
        
        # 随机选择两个起始索引  
        start_indices = np.random.randint(0, max_start_index + 1, size=num_samples)  
        
        res = {  
            'observation': {},  
            'action': {}  
        }  
        
        camera_candidates = self._cameras
  
        # 随机选择一个相机候选  
        selected_camera = random.choice(camera_candidates)  
        # 假设我们只处理joint_positions和gripper_open  
        for i, start_index in enumerate(start_indices):  
            # 提取序列  
            action_seq = torch.zeros((sequence_length, 7))  
            gripper_open_seq = np.zeros((sequence_length, 1))
            obs_imgs_tensor = torch.zeros((num_obs, 3, 224 ,224, ))
            loss_weight = np.ones((sequence_length, 1))
            for j in range(sequence_length):  
                index = start_index + j  
                if j < num_obs:
                    img_path = os.path.join(new_path(root_path), selected_camera, f'{i}.png')
                    from PIL import Image
                    import io
                    img = np.asarray(Image.open(io.BytesIO(self.client.get(img_path))))
                    # img = np.asarray(Image.open(img_path))
                    
                    obs_imgs_tensor[i] = self.data_transform(img)


                
                # print(len_trajectory, index)
                if index < len_trajectory: obs = observations.__getitem__(index)  
                # if index < len_trajectory -1:
                #     obs_next = observations.__getitem__(index+1)  
                #     pose1 = torch.tensor(obs.joint_positions)
                #     pose2 = torch.tensor(obs_next.joint_positions)

                #     translation_delta, rotation_delta = process_traj_v3(torch.eye(4), pose1, pose2)
                #     action_seq[j] = torch.cat([translation_delta, rotation_delta], dim=-1)
                # else:
                #     action_seq[j] = torch.zeros_like(action_seq[j-1])
                if index >= len_trajectory:
                    loss_weight[j] = 0.
                    action_seq[j] = torch.zeros_like(action_seq[j-1])
                    gripper_open_seq[j] = np.zeros_like(gripper_open_seq[j-1])
                else:
                    action_seq[j] = torch.tensor(obs.gripper_pose)
                    gripper_open_seq[j] = obs.gripper_open  
                    loss_weight[j] = 1.
            # 转换为PyTorch张量  
            joint_positions_tensor = action_seq
            gripper_open_tensor = torch.tensor(gripper_open_seq, dtype=torch.float32)
            loss_weight_tensor = torch.tensor(loss_weight, dtype=torch.float32)
            
            # 假设所有序列都放入同一个batch（这里我们简单地重复它们以模拟两个batch）  
            if i == 0:  
                res['observation']['image'] = obs_imgs_tensor.unsqueeze(0)
                res['action']['action'] = joint_positions_tensor.unsqueeze(0)
                res['action']['gripper_closedness_action'] = gripper_open_tensor.unsqueeze(0)
                res['action']['loss_weight'] = loss_weight_tensor.unsqueeze(0)
            else:  
                res['observation']['image'] = torch.cat((res['observation']['image'], obs_imgs_tensor.unsqueeze(0)), dim=0)
                res['action']['action'] = torch.cat((res['action']['action'],   
                                                                joint_positions_tensor.unsqueeze(0)), dim=0)  
                res['action']['gripper_closedness_action'] = torch.cat((res['action']['gripper_closedness_action'],   
                                                                gripper_open_tensor.unsqueeze(0)), dim=0)
                res['action']['loss_weight'] = torch.cat((res['action']['loss_weight'],
                                                                loss_weight_tensor.unsqueeze(0)), dim=0) 
                                                                
        
        
        
        res['action']['terminate_episode'] = torch.zeros_like(res['action']['action'][...,:3])
        # res['action']['gripper_change_pose'] = torch.randn(2, 11)
        # print(res['action']['loss_weight'])

        # 模拟action和其他数据  
        # res['action']['gripper_closedness_action'] = torch.randn(2, sequence_length, 1)  
        # res['action']['loss_weight'] = torch.randn(2, sequence_length, 3)  
        # res['action']['world_vector'] = torch.randn(2, sequence_length, 3)  
        # res['action']['action'] = torch.randn(2, sequence_length, 7)  
        
        # res['observation']['image'] = torch.randn(2,2,3,224,224)
        # res['observation']['natural_language_embedding'] = torch.randn(2,32,77, 768)
        res['observation']['camera_extrinsic_cv'] = torch.randn(num_obs, sequence_length, 4, 4)
        res['instruction'] = task

        
        return res  

    def __getitem__(self, episode_id):
        """
        the episode item: [
            [frame_ids],  # we use chunk and max_episode_length to index it
            [obs_tensors],  # wrt frame_ids, (n_cam, 2, 3, 256, 256)
                obs_tensors[i][:, 0] is RGB, obs_tensors[i][:, 1] is XYZ
            [action_tensors],  # wrt frame_ids, (1, 8)
            [camera_dicts],
            [gripper_tensors],  # wrt frame_ids, (1, 8)
            [trajectories]  # wrt frame_ids, (N_i, 8)
        ]
        """
        # import ipdb;ipdb.set_trace()
        episode_id %= self._num_episodes
        
        while True:
            try:
                task, path = self._episodes[episode_id]
                demo = pickle.loads(self.client.get(os.path.join(new_path(path), 'low_dim_obs.pkl')))
                return self.sample_observations(path, demo, sequence_length=self.sequence_length, task=task)
            except:
                import traceback
                traceback.print_exc()
                print(new_path(path))
                import random
                episode_id = random.randint(0, self._num_episodes)



    def __len__(self):
        if self._num_iters is not None:
            return self._num_iters
        return self._num_episodes

def find_files_with_substring(root_dir, substring):  
    """  
    在指定目录下查找文件名包含特定子字符串的所有文件，并返回它们的完整路径列表。  
      
    :param root_dir: 要搜索的根目录  
    :param substring: 文件名中必须包含的子字符串  
    :return: 包含指定子字符串的文件名的完整路径列表  
    """  
    file_list = []  
    for root, dirs, files in os.walk(root_dir):  
        for file in files:  
            if substring in file: 
                task_name = root.split('train/')[1].split('/')[0]
                file_list.append([task_name, root, ])  
    return file_list  

if __name__ == "__main__":
    # import ipdb;ipdb.set_trace()
    # aa = find_files_with_substring('/home/Anonymous/Anonymous/3d_diffuser_actor/data/peract/raw/train', 'low_dim_obs.pkl')
    # pickle.dump(aa, open("file_list_train.pkl", 'wb'))
    
    
    def load_instructions(
        instructions ,
        tasks  = None,
        variations  = None,
    ):
        if instructions is not None:
            with open(instructions, "rb") as fid:
                data: Instructions = pickle.load(fid)
            if tasks is not None:
                data = {task: var_instr for task, var_instr in data.items() if task in tasks}
            if variations is not None:
                data = {
                    task: {
                        var: instr for var, instr in var_instr.items() if var in variations
                    }
                    for task, var_instr in data.items()
                }
            return data
        return None

    # import ipdb;ipdb.set_trace()
    instruction = load_instructions(
        'instructions/peract/instructions.pkl',
        tasks='place_cups close_jar insert_onto_square_peg light_bulb_in meat_off_grill open_drawer place_shape_in_shape_sorter place_wine_at_rack_location push_buttons put_groceries_in_cupboard put_item_in_drawer put_money_in_safe reach_and_drag slide_block_to_color_target stack_blocks stack_cups sweep_to_dustpan_of_size turn_tap',
        variations=tuple(list(range(0, 200)))
    )
    if instruction is None:
        raise NotImplementedError()
    else:
        taskvar = [
            (task, var)
            for task, var_instr in instruction.items()
            for var in var_instr.keys()
        ]    
    train_dataset = RLBenchDataset(
            root = 's3://Anonymous/rlbench/peract/raw/train/',
            # root='/home/Anonymous/Anonymous/3d_diffuser_actor/data/peract/Peract_packaged/train',
            instructions=instruction,
            taskvar=taskvar,
            data_list_pkl='file_list_train.pkl',
            max_episode_length=5,
            cache_size=600,
            max_episodes_per_task=-1,
            num_iters=600000,
            cameras=['left_shoulder_rgb', 'right_shoulder_rgb', 'overhead_rgb', 'front_rgb'],
            training=True,
            image_rescale=tuple(
                float(x) for x in "0.75,1.25".split(",")
            ),
            return_low_lvl_trajectory=True,
            dense_interpolation=True,
            interpolation_length=100
        )
    import ipdb;ipdb.set_trace()
    for item in train_dataset:
        # print(item.keys())
        # import ipdb;ipdb.set_trace()        
        pass
    item = train_dataset.__getitem__(0)
    
    pass