from collections import defaultdict, Counter
import io
import itertools
import math
import os
import random
from pathlib import Path

import torch

from torch.utils.data import Dataset
from timm.data.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
)
# import a s3 package as Client

def unnormalize(x):
    x = x.clone()
    for i in range(3):
        x[..., i] = x[..., i] * IMAGENET_DEFAULT_STD[i] + IMAGENET_DEFAULT_MEAN[i]

    return x

class CalvinDataset(Dataset):

    def __init__(
        self,
        # required
        root,
        instructions=None,
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
        relative_action=True
    ):
        self._cache = {}
        self._cache_size = cache_size
        self._cameras = cameras
        self._max_episode_length = max_episode_length
        self._num_iters = num_iters
        self._training = training
        self._taskvar = taskvar
        self._return_low_lvl_trajectory = return_low_lvl_trajectory

        self.root = root
        self._relative_action = relative_action

        # For trajectory optimization, initialize interpolation tools


        # Keep variations and useful instructions
        self._instructions = instructions
        self._num_vars = Counter()  # variations of the same task
        
        # If training, initialize augmentation classes
        # if self._training:
        #     self._resize = Resize(scales=image_rescale)
        self.client = Client()


        import numpy as np


        annotations = np.load(io.BytesIO(self.client.get(f"{root}/lang_annotations/auto_lang_ann.npy")), allow_pickle=True).item()
        self.annotations = list(zip(annotations["info"]["indx"], annotations["language"]["ann"]))

        try:
            indices = np.load(io.BytesIO(self.client.get(f"{self.root}/scene_info.npy")), allow_pickle=True).item()
            self.indices = list(range(min([indices[k][0] for k in indices.keys()]), max([indices[k][1] for k in indices.keys()]) + 1))
        except:
            self.indices = list(range(0, max([max(item[0]) for item in self.annotations])))
            pass
        
        


        # Collect and trim all episodes in the dataset
        self._num_episodes = len(self.annotations)
        import torchvision
        self.data_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize(OPENAI_CLIP_MEAN, OPENAI_CLIP_STD)
                torchvision.transforms.Resize((224,224), antialias=True),
                torchvision.transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ]
        )

    def sample_observations(self, start, end, anno, sequence_length: int = 32) -> dict:  

        # ['rgb_static'] (dtype=np.uint8, shape=(200, 200, 3)),
# ['rgb_gripper'] (dtype=np.uint8, shape=(84, 84, 3)),
# ['rgb_tactile'] (dtype=np.uint8, shape=(160, 120, 6)),
# ['depth_static'] (dtype=np.float32, shape=(200, 200)),
# ['depth_gripper'] (dtype=np.float32, shape=(84, 84)),
# ['depth_tactile'] (dtype=np.float32, shape=(160, 120, 2))
#['actions']
# (dtype=np.float32, shape=(7,))
# tcp position (3): x,y,z in absolute world coordinates
# tcp orientation (3): euler angles x,y,z in absolute world coordinates
# gripper_action (1): binary (close = -1, open = 1)

# ['rel_actions']
# (dtype=np.float32, shape=(7,))
# tcp position (3): x,y,z in relative world coordinates normalized and clipped to (-1, 1) with scaling factor 50
# tcp orientation (3): euler angles x,y,z in relative world coordinates normalized and clipped to (-1, 1) with scaling factor 20
# gripper_action (1): binary (close = -1, open = 1)
# ['scene_obs']
# (dtype=np.float32, shape=(24,))
# sliding door (1): joint state
# drawer (1): joint state
# button (1): joint state
# switch (1): joint state
# lightbulb (1): on=1, off=0
# green light (1): on=1, off=0
# red block (6): (x, y, z, euler_x, euler_y, euler_z)
# blue block (6): (x, y, z, euler_x, euler_y, euler_z)
# pink block (6): (x, y, z, euler_x, euler_y, euler_z)
#['robot_obs']
# (dtype=np.float32, shape=(15,))
# tcp position (3): x,y,z in world coordinates
# tcp orientation (3): euler angles x,y,z in world coordinates
# gripper opening width (1): in meter
# arm_joint_states (7): in rad
# gripper_action (1): binary (close = -1, open = 1)


        num_samples = 2  # 您需要的样本数量  
        num_obs = 2
        episode_start_idx = start
        len_trajectory = end - start + 1
        # import ipdb;ipdb.set_trace()
        # print(len_trajectory)
        max_start_index = len_trajectory - num_obs  # 最大的起始索引  
        
        if max_start_index < 0:  
            raise ValueError("Observation list is too short to sample sequences of length {}".format(sequence_length))  
        
        import numpy as np
        # 随机选择两个起始索引  
        start_indices = np.random.randint(-(num_obs-1), max_start_index + 1, size=num_samples)  
        
        res = {  
            'observation': {},  
            'action': {}  
        }  
        

        # 随机选择一个相机候选  

        # 假设我们只处理joint_positions和gripper_open  
        for i, start_index in enumerate(start_indices):  
            # 提取序列  
            action_seq = np.zeros((sequence_length, 6))
            robot_obs_seq = np.zeros((sequence_length, 7))
            gripper_open_seq = np.zeros((sequence_length, 1))
            obs_imgs_tensor = torch.zeros((num_obs, 3, 224 ,224, ))
            obs_imgs1_tensor = torch.zeros((num_obs, 3, 224 ,224, ))
            loss_weight = np.ones((sequence_length, 1))
            start_index = 0
            for j in range(sequence_length):  
                index = episode_start_idx + max(start_index + j , 0)
                ep = np.load(io.BytesIO(self.client.get(f"{self.root}/episode_{self.indices[index]:07d}.npz")), allow_pickle=True)
                if j < num_obs:
                    from PIL import Image
                    img = ep['rgb_static']
                    img_gripper = ep['rgb_gripper']
                    
                    obs_imgs_tensor[j] = self.data_transform(img)
                    obs_imgs1_tensor[j] = self.data_transform(img_gripper)
                    # if start_index == 0 and j == 1:
                    #     import random
                    #     obs_imgs_tensor[j] = obs_imgs_tensor[j] if random.random() > 0.5 else obs_imgs_tensor[j-1]

                if (start_index + j) >= len_trajectory:
                    loss_weight[j] = 0.
                    gripper_open_seq[j] = np.zeros_like(gripper_open_seq[j-1])
                    action_seq[j] = np.zeros_like(action_seq[j-1])
                    robot_obs_seq[j] = np.zeros_like(robot_obs_seq[j-1])
                else:
                    action_seq[j] = ep['rel_actions'][:6]
                    gripper_open_seq[j] = ep['rel_actions'][6:]
                    loss_weight[j] = 1.
                    robot_obs_seq[j][:6] = ep['robot_obs'][:6]
                    robot_obs_seq[j][6:7] = ep['robot_obs'][6:7]
            # 转换为PyTorch张量  
            joint_positions_tensor = torch.tensor(action_seq, dtype=torch.float32)
            gripper_open_tensor = torch.tensor(gripper_open_seq, dtype=torch.float32)
            loss_weight_tensor = torch.tensor(loss_weight, dtype=torch.float32)
            robot_obs_seq_tensor = torch.tensor(robot_obs_seq, dtype=torch.float32)
            # 假设所有序列都放入同一个batch（这里我们简单地重复它们以模拟两个batch）  
            if i == 0:  
                res['observation']['image'] = obs_imgs_tensor.unsqueeze(0)
                res['observation']['wrist_image'] = obs_imgs1_tensor.unsqueeze(0)
                res['action']['action'] = joint_positions_tensor.unsqueeze(0)
                res['action']['gripper_closedness_action'] = gripper_open_tensor.unsqueeze(0)
                res['action']['loss_weight'] = loss_weight_tensor.unsqueeze(0)
                res['action']['abs_tar_pose'] = robot_obs_seq_tensor.unsqueeze(0)
                
            else:  
                res['observation']['image'] = torch.cat((res['observation']['image'], obs_imgs_tensor.unsqueeze(0)), dim=0)
                res['observation']['wrist_image'] = torch.cat((res['observation']['wrist_image'], obs_imgs1_tensor.unsqueeze(0)), dim=0)
                res['action']['action'] = torch.cat((res['action']['action'],   
                                                                joint_positions_tensor.unsqueeze(0)), dim=0)  
                res['action']['gripper_closedness_action'] = torch.cat((res['action']['gripper_closedness_action'],   
                                                                gripper_open_tensor.unsqueeze(0)), dim=0)
                res['action']['loss_weight'] = torch.cat((res['action']['loss_weight'],
                                                                loss_weight_tensor.unsqueeze(0)), dim=0) 
                res['action']['abs_tar_pose'] = torch.cat((res['action']['abs_tar_pose'],
                                                                robot_obs_seq_tensor.unsqueeze(0)), dim=0) 
                                                                
        res['action']['terminate_episode'] = torch.zeros_like(res['action']['action'][...,:3])
        # 模拟action和其他数据  
        # res['action']['gripper_closedness_action'] = torch.randn(2, sequence_length, 1)  
        # res['action']['loss_weight'] = torch.randn(2, sequence_length, 3)  
        # res['action']['world_vector'] = torch.randn(2, sequence_length, 3)  
        # res['action']['action'] = torch.randn(2, sequence_length, 7)  
        
        # res['observation']['image'] = torch.randn(2,2,3,224,224)
        # res['observation']['natural_language_embedding'] = torch.randn(2,32,77, 768)
        res['observation']['camera_extrinsic_cv'] = torch.randn(num_obs, sequence_length, 4, 4)
        res['instruction'] = anno

        
        return res  
 

    def __len__(self):
        return self._num_episodes * 10

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
        
        item1, anno = self.annotations[episode_id]
        start, end = item1
        return self.sample_observations(start, end, anno, sequence_length = self._max_episode_length)



if __name__ == "__main__":
    import pickle
    # def load_instructions(
    #     instructions ,
    #     tasks  = None,
    #     variations  = None,
    # ):
    #     if instructions is not None:
    #         with open(instructions, "rb") as fid:
    #             data: Instructions = pickle.load(fid)
    #         if tasks is not None:
    #             data = {task: var_instr for task, var_instr in data.items() if task in tasks}
    #         if variations is not None:
    #             data = {
    #                 task: {
    #                     var: instr for var, instr in var_instr.items() if var in variations
    #                 }
    #                 for task, var_instr in data.items()
    #             }
    #         return data
    #     return None

    # import ipdb;ipdb.set_trace()
    # instruction = load_instructions(
    #     'instructions/peract/instructions.pkl',
    #     tasks='place_cups close_jar insert_onto_square_peg light_bulb_in meat_off_grill open_drawer place_shape_in_shape_sorter place_wine_at_rack_location push_buttons put_groceries_in_cupboard put_item_in_drawer put_money_in_safe reach_and_drag slide_block_to_color_target stack_blocks stack_cups sweep_to_dustpan_of_size turn_tap',
    #     variations=tuple(list(range(0, 200)))
    # )
    # if instruction is None:
    #     raise NotImplementedError()
    # else:
        # taskvar = [
        #     (task, var)
        #     for task, var_instr in instruction.items()
        #     for var in var_instr.keys()
        # ]    


    # Initialize datasets with arguments
    train_dataset = CalvinDataset(
        root='s3://Anonymous/task_D_D/training',
        taskvar=[
            ("A", 0), ("B", 0), ("C", 0), ("D", 0),
        ],
        max_episode_length=32,
        cache_size=0,
        max_episodes_per_task=-1,
        num_iters=600000,
        training=True,
        image_rescale=tuple(
            float(x) for x in "0.75,1.25".split(",")
        ),
        return_low_lvl_trajectory=False,
        dense_interpolation=1,
        interpolation_length=20,
        relative_action=bool(1)
    )    
    print(len(train_dataset), 'aaa')

    for item in train_dataset:
        import ipdb;ipdb.set_trace()
        from PIL import Image
        aaa = train_dataset.__getitem__(198)
        Image.fromarray((unnormalize(aaa['observation']['image'][0][0].permute(1,2,0))*255).to(torch.uint8).cpu().numpy()).save('temp112.png')
        print(item['action']['action'])
        print(item['action']['loss_weight'])
        pass
    # item = train_dataset.__getitem__(0)
    import ipdb;ipdb.set_trace()        
    pass
