# Code is based on GR-MG
import io
import os
from PIL import Image
import random
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as T
import time
import numpy as np
from Dataset_VLA.utils import euler2rotm, rotm2euler
from utils import *
# from petrel_client.client import Client
ACTION_POS_SCALE = 50
ACTION_ROT_SCALE = 20

# source: https://github.com/facebookresearch/drqv2/blob/main/drqv2.py
class RandomShiftsAug(nn.Module):
    # perform random shifts independently for each image in the batch
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = x.float()
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)

class RandomShiftsSingleAug(nn.Module):
    #  perform same random shifts for each image in the batch
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = x.float()
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate") # pad before shift
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0, 2 * self.pad + 1, size=(1, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift = shift.repeat(n, 1, 1, 1)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class CalvinDataset_Policy(Dataset):
    def __init__(self,
                 data_dir,
                 seq_len=10,
                 act_len=5, 
                 forward_n_max=25, 
                 mode='train',
                 subfolder='task_ABC_D',
                 use_data_augmentation=True,
                 task_num=10000,
                 use_play=True,
                 use_labeled=True,
                 wrap_grmg_data=0,
                 ):
        super().__init__()
        '''
        seq_len: length of observation
        act_len: length of output action trajectroy
        forward_n_max:  the maximum interval of goal image
        use_data_augmentation: whether to use RandomShiftsSingleAug
        task_num: It determins the maximum number of trajectories that could be sampled from a trajectory. If you want to train with all the trajectories,
        just set a very great value. When you want to do fewshot experiments, you should set it as a specific value, such as 52( 10% of total data ).
        use_full: whether to use play data
        use_labeled: whether to use data annotated with language
        wrap_grmg_data: This is the default wrap setting
        '''
        self.wrap_grmg_data = wrap_grmg_data
        self.dataset_dir = os.path.join(data_dir, subfolder)
        self.seq_len = seq_len 
        self.act_len = act_len
        self.forward_n_max = forward_n_max
        self.mode = mode
        self.use_data_augmentation = use_data_augmentation
        self.task_num = task_num
        self.use_play=use_play
        self.use_labeled=use_labeled

        self.action_dim = 7
        self.state_dim = 7  # ( x,y,z,r,p,y,gripper) 
        # self.client = Client()

        if mode == 'validate':
            tag = "validation"
        elif mode == 'train':
            tag = "training"
        else:
            raise ValueError("Mode must be either train or validate")
        
        # As we use MAE as Encoder, we need to resize and normalize
        self.input_size = (224, 224)
        self.clip_mean = (0.485, 0.456, 0.406) 
        self.clip_std = (0.229, 0.224, 0.225)

        if self.use_data_augmentation:
            self.static_rgb_preprocess_train = T.Compose([
                RandomShiftsSingleAug(pad=10), # 10 for static rgb
                T.Resize(self.input_size, interpolation=Image.BICUBIC),
                T.Normalize(self.clip_mean, self.clip_std)])
            self.hand_rgb_preprocess_train = T.Compose([
                RandomShiftsSingleAug(pad=4), # 4 for hand rgb
                T.Resize(self.input_size, interpolation=Image.BICUBIC),
                T.Normalize(self.clip_mean, self.clip_std)])
        else:
            self.static_rgb_preprocess_train = T.Compose([
                T.Resize(self.input_size, interpolation=Image.BICUBIC),
                T.Normalize(self.clip_mean, self.clip_std)])
            self.hand_rgb_preprocess_train = T.Compose([
                T.Resize(self.input_size, interpolation=Image.BICUBIC),
                T.Normalize(self.clip_mean, self.clip_std)])
        
        self.static_rgb_preprocess_val = T.Compose([
            T.Resize(self.input_size, interpolation=Image.BICUBIC),
            T.Normalize(self.clip_mean, self.clip_std)])
        self.hand_rgb_preprocess_val = T.Compose([
            T.Resize(self.input_size, interpolation=Image.BICUBIC),
            T.Normalize(self.clip_mean, self.clip_std)])

        self.dataset_dir = os.path.join(self.dataset_dir, tag)
        self._initialize()
        print(f'{len(self)} samples in total')
        

    def load(self,path):
        # Sometimes we can not immediately load the data when training in our GPU cluster. We set this protection mechanism
        t1=time.time()
        while True:
            try:
                result=np.load(path)
                break
            except:
                t2=time.time()
                if (t2-t1) >60:
                    assert False , "bad load"
                else:
                    continue
        return result


    def _initialize(self):
        """Generate the sequence index pair."""        
        self.anns = np.load(
            os.path.join(self.dataset_dir, "lang_annotations", "auto_lang_ann.npy"), 
            allow_pickle=True).item()
        # (traj_index, start, end, act_end, traj_st)
        self.seq_tuple = []
        task_dict = {}
        # import ipdb;ipdb.set_trace()
        n_trajs = len(self.anns['info']['indx'])
        print(f"{n_trajs} labeled trajectories!")

        if self.use_labeled: 
            for traj_idx in range(n_trajs):
                traj_st, traj_ed = self.anns['info']['indx'][traj_idx]
                traj_task = self.anns['language']['task'][traj_idx]
                text=self.anns["language"]["ann"][traj_idx]
                if (traj_ed - self.seq_len - self.act_len + 2) <= traj_st:
                    continue # exclude extremely short trajectory

                # sample trajectories based on task_num
                if traj_task not in task_dict:
                    task_dict[traj_task] = 1
                else:
                    task_dict[traj_task] = task_dict[traj_task] + 1
                if task_dict[traj_task] > self.task_num: # when arriving the maximum number, stop sampling from the task
                    continue
                # the action in the last frame is discarded
                for st in range(traj_st, traj_ed - self.seq_len + 1):
                    ed = st + self.seq_len
                    act_ed = st + self.seq_len + self.act_len - 1 
                    self.seq_tuple.append([traj_idx, st, ed, act_ed, traj_st, traj_ed,text])
        if self.use_play:
            # if you want to train with play data, please first generate play.json, and replace the following path
            # np.load(io.BytesIO(self.client.get(f"{self.dataset_dir}/lang_annotations/auto_lang_ann.npy")), allow_pickle=True)[()]
            # with open(io.BytesIO(self.client.get(os.path.join(self.dataset_dir, "play.json"), 'r'))) as file:
            #     self.play_traj = json.load(file)
            print('useplay')
            import json
            with open(os.path.join(self.dataset_dir, "play.json"), 'r') as file:
                self.play_traj = json.load(file)
            n_trajs = len(self.play_traj['st_ed_list'])
            for traj_idx in range(n_trajs):
                traj_st, traj_ed = self.play_traj['st_ed_list'][traj_idx]
                if (traj_ed - self.seq_len - self.act_len + 2) <= traj_st:
                    continue
                # the action in the last frame is discarded
                for st in range(traj_st, traj_ed - self.seq_len + 1):
                    ed = st + self.seq_len
                    act_ed = st + self.seq_len + self.act_len - 1
                    self.seq_tuple.append([traj_idx, st, ed, act_ed, traj_st, traj_ed,''])
    
    def __len__(self):
        return len(self.seq_tuple)

    def __getitem__(self, index):
    #     world2cam = np.asarray([[ 0.723, -0.03 ,  0.69 ,  0.   ],                                                                                           
    #                     [ 0.514,  0.691, -0.509,  0.   ],                   
    #                     [-0.461,  0.723,  0.515,  0.   ],                                                                                                 
    #                     [ 0.215, -0.263, -4.399,  1.   ]]).T
        curr_tuple = self.seq_tuple[index]
        st = curr_tuple[1]
        ed = curr_tuple[2]
        act_ed = curr_tuple[3]
        traj_st = curr_tuple[4]
        traj_ed = curr_tuple[5]       
        text=curr_tuple[6]

        static_rgbs = []
        hand_rgbs = []
        states = []


        state_buffer = [np.zeros(self.state_dim) for _ in range(self.seq_len + self.act_len - 1)]
        state_valid = [0 for _ in range(self.seq_len + self.act_len - 1)]

        tlen = ed - st # sequence length
        assert tlen == self.seq_len

        for i in range(st, act_ed):
            if i <= traj_ed:
                frame = self.load(os.path.join(self.dataset_dir, f"episode_{i:07d}.npz"))
                
                if i <= traj_ed:
                    state_buffer[i - st] = frame['robot_obs']
                    state_valid[i - st] = 1
            
            if i < ed:
                states += [frame['robot_obs']]
                
                static_rgb = frame['rgb_static']
                static_rgb = Image.fromarray(static_rgb)
                static_rgb = T.ToTensor()(static_rgb.convert("RGB"))
                static_rgbs.append(static_rgb)

                hand_rgb = frame['rgb_gripper']
                hand_rgb = Image.fromarray(hand_rgb)
                hand_rgb = T.ToTensor()(hand_rgb.convert("RGB"))
                hand_rgbs.append(hand_rgb)   
        # we use relative action represented in current end effector coordinate
        action_buffer = []
        action_valid = []
        if act_ed <= traj_ed:
            act_ed_frame = self.load(os.path.join(self.dataset_dir, f"episode_{act_ed:07d}.npz"))
            state_buffer += [act_ed_frame['robot_obs']]
            state_valid += [1]
        else:
            state_buffer += [np.zeros(self.action_dim)] # mask non-existent action
            state_valid += [0]
        assert len(state_buffer) == (self.seq_len + self.act_len)
        assert len(state_valid) == (self.seq_len + self.act_len)
        for k in range(0, self.seq_len + self.act_len - 1):
            
            if (state_valid[k] and state_valid[k+1]):
                xyz0 = state_buffer[k][0:3]
                rpy0 = state_buffer[k][3:6]
                rotm0 = euler2rotm(rpy0)
                xyz1 = state_buffer[k+1][0:3]
                rpy1 = state_buffer[k+1][3:6]
                rotm1 = euler2rotm(rpy1)                
                gripper1 = state_buffer[k+1][-1]
                delta_xyz = rotm0.T @ (xyz1 - xyz0)
                delta_rotm = rotm0.T @ rotm1
                delta_rpy = rotm2euler(delta_rotm)
                temp_action = np.zeros(7)
                temp_action[:3] = delta_xyz * ACTION_POS_SCALE
                temp_action[3:6] = delta_rpy * ACTION_ROT_SCALE
                temp_action[-1] = (gripper1 + 1.0) / 2.0  # (-1, 1) -> (0, 1)
                action_buffer += [temp_action]
                action_valid += [1]
            else:
                action_buffer += [np.zeros(self.action_dim)]
                action_valid += [0]
        
        assert len(action_buffer) == (self.seq_len + self.act_len - 1)
        assert len(action_valid) == (self.seq_len + self.act_len - 1)

        # Prepare goal Image
        goal_min_idx = ed
        goal_max_idx = min(traj_ed + 1, ed + self.forward_n_max)
        goal_ids = list(range(goal_min_idx, goal_max_idx))
        goal_idx = random.choice(goal_ids)
        assert (goal_idx >= ed) and (goal_idx <= traj_ed)
        goal_frame = self.load(os.path.join(self.dataset_dir, f"episode_{goal_idx:07d}.npz"))
        goal_rgb = goal_frame['rgb_static']
        goal_rgb = Image.fromarray(goal_rgb)
        goal_rgb = T.ToTensor()(goal_rgb.convert("RGB"))
        static_rgbs.append(goal_rgb) # put goal_rgb at the end of static_rgbs

        # preprocess all Images
        static_rgbs = torch.stack(static_rgbs, dim=0)
        hand_rgbs = torch.stack(hand_rgbs, dim=0)
        if self.mode == 'train':
            static_rgbs = self.static_rgb_preprocess_train(static_rgbs)
            hand_rgbs = self.hand_rgb_preprocess_train(hand_rgbs)
        else:
            static_rgbs = self.static_rgb_preprocess_val(static_rgbs)
            hand_rgbs = self.static_rgb_preprocess_val(hand_rgbs)
        goal_rgb = static_rgbs[-1]
        static_rgbs = static_rgbs[:-1]

        # transform them into torch tensor
        _, C, H, W = static_rgbs.shape
        rgb_data  = torch.zeros((self.seq_len, C, H, W)).float() # (len, C, H, W)
        _, C, H, W = hand_rgbs.shape
        hand_rgb_data = torch.zeros((self.seq_len, C, H, W)).float() # (len, C, H, W)
        rgb_data [:tlen] = static_rgbs
        hand_rgb_data[:tlen] = hand_rgbs
        goal_rgb_data = goal_rgb.float() # (C, H, W)
    
        # State
        states = np.array(states)
        arm_states = states[:, :6]
        gripper_states = (states[:, -1:] + 1.0) / 2.0 # (-1, 1) -> (0, 1)
        states = np.hstack([arm_states, gripper_states])
        states = torch.from_numpy(states)
        state_data = torch.zeros(self.seq_len, self.state_dim).float() # (len, state_dim)
        state_data[:tlen] = states

        # Relative state
        rel_states = np.zeros((tlen, self.state_dim), dtype=np.float32)
        first_xyz = arm_states[0, 0:3]
        first_rpy = arm_states[0, 3:6]
        first_rotm = euler2rotm(first_rpy)
        first_gripper = gripper_states[0]
        rel_states[0, -1] = first_gripper
        for i in range(1, tlen):
            curr_xyz = arm_states[i, 0:3]
            curr_rpy = arm_states[i, 3:6]
            curr_rotm = euler2rotm(curr_rpy)
            curr_gripper = gripper_states[i]
            rel_rotm = first_rotm.T @ curr_rotm
            rel_rpy = rotm2euler(rel_rotm)
            rel_xyz = first_rotm.T @ (curr_xyz - first_xyz)
            rel_states[i, 0:3] = rel_xyz
            rel_states[i, 3:6] = rel_rpy
            rel_states[i, 6] = curr_gripper
        rel_state_data= torch.zeros(self.seq_len, self.state_dim).float() # (len, state_dim)
        rel_state_data[:tlen] = torch.from_numpy(rel_states)

        # Action and action mask
        actions = []
        action_masks = []
        for i in range(0, ed - st):
            actions.append(action_buffer[i:(i+self.act_len)])
            action_masks.append(action_valid[i:(i+self.act_len)])
        actions = np.array(actions) # (len, act_len, act_dim)
        actions = torch.from_numpy(actions)
        action_data  = torch.zeros(self.seq_len, self.act_len, self.action_dim).float() # (len, act_len, action_dim)
        action_data [:tlen] = actions
        action_masks = np.array(action_masks) # (len, act_len, act_dim)
        action_masks = torch.from_numpy(action_masks).long()
        action_mask_data = torch.zeros(self.seq_len, self.act_len).long() # (len, act_len, action_dim)
        action_mask_data[:tlen] = action_masks

        # Attention mask (should be all 1 for full dataset)
        attention_mask = np.ones(self.seq_len, dtype=np.int32) # (len)
        attention_mask[tlen:] = 0.0
        assert np.sum(attention_mask) == self.seq_len
        attention_mask_data = torch.from_numpy(attention_mask).long()


        #progress
        progress_data=torch.zeros(self.seq_len).float()
        for i in range(self.seq_len):
            progress=(st+i-traj_st)/(traj_ed-traj_st)
            progress_data[i]=progress

        data = dict()
        data['goal_rgb'] = goal_rgb_data # (C, H, W)
        data['rgb'] = rgb_data # (len, C, H, W)
        data['hand_rgb'] = hand_rgb_data # (len, C, H, W)
        data['state'] = state_data # (len, state_dim)
        data['rel_state'] = rel_state_data # (len, state_dim)
        data['action'] = action_data # (len, act_len, action_dim)
        data['attention_mask'] = attention_mask_data # (len,)
        data['action_mask'] = action_mask_data # (len, act_len, action_dim)
        data['text'] = [text]
        data["progress"]=progress_data #(len)
        # import ipdb;ipdb.set_trace()

        # TODO this is to wrap grmg dataset for our model
        if self.wrap_grmg_data == 0:
            # only use a single trajectory
            data['observation'] = {}
            data['action'] = {}
            data['observation']['image'] = rgb_data[None,]
            data['observation']['wrist_image'] = hand_rgb_data[None,]
            data['action']['action'] = action_data[None, 0, ..., :6]
            data['action']['gripper_closedness_action'] = action_data[None, 0, ..., 6:]
            data['action']['loss_weight'] = action_mask_data[:1][...,None]
                                                            
            data['action']['terminate_episode'] = torch.zeros_like(data['action']['action'][...,:3])

            data['action']['abs_tar_pose'] = data['state'][None, ]
            # import ipdb;ipdb.set_trace()
            # 模拟action和其他数据  
            # res['action']['gripper_closedness_action'] = torch.randn(2, sequence_length, 1)  
            # res['action']['loss_weight'] = torch.randn(2, sequence_length, 3)  
            # res['action']['world_vector'] = torch.randn(2, sequence_length, 3)  
            # res['action']['action'] = torch.randn(2, sequence_length, 7)  
            
            # res['observation']['image'] = torch.randn(2,2,3,224,224)
            # res['observation']['natural_language_embedding'] = torch.randn(2,32,77, 768)
            data['observation']['camera_extrinsic_cv'] = torch.randn(1, len(action_data), 4, 4)
            data['instruction'] = text
        elif self.wrap_grmg_data == 1:
            # we assume that we have only a single input image
            data['observation'] = {}
            data['action'] = {}
            data['observation']['image'] = rgb_data[:, None,]
            data['observation']['wrist_image'] = hand_rgb_data[:, None,]
            data['action']['action'] = action_data[..., :6]
            data['action']['gripper_closedness_action'] = action_data[..., 6:]
            data['action']['loss_weight'] = action_mask_data[...,None]
                                                            
            data['action']['terminate_episode'] = torch.zeros_like(data['action']['action'][...,:3])

            data['action']['abs_tar_pose'] = data['state'][:, None, ]
            # import ipdb;ipdb.set_trace()
            # 模拟action和其他数据  
            # res['action']['gripper_closedness_action'] = torch.randn(2, sequence_length, 1)  
            # res['action']['loss_weight'] = torch.randn(2, sequence_length, 3)  
            # res['action']['world_vector'] = torch.randn(2, sequence_length, 3)  
            # res['action']['action'] = torch.randn(2, sequence_length, 7)  
            
            # res['observation']['image'] = torch.randn(2,2,3,224,224)
            # res['observation']['natural_language_embedding'] = torch.randn(2,32,77, 768)
            data['observation']['camera_extrinsic_cv'] = torch.randn(1, len(action_data), 4, 4)
            data['instruction'] = text
        elif self.wrap_grmg_data == 2:
            # we assume that we have only a two input image
            data['observation'] = {}
            data['action'] = {}
            data['observation']['image'] = torch.cat([rgb_data[:-1, None,], rgb_data[1:, None,]], axis=1)
            data['observation']['wrist_image'] = torch.cat([hand_rgb_data[:-1, None,], hand_rgb_data[1:, None,]], axis=1)

            action_data = torch.cat([action_data[:-1, :1], action_data[1:, :]], axis=1) 
            data['action']['action'] = action_data[..., :6]
            data['action']['gripper_closedness_action'] = action_data[..., 6:]

            action_mask_data = torch.cat([action_mask_data[:-1, :1], action_mask_data[1:, :]], axis=1) 
            data['action']['loss_weight'] = action_mask_data[...,None]
                                                            
            data['action']['terminate_episode'] = torch.zeros_like(data['action']['action'][...,:3])

            
            state_data = torch.cat([rgb_data[:-1, None,], rgb_data[1:, None,]], axis=1)
            data['action']['abs_tar_pose'] = state_data[:, ]
            # import ipdb;ipdb.set_trace()
            # 模拟action和其他数据  
            # res['action']['gripper_closedness_action'] = torch.randn(2, sequence_length, 1)  
            # res['action']['loss_weight'] = torch.randn(2, sequence_length, 3)  
            # res['action']['world_vector'] = torch.randn(2, sequence_length, 3)  
            # res['action']['action'] = torch.randn(2, sequence_length, 7)  
            
            # res['observation']['image'] = torch.randn(2,2,3,224,224)
            # res['observation']['natural_language_embedding'] = torch.randn(2,32,77, 768)
            data['observation']['camera_extrinsic_cv'] = torch.randn(1, len(action_data), 4, 4)
            data['instruction'] = text

        elif self.wrap_grmg_data == 3:
            # we assume that we have only a three input image
            # 1, 2, 3, 4
            # 1, 2, 3
            # 2, 3, 4
            # 3, 4, 5
            # import ipdb;ipdb.set_trace()
            data['observation'] = {}
            data['action'] = {}

            data['observation']['image'] = torch.cat([rgb_data[ii:ii+1, None] for ii in range(3)], axis=1)
            data['observation']['wrist_image'] = torch.cat([hand_rgb_data[ii:ii+1, None] for ii in range(3)], axis=1)


            # 1 frame, -> 10

            action_data = torch.cat([action_data[:1, :self.seq_len-1], action_data[-1:, :]], axis=1) 
            data['action']['action'] = action_data[..., :6]
            data['action']['gripper_closedness_action'] = action_data[..., 6:]

            action_mask_data = torch.cat([action_mask_data[:1, :self.seq_len-1], action_mask_data[-1:, :]], axis=1) 
            data['action']['loss_weight'] = action_mask_data[...,None]
                                                            
            data['action']['terminate_episode'] = torch.zeros_like(data['action']['action'][...,:3])

            
            state_data = torch.cat([state_data[ii:ii+1, None] for ii in range(3)], axis=1)
            data['action']['abs_tar_pose'] = state_data[:, ]
            # import ipdb;ipdb.set_trace()
            # 模拟action和其他数据  
            # res['action']['gripper_closedness_action'] = torch.randn(2, sequence_length, 1)  
            # res['action']['loss_weight'] = torch.randn(2, sequence_length, 3)  
            # res['action']['world_vector'] = torch.randn(2, sequence_length, 3)  
            # res['action']['action'] = torch.randn(2, sequence_length, 7)  
            
            # res['observation']['image'] = torch.randn(2,2,3,224,224)
            # res['observation']['natural_language_embedding'] = torch.randn(2,32,77, 768)
            data['observation']['camera_extrinsic_cv'] = torch.randn(1, len(action_data), 4, 4)
            data['instruction'] = text
            pass
        # data['observation']['camera_extrinsic_cv'][:,:,:4,:4] = torch.from_numpy(world2cam)
        return data
  

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    data_dir = "vc_new:s3://houzhi/"
    subfolder = 'task_ABC_D'
    # seq_len = 2
    # num_pred_action = 31
    # act_len = seq_len + num_pred_action  - 1
    # DS0 = CalvinDataset_Policy(
    #     data_dir,
    #     seq_len,
    #     act_len,
    #     forward_n_max=25,
    #     mode='train',
    #     subfolder=subfolder,
    #     use_data_augmentation=False,
    #     use_play=False,
    #     task_num=10000,
    #     wrap_grmg_data=1)
    
    seq_len = 2
    num_pred_action = 10
    act_len = num_pred_action
    DS0 = CalvinDataset_Policy(
        data_dir,
        seq_len,
        act_len,
        forward_n_max=25,
        mode='train',
        subfolder=subfolder,
        use_data_augmentation=False,
        use_play=False,
        task_num=10000,
        wrap_grmg_data=2,
        )
    
