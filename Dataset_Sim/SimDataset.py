import copy
import os
import pickle
import random
import sys
import time
from collections import defaultdict
from PIL import Image

import datasets
import matplotlib.pyplot as plt
import numpy as np
import pytorch3d.transforms as Pose3d
import torch
import torchvision.transforms

from pytorch3d.transforms import (
    Transform3d,
    matrix_to_euler_angles,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_to_matrix,
)
from scipy.spatial.transform import Rotation as R
from timm.data.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
)
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset, IterableDataset
import copy
import io
import nltk

from utils.data_utils import process_traj_v3, get_pose_cam, add_noise_to_pose



def repair(rotation_delta):

    new_rotation = np.array([0, 0, 0], dtype=float)
    for i in range(3):
        ro = rotation_delta[i].numpy()

        assert ro >= -np.pi * 2, f"rotation delta is smaller than -2pi, something is wrong, ro = {ro}"
        assert ro <= np.pi * 2, f"rotation delta is larger than 2pi, something is wrong, ro = {ro}"

        while ro < -np.pi * 2:
            ro += np.pi * 2
        while ro > np.pi * 2:
            ro -= np.pi * 2

        other_dirc = np.pi * 2 - np.abs(ro)
        if other_dirc < np.abs(ro):
            new_data = other_dirc
            if ro > 0:
                new_data *= -1.0
        else:
            new_data = ro
        new_rotation[i] = new_data
        # assert new_rotation[i] <= np.pi / 2.0, f"rotation delta is larger than pi/2, something is wrong, ro = {new_rotation[i]}"
        # assert new_rotation[i] >= -np.pi / 2.0, f"rotation delta is smaller than -pi/2, something is wrong, ro = {new_rotation[i]}"

    return torch.tensor(new_rotation, dtype=torch.float32)



def get_delta(world2cam, pose1, pose2):

    pose1_in_world = Pose3d.Transform3d().rotate(Pose3d.quaternion_to_matrix(pose1[3:]).mT).translate(*pose1[:3])
    pose2_in_world = Pose3d.Transform3d().rotate(Pose3d.quaternion_to_matrix(pose2[3:]).mT).translate(*pose2[:3])

    # rotation_delta_in_base = Pose3d.matrix_to_euler_angles(pose2_in_world.get_matrix()[0, :3, :3], "XYZ") - Pose3d.matrix_to_euler_angles(pose1_in_world.get_matrix()[0, :3, :3], "XYZ")

    pose1_in_cam = pose1_in_world.compose(world2cam).get_matrix()
    pose2_in_cam = pose2_in_world.compose(world2cam).get_matrix()

    translation_delta = pose2_in_cam[0, -1, :3] - pose1_in_cam[0, -1, :3]
    rotation_delta = Pose3d.matrix_to_euler_angles(pose2_in_cam[0, :3, :3].T, "XYZ") - Pose3d.matrix_to_euler_angles(pose1_in_cam[0, :3, :3].T, "XYZ")

    return translation_delta, rotation_delta


def process_traj(extrinsic, frame1, frame2):

    camera_extrinsic_cv = torch.tensor(extrinsic)

    world2cam = Pose3d.Transform3d(matrix=camera_extrinsic_cv.mT)

    pose1 = torch.tensor(copy.deepcopy(frame1["prev_ee_pose"]))
    pose2 = torch.tensor(copy.deepcopy(frame2["prev_ee_pose"]))
    pose1[0] -= 0.615  # base to world
    pose2[0] -= 0.615  # base to world

    translation_delta, rotation_delta = get_delta(world2cam, pose1, pose2)

    return translation_delta, rotation_delta


@torch.no_grad()
def process_traj_v2(camera_extrinsic_cv, pose1, pose2):

    pose1 = torch.tensor(pose1, dtype=torch.float32)
    pose2 = torch.tensor(pose2, dtype=torch.float32)
    world2cam = torch.tensor(camera_extrinsic_cv, dtype=torch.float32)

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

    pose1_cam_euler = -1.0 * matrix_to_euler_angles(pose1_cam.get_matrix()[0, :3, :3], convention="XYZ")
    pose2_cam_euler = -1.0 * matrix_to_euler_angles(pose2_cam.get_matrix()[0, :3, :3], convention="XYZ")

    diff = pose1_cam_euler - pose2_cam_euler

    return diff.to(torch.float32), pose1_cam_euler, pose2_cam_euler





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


def perturb_extrinsic(camera_extrinsic_cv):

    peturb_matrix = np.random.uniform(-0.05, 0.05, size = (3,4))
    peturb_matrix = copy.deepcopy(camera_extrinsic_cv[:-1, :]) * peturb_matrix
    camera_extrinsic_cv[:-1, :] += peturb_matrix
    return camera_extrinsic_cv

def preprocess_instruction(text):

    tokens = nltk.tokenize.word_tokenize(text)
    tagged_tokens = nltk.pos_tag(tokens)
    
    ins = []
    for i, data in enumerate(tagged_tokens):
        (word, pos) = data
        if pos in ['NN', 'NNS', 'NNP', 'NNPS']:
            if i-1>=0 and tagged_tokens[i-1][1] == 'JJ':
                ins.append(tagged_tokens[i-1][0]+" "+ word)
            else:
                ins.append(word)
    final_ins = ''
    for i in ins:
        final_ins += i + '. '
    return final_ins

def process_segmentation_labels(orig_gt):
    
    def normalize_boxes(obj_gt):

        H,W = obj_gt['mask'][-1]
        xmin,ymin,xmax,ymax = obj_gt['bbox']
        center_x = (xmin + xmax) / 2.0 / W
        center_y = (ymin + ymax) / 2.0 / H
        w = 1.0 * (xmax - xmin) / W
        h = 1.0 * (ymax - ymin) / H
        return center_x, center_y, w, h
 

    object_class_dict_url = '/xxx/xxx/share_data/Anonymous/Dataset/Sim_Data/maniskill_fewcam_datalist/object_class_dict.pkl'
    object_class_dict = pickle.load(open(object_class_dict_url,"rb"))
    
    # new_dict = {
    #     'class_labels' : torch.LongTensor([object_class_dict[k] for k in orig_gt.keys()]),
    #     'boxes' :  torch.FloatTensor([normalize_boxes(orig_gt[k]) for k in orig_gt.keys()]),
    # }
    new_dict = {}
    boxes = [normalize_boxes(orig_gt[k]) for k in orig_gt.keys()]
    detect_labels = [object_class_dict[k] for k in orig_gt.keys()]
    caption_list = ['object' if k =='' else k for k in orig_gt.keys()]
    c = list(zip(boxes, caption_list, detect_labels))
    random.shuffle(c)
    boxes, caption_list, detect_labels = zip(*c)
    uni_caption_list = list(set(caption_list))
    label_map = {}
    for idx in range(len(uni_caption_list)):
        label_map[uni_caption_list[idx]] = idx
    classes = [label_map[cap] for cap in caption_list]
    caption = ' . '.join(uni_caption_list)+ ' .'
    boxes = torch.as_tensor(boxes, dtype = torch.float32).reshape(-1,4)
    classes = torch.as_tensor(classes, dtype = torch.int64)
    detect_labels = torch.as_tensor(detect_labels, dtype = torch.int64)
    caption_list = uni_caption_list
    new_dict["cap_list"] = caption_list
    new_dict["caption"] = caption
    new_dict["boxes"] = boxes
    new_dict["class_labels"] = classes
    new_dict["detect_labels"] = detect_labels

    return new_dict

def custom_collate_fn(data_list):

    batch_size = len(data_list)
    data_sample = data_list[0]
    batch = {"observation": defaultdict(list), "action": defaultdict(list)}
    # import ipdb;ipdb.set_trace()
    for i in range(batch_size):
        for k1 in data_sample.keys():
            for k2 in data_sample[k1].keys():
                if k2 == 'segmentation' or k2 == 'natural_language_instruction':
                    batch[k1][k2] += data_list[i][k1][k2]
                else:
                    batch[k1][k2].append(data_list[i][k1][k2])

    for k1 in data_sample.keys():
            for k2 in data_sample[k1].keys():
                if k2 != 'segmentation' and k2 != 'natural_language_instruction':
                    batch[k1][k2] = torch.stack(batch[k1][k2],dim=0)

    return batch       


class SimDataset(Dataset):

    def __init__(
        self,
        data_path = "s3://Anonymous/Sim_Data/ManiSkill2-0413/",
        # data_path="s3://Anonymous/Sim_Data",
        # language_embedding_path="s3://Anonymous/Sim_Data_language_embeddings_77token_0413",
        language_embedding_path="s3://Anonymous/Sim_Data_language_embeddings_77token_0413",
        traj_per_episode=8,
        traj_length=15,
        cameras_per_scene=6,
        dataset_type=0,
        use_baseframe_action=False,
        split_type="fix_traj",
        data_cam_list=None,
        stride=4,
        num_given_observation = None,
        include_target= 0,
        aug_gripper_status_pose=0,
        use_euler = 0,
        use_language_instruction = False,
        use_segmentation = False,
    ):

        self.data_path = data_path
        self.aug_gripper_status_pose = aug_gripper_status_pose
        self.traj_per_episode = traj_per_episode
        self.traj_length = traj_length
        self.cameras_per_scene = cameras_per_scene
        self.use_baseframe_action = use_baseframe_action
        self.split_type = split_type
        self.language_embedding_path = language_embedding_path
        # import a s3 package as Client
        self.client = Client()
        self.stride = stride
        self.dataset_type = dataset_type
        self.include_target = include_target
        print('include_target:', self.include_target, traj_length, 'traj_per_episode', traj_per_episode, num_given_observation)
        if self.split_type == "overfit":
            self.cache_list = {}

        with open(data_cam_list, "rb") as f:
            self.data_cam_list = pickle.load(f)

        if 'high_resolution' in data_cam_list:
            self.high_resolution = True
            self.data_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    # torchvision.transforms.Resize((448,448),antialias=True),
                    torchvision.transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                ]
            )
        else:
            self.high_resolution = False
            self.data_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    # torchvision.transforms.Normalize(OPENAI_CLIP_MEAN, OPENAI_CLIP_STD)
                    torchvision.transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                ]
            )
        self.high_resolution = False
        self.use_language_instruction = use_language_instruction
        self.use_euler = use_euler
        self.num_given_observation = num_given_observation
        self.use_segmentation = use_segmentation


    def __len__(self):

        total_dataset_length = len(self.data_cam_list)

        if self.split_type == "overfit":
            return total_dataset_length * 100000
        else:
            return total_dataset_length

    @torch.no_grad()
    def construct_traj(self, episode, episode_path):

        stride = self.stride
        base_episode = episode

        gripper_closeness = np.array([episode["step"][_]["action"][-1] for _ in range(len(episode["step"]))])
        gripper_change = np.where(gripper_closeness[1:] != gripper_closeness[:-1])[0]

        # for i in range(len(gripper_change)):
        #     episode["step"][gripper_change[i]]["action"][-1] *= -1

        # gripper_change = np.concatenate([gripper_change - 2, gripper_change -1 , gripper_change, gripper_change + 1, gripper_change + 2])
        # gripper_change = [x for x in gripper_change if x>=0 and x<len(episode['step'])]
        gripper_change = np.concatenate([gripper_change, gripper_change + 1])
        gripper_change.sort()

        # loss_weight = torch.ones((len(episode["step"]), 9))
        # loss_weight[gripper_change, 7] = 100

        episode_step = []
        base_episode_step = []

        start = random.randint(0, stride - 1)

        for i in range(len(gripper_change)):
            episode_step.extend(episode["step"][start : gripper_change[i] : stride])
            base_episode_step.extend(base_episode["step"][start : gripper_change[i] : stride])
            start = gripper_change[i]

        episode_step.extend(episode["step"][start:-1:stride])
        base_episode_step.extend(base_episode["step"][start:-1:stride])
        episode_step.append(episode["step"][-1])
        base_episode_step.append(base_episode["step"][-1])
        episode["step"] = episode_step
        base_episode["step"] = base_episode_step

        # episode["step"] = episode["step"][start::stride] + [episode["step"][-1]]
        # base_episode["step"] = base_episode["step"][start::stride] + [base_episode["step"][-1]]

        steps = len(episode["step"])
        if self.include_target:
            start_frame = np.random.permutation(steps - self.num_given_observation if self.num_given_observation is not None else steps - 2)[: self.traj_per_episode]
        else:
            start_frame = np.random.permutation(max(1, steps - self.traj_length + 1))[: self.traj_per_episode]
        if len(start_frame) < self.traj_per_episode:
            start_frame = np.random.choice(start_frame, self.traj_per_episode, replace=True)
        # start_frame = [0] * self.traj_per_episode
        camera_extrinsic_cv = torch.tensor(episode["step"][0]["camera_extrinsic_cv"])
        gripper_change_list = [0]
        for i in range(1, steps):
            # currently we use observation, we might require to use action_dict, I think there are no apparent difference.
            GRIPER_CLOSE_POSITION = 0.11
            # if episode["steps"][i]["observation"]["gripper_position"] >= GRIPER_CLOSE_POSITION \
            #     != episode["steps"][i-1]["observation"]["gripper_position"] >= GRIPER_CLOSE_POSITION:  # 0.11 indicate status change, we might fix this value
            if episode["step"][i]["action"][-1] != episode["step"][i-1]["action"][-1]:
                gripper_change_list.append(1)
            else:
                gripper_change_list.append(0)
        next_gripper_change_position_dict = {}
        has_changed_gripper_step = False
        tmp_gripper_change_position = steps-1  # TODO
        for i in range(steps-1, -1, -1):
            next_gripper_change_position_dict[i] = tmp_gripper_change_position
            if gripper_change_list[i] and (i + 1 >=steps or i + 1 < steps and not gripper_change_list[i+1]):
                # we use the last change step
                has_changed_gripper_step = True
                tmp_gripper_change_position = i
        
        # TODO interpolate the trajectory
        # if self.dataset_type == 0:
        #     camera_extrinsic_cv = perturb_extrinsic(camera_extrinsic_cv)
        # print(steps - self.num_given_observation if self.num_given_observation is not None else steps - 2, start_frame, steps, self.traj_length)
        trajs = {"observation": defaultdict(list), "action": defaultdict(list)}
        if self.high_resolution and 'PickClutterYCB-v0' in episode_path:
            language_embedding = pickle.loads(self.client.get(os.path.join(self.language_embedding_path.replace('camerabase1','camerabase4'), episode_path)))
        else:    
            language_embedding = pickle.loads(self.client.get(os.path.join(self.language_embedding_path, episode_path)))
        for i in range(self.traj_per_episode):
            frame_idx = start_frame[i]
            traj = {"observation": defaultdict(list), "action": defaultdict(list)}
            # frame_idx = 135
            for j in range(self.traj_length):
                # j = 0
                current_frame_idx = frame_idx + j
                observation = {}
                action = {}

                
                # observation["image"] = self.data_transform(episode["step"][min(steps - 1, current_frame_idx)]["observation"]["image"]).contiguous()
                if (self.num_given_observation is None or j < self.num_given_observation) and current_frame_idx < steps:
                    if self.high_resolution and 'PickClutterYCB-v0' in episode_path:
                        #print(os.path.join(self.data_path.replace('camerabase1','camerabase4'), episode_path.replace('/data.pkl',''), episode["step"][current_frame_idx]["observation"]["image"]),flush = True)
                        observation["image"] = np.load(io.BytesIO(self.client.get(os.path.join(self.data_path.replace('camerabase1','camerabase4'), episode_path.replace('/data.pkl',''), episode["step"][current_frame_idx]["observation"]["image"]))))['data']  
                    else:
                        observation["image"] = np.load(io.BytesIO(self.client.get(os.path.join(self.data_path, episode_path.replace('/data.pkl',''), episode["step"][current_frame_idx]["observation"]["image"]))))['data']
                    if self.use_language_instruction:
                        observation["image_transformed"] = self.data_transform(observation["image"]).contiguous()
                        H,W = observation['image_transformed'].shape[-2:]
                        image = Image.fromarray(observation["image"])
                        image = image.resize((H,W),Image.LANCZOS)
                        observation["image"] = torch.tensor(np.array(image))
                    else:
                        observation["image"] = self.data_transform(observation["image"]).contiguous()

                    if self.use_segmentation:
                        observation['segmentation'] = process_segmentation_labels(episode["step"][current_frame_idx]["observation"]["segmentation"])
                    # image_shape = observation['image'].shape
                # else:
                #     observation['image'] = torch.zeros(image_shape) 
                # observation["wrist_image"] = self.data_transform(base_episode["step"][min(steps - 1, current_frame_idx)]["observation"]["image"]).contiguous()
                # observation["depth_image"] = (
                #     torch.tensor(episode["step"][min(steps - 1, current_frame_idx)]["observation"]["depth"]).permute(2, 0, 1).contiguous() / 10
                # )

                observation["natural_language_embedding"] = torch.tensor(language_embedding, dtype=torch.float32)
                observation["camera_extrinsic_cv"] = camera_extrinsic_cv
                if self.use_language_instruction:
                    natural_instruction = episode['step'][current_frame_idx]['observation']['natural_instruction']
                    observation["natural_language_instruction"] = preprocess_instruction(natural_instruction)
                    if self.use_segmentation:
                        observation["natural_language_instruction"] = observation['segmentation']['caption']



                if current_frame_idx < steps:
                    observation['language_instruction'] = base_episode['step'][current_frame_idx]['observation']['natural_instruction']
                    if episode["step"][min(steps - 1, current_frame_idx)]["is_terminal"] == True:
                        action["terminate_episode"] = torch.tensor([1, 0, 0], dtype=torch.int32)
                    else:
                        action["terminate_episode"] = torch.tensor([0, 1, 0], dtype=torch.int32)
                    action["gripper_closedness_action"] = torch.tensor(
                        episode["step"][min(steps - 1, current_frame_idx)]["action"][-1],
                        dtype=torch.float32,
                    ).unsqueeze(-1)
                else:
                    observation['language_instruction'] = base_episode['step'][0]['observation']['natural_instruction']
                    action["terminate_episode"] = torch.tensor([1, 0, 0], dtype=torch.int32)
                    action["gripper_closedness_action"] = torch.tensor([1], dtype=torch.float32)
                    pass
                

                if current_frame_idx < steps - 1:
                    action["loss_weight"] = torch.ones((9))
                    pose1 = torch.tensor(base_episode["step"][current_frame_idx]["prev_ee_pose"]).clone()
                    pose2 = torch.tensor(base_episode["step"][current_frame_idx + 1]["prev_ee_pose"]).clone()
                    # pose2 = torch.tensor(base_episode["step"][current_frame_idx]["target_ee_pose"]).clone()
                    pose1[0] -= 0.615  # base to world
                    pose2[0] -= 0.615  # base to world
                    action["world_vector"], action["rotation_delta"] = process_traj_v3(
                        (camera_extrinsic_cv if self.use_baseframe_action == False else torch.eye(4)),
                        pose1,
                        pose2,
                        use_euler= self.use_euler
                    )
                    if not self.use_euler:
                        action["rotation_delta"][0] -= 1.0

                    # assert (
                    #     action["world_vector"].min() > -0.16
                    #     and action["world_vector"].max() < 0.16
                    #     and action["rotation_delta"].min() > -0.512
                    #     and action["rotation_delta"].max() < 0.512
                    # ), f"{action['world_vector']}, {action['rotation_delta']}, {episode_path}, {current_frame_idx}, {j}"
                    # if not (
                    #     action["world_vector"].min() > -0.0768
                    #     and action["world_vector"].max() < 0.0768
                    #     and action["rotation_delta"].min() > -0.0768
                    #     and action["rotation_delta"].max() < 0.0768
                    # ):
                    #     # if not (
                    #     #     action["world_vector"].min() > -0.064
                    #     #     and action["world_vector"].max() < 0.064
                    #     #     and action["rotation_delta"].min() > -0.064
                    #     #     and action["rotation_delta"].max() < 0.064
                    #     # ):
                    #     print(f"{action['world_vector']}, {action['rotation_delta']}, {episode_path}, {current_frame_idx}, {j}", flush=True)

                    # if episode["step"][current_frame_idx]["action"][-1] != episode["step"][current_frame_idx + 1]["action"][-1]:
                    #     action["loss_weight"][7] = 100.0
                    if (
                        current_frame_idx > 0
                        and episode["step"][current_frame_idx]["action"][-1] != episode["step"][current_frame_idx - 1]["action"][-1]
                    ):
                        action["loss_weight"][7] = 100.0
                    if (
                        current_frame_idx > 1
                        and episode["step"][current_frame_idx]["action"][-1] != episode["step"][current_frame_idx - 2]["action"][-1]
                    ):
                        action["loss_weight"][7] = 100.0
                    action['abs_tar_pose'] = get_pose_cam(camera_extrinsic_cv if self.use_baseframe_action == False else torch.eye(4), pose2, self.use_euler)
                else:
                    action["loss_weight"] = torch.zeros((9))
                    action["world_vector"] = torch.tensor([0, 0, 0], dtype=torch.float32)
                    if self.use_euler:
                        action["rotation_delta"] = torch.tensor([0, 0, 0], dtype=torch.float32)
                    else:
                        action["rotation_delta"] = torch.tensor([0, 0, 0, 0], dtype=torch.float32)
                    action['abs_tar_pose'] = get_pose_cam(camera_extrinsic_cv if self.use_baseframe_action == False else torch.eye(4), torch.tensor(base_episode["step"][-1]["prev_ee_pose"]).clone(), self.use_euler)

                
                for k in observation.keys():
                    traj["observation"][k].append(observation[k])
                    if k == 'segmentation' or k == 'natural_language_instruction':
                        continue
                    if j == self.traj_length - 1:
                        if k != 'language_instruction':
                            traj["observation"][k] = torch.stack(traj["observation"][k], dim=0)
                        # else:
                        #     traj["observation"][k] = np.stack(traj["observation"][k], axis=0)
                if j == self.traj_length - 1 and 'image' not in observation.keys():
                    traj["observation"]['image'] = torch.stack(traj["observation"]['image'], dim=0)
                for k in action.keys():
                    traj["action"][k].append(action[k])
                    if j == self.traj_length - 1:
                        traj["action"][k] = torch.stack(traj["action"][k], dim=0)

            # cur
            pose_cur = torch.tensor(base_episode["step"][frame_idx]["prev_ee_pose"]).clone()
            tmp_abs_cur_pose = get_pose_cam(camera_extrinsic_cv if self.use_baseframe_action == False else torch.eye(4), pose_cur, self.use_euler)
            traj['action']['abs_tar_pose'] = torch.cat([tmp_abs_cur_pose[None,...],traj['action']['abs_tar_pose']], axis=0)

            for k in traj["observation"].keys():
                if k == 'segmentation' or k == 'natural_language_instruction' :
                    trajs["observation"][k] += traj["observation"][k]
                    continue
                trajs["observation"][k].append(traj["observation"][k])
                if i == self.traj_per_episode - 1:
                    if k != 'language_instruction':
                        trajs["observation"][k] = torch.stack(trajs["observation"][k], dim=0)
                    # else:
                    #     trajs["observation"][k] = np.stack(trajs["observation"][k], axis=0)
            for k in traj["action"].keys():
                trajs["action"][k].append(traj["action"][k])
                if i == self.traj_per_episode - 1:
                    trajs["action"][k] = torch.stack(trajs["action"][k], dim=0)
            # if self.use_language_instruction:
            #     natural_instrction = episode['step'][0]['observation']['natural_instruction']
            #     trajs["observation"]["natural_language_instruction"] = preprocess_instruction(natural_instrction)
                 


            if has_changed_gripper_step:
                # the target position of current observation
                # we use the gripper change step idx to indicate
                # frame_idx is the start index
                assert self.num_given_observation is not None, 'num given observation '
                # episode["step"][i]["action"]
                target_position_step = next_gripper_change_position_dict[frame_idx+(self.num_given_observation - 1)]
                target_position_pose = torch.tensor(episode['step'][target_position_step]['prev_ee_pose']).clone()

                target_position_pose[0] -= 0.615

                gripper_change_pose = get_pose_cam(camera_extrinsic_cv if self.use_baseframe_action == False else torch.eye(4), target_position_pose, self.use_euler)
                
                

                t_gripper_position = torch.tensor(episode["step"][target_position_step]["action"][-1], dtype=torch.float32,).unsqueeze(-1)
                if episode["step"][target_position_step]["is_terminal"] == True:
                    t_terminate_episode = torch.tensor([1, 0, 0], dtype=torch.int32)
                else:
                    t_terminate_episode = torch.tensor([0, 1, 0], dtype=torch.int32)
                
                if self.aug_gripper_status_pose == 3:
                    noise_stddev_rot = 0.05  # Standard deviation of noise for Euler angles  
                    noise_stddev_trans = 0.1   # Standard deviation of noise for translation 
                elif self.aug_gripper_status_pose == 4:
                    noise_stddev_rot = 0.4  # Standard deviation of noise for Euler angles  
                    noise_stddev_trans = 0.4   # Standard deviation of noise for translation 
                elif self.aug_gripper_status_pose == 5:
                    noise_stddev_rot = 1. # Standard deviation of noise for Euler angles  
                    noise_stddev_trans = 1.   # Standard deviation of noise for translation 
                else:
                    noise_stddev_rot = 0.02  # Standard deviation of noise for Euler angles  
                    noise_stddev_trans = 0.04   # Standard deviation of noise for translation  
                if self.aug_gripper_status_pose:
                    gripper_change_pose = add_noise_to_pose(gripper_change_pose, noise_stddev_rot, noise_stddev_trans)

                gripper_change_pose = torch.cat([gripper_change_pose, t_gripper_position, t_terminate_episode], dim=-1)

                target_position_step = next_gripper_change_position_dict[0]
                target_position_pose = torch.tensor(episode['step'][target_position_step]['prev_ee_pose']).clone()

                target_position_pose[0] -= 0.615

                gripper_first_change_pose = get_pose_cam(camera_extrinsic_cv if self.use_baseframe_action == False else torch.eye(4), target_position_pose, self.use_euler)
                t_gripper_position = torch.tensor(episode["step"][target_position_step]["action"][-1], dtype=torch.float32,).unsqueeze(-1)
                if episode["step"][target_position_step]["is_terminal"] == True:
                    t_terminate_episode = torch.tensor([1, 0, 0], dtype=torch.int32)
                else:
                    t_terminate_episode = torch.tensor([0, 1, 0], dtype=torch.int32)
                if self.aug_gripper_status_pose:
                    gripper_first_change_pose = add_noise_to_pose(gripper_first_change_pose, noise_stddev_rot, noise_stddev_trans)

                gripper_first_change_pose = torch.cat([gripper_first_change_pose, t_gripper_position, t_terminate_episode], dim=-1)
                
                
            else:
                gripper_change_pose = torch.zeros(11).to(torch.float32) # indicate no target position or we dont know
                gripper_first_change_pose = torch.zeros(11).to(torch.float32) # indicate no target position or we dont know

            trajs["action"]['gripper_change_pose'].append(gripper_change_pose)
            trajs["action"]['gripper_first_change_pose'].append(gripper_first_change_pose)
            if i == self.traj_per_episode - 1:
                trajs["action"]['gripper_change_pose'] = torch.stack(trajs["action"]['gripper_change_pose'], dim=0)
                trajs["action"]['gripper_first_change_pose'] = torch.stack(trajs["action"]['gripper_first_change_pose'], dim=0)


            if has_changed_gripper_step:
                # the target position of current observation
                # we use the gripper change step idx to indicate
                # frame_idx is the start index
                assert self.num_given_observation is not None, 'num given observation '
                # episode["step"][i]["action"]
                target_position_step = next_gripper_change_position_dict[frame_idx+(self.num_given_observation - 1)]
                target_position_pose = torch.tensor(episode['step'][target_position_step]['prev_ee_pose']).clone()

                target_position_pose[0] -= 0.615

                gripper_change_pose = get_pose_cam(camera_extrinsic_cv if self.use_baseframe_action == False else torch.eye(4), target_position_pose, self.use_euler)
                
                

                t_gripper_position = torch.tensor(episode["step"][target_position_step]["action"][-1], dtype=torch.float32,).unsqueeze(-1)
                if episode["step"][target_position_step]["is_terminal"] == True:
                    t_terminate_episode = torch.tensor([1, 0, 0], dtype=torch.int32)
                else:
                    t_terminate_episode = torch.tensor([0, 1, 0], dtype=torch.int32)
                
                if self.aug_gripper_status_pose == 3:
                    noise_stddev_rot = 0.05  # Standard deviation of noise for Euler angles  
                    noise_stddev_trans = 0.1   # Standard deviation of noise for translation 
                elif self.aug_gripper_status_pose == 4:
                    noise_stddev_rot = 0.4  # Standard deviation of noise for Euler angles  
                    noise_stddev_trans = 0.4   # Standard deviation of noise for translation 
                elif self.aug_gripper_status_pose == 5:
                    noise_stddev_rot = 1. # Standard deviation of noise for Euler angles  
                    noise_stddev_trans = 1.   # Standard deviation of noise for translation 
                else:
                    noise_stddev_rot = 0.02  # Standard deviation of noise for Euler angles  
                    noise_stddev_trans = 0.04   # Standard deviation of noise for translation  
                if self.aug_gripper_status_pose:
                    gripper_change_pose = add_noise_to_pose(gripper_change_pose, noise_stddev_rot, noise_stddev_trans)

                gripper_change_pose = torch.cat([gripper_change_pose, t_gripper_position, t_terminate_episode], dim=-1)

                target_position_step = next_gripper_change_position_dict[0]
                target_position_pose = torch.tensor(episode['step'][target_position_step]['prev_ee_pose']).clone()

                target_position_pose[0] -= 0.615

                gripper_first_change_pose = get_pose_cam(camera_extrinsic_cv if self.use_baseframe_action == False else torch.eye(4), target_position_pose, self.use_euler)
                t_gripper_position = torch.tensor(episode["step"][target_position_step]["action"][-1], dtype=torch.float32,).unsqueeze(-1)
                if episode["step"][target_position_step]["is_terminal"] == True:
                    t_terminate_episode = torch.tensor([1, 0, 0], dtype=torch.int32)
                else:
                    t_terminate_episode = torch.tensor([0, 1, 0], dtype=torch.int32)
                if self.aug_gripper_status_pose:
                    gripper_first_change_pose = add_noise_to_pose(gripper_first_change_pose, noise_stddev_rot, noise_stddev_trans)

                gripper_first_change_pose = torch.cat([gripper_first_change_pose, t_gripper_position, t_terminate_episode], dim=-1)
                
                
            else:
                gripper_change_pose = torch.zeros(11).to(torch.float32) # indicate no target position or we dont know
                gripper_first_change_pose = torch.zeros(11).to(torch.float32) # indicate no target position or we dont know

            trajs["action"]['gripper_change_pose'].append(gripper_change_pose)
            trajs["action"]['gripper_first_change_pose'].append(gripper_first_change_pose)
            if i == self.traj_per_episode - 1:
                trajs["action"]['gripper_change_pose'] = torch.stack(trajs["action"]['gripper_change_pose'], dim=0)
                trajs["action"]['gripper_first_change_pose'] = torch.stack(trajs["action"]['gripper_first_change_pose'], dim=0)

        return trajs

    @torch.no_grad()
    def __getitem__(self, index):
        while True:

            try:
                if self.split_type == "overfit":
                    index %= len(self.data_cam_list)
                # self.data_cam_list[index] = 'PickSingleYCB-v0-0321-part5/PickSingleYCB-v0_065-h_cups_traj_90_camera_5.pkl'
                # task_name = self.data_cam_list[index].split('/')[0]
                # if task_name != 'PickClutterYCB-v0':
                #     data_url = os.path.join(self.data_path, self.data_cam_list[index].replace('.pkl',''), 'data.pkl')
                # else:
                #     data_url = os.path.join(self.data_path, self.data_cam_list[index])
                data_url = os.path.join(self.data_path, self.data_cam_list[index])
                if self.high_resolution and 'PickClutterYCB-v0' in self.data_cam_list[index]:
                    data_url = data_url.replace('camerabase1','camerabase4')
                # base_url = os.path.join(self.data_path, self.data_cam_list[index][:-5] + "0.pkl")
                # base_url = os.path.join(self.data_path, self.data_cam_list[index][: self.data_cam_list[index].find("camera")] + "camera_0.pkl")
                if self.split_type == "overfit":
                    if index in self.cache_list:
                        data_pkl = self.cache_list[index]
                    else:
                        data_pkl = pickle.loads(self.client.get(data_url))
                        self.cache_list[index] = data_pkl
                        # base_pkl = pickle.loads(self.client.get(base_url))
                        # self.cache_list[index] = (data_pkl, base_pkl)
                else:
                    data_pkl = pickle.loads(self.client.get(data_url))
                    # base_pkl = pickle.loads(self.client.get(base_url))
                trajs = self.construct_traj(data_pkl, self.data_cam_list[index])
                
                break
            except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(e, data_url)
                    print(f"Fail to load data {self.data_cam_list[index].replace('.pkl','')}", flush = True)
                    index = random.randint(0, len(self.data_cam_list)-1)

        return trajs

class SimDatasetDumpy(Dataset):
    def __init__(
        self,
        data_path = "s3://Anonymous/Sim_Data/ManiSkill2-0413/",
        # data_path="s3://Anonymous/Sim_Data",
        # language_embedding_path="s3://Anonymous/Sim_Data_language_embeddings_77token_0413",
        language_embedding_path="s3://Anonymous/Sim_Data_language_embeddings_77token_0413",
        traj_per_episode=8,
        traj_length=15,
        cameras_per_scene=6,
        dataset_type=0,
        use_baseframe_action=False,
        split_type="fix_traj",
        data_cam_list=None,
        stride=4,
        num_given_observation = None,
        include_target= 0,
    ):
        self.traj_length = traj_length
        self.num_given_observation = num_given_observation
        self.traj_per_episode = traj_per_episode

        pass
    
    def __len__(self):
        return 10000

    def __getitem__(self, index):

    #         observation image torch.Size([2, 2, 3, 224, 224])
    # observation natural_language_embedding torch.Size([2, 32, 77, 768])
    # observation camera_extrinsic_cv torch.Size([2, 32, 4, 4])
    # action terminate_episode torch.Size([2, 32, 3])
    # action gripper_closedness_action torch.Size([2, 32, 1])
    # action loss_weight torch.Size([2, 32, 9])
    # action world_vector torch.Size([2, 32, 3])
    # action rotation_delta torch.Size([2, 32, 4])
    # action gripper_change_pose torch.Size([2, 11])
        res = {}
        res['observation']={}
        res['action'] = {}
        res['observation']['image'] = torch.randn(2,2,3,224,224)
        res['observation']['natural_language_embedding'] = torch.randn(2,32,77, 768)
        res['observation']['camera_extrinsic_cv'] = torch.randn(2, 32, 4, 4)
        res['instruction'] = 'you are stupid'

        res['action']['terminate_episode'] = torch.randn(2, 32, 3)
        res['action']['gripper_closedness_action'] = torch.randn(2, 32, 1)
        res['action']['loss_weight'] = torch.randn(2, 32, 3)
        res['action']['world_vector'] = torch.randn(2, 32, 3)
        res['action']['rotation_delta'] = torch.randn(2, 32, 3)
        res['action']['gripper_change_pose'] = torch.randn(2, 11)
        res['dataset_name'] = 'a'
        return res
        

if __name__ == "__main__":

    # import ipdb

    # ipdb.set_trace()
    import time

    start = time.time()

    dataset = SimDataset(
        data_path="s3://Anonymous/Sim_Data/ManiSkill2-camerabase3/",
        language_embedding_path="s3://Anonymous/Sim_Data_language_embeddings_77token_camerabase3",
        data_cam_list="/xxx/xxx/share_data/Anonymous/Dataset/Sim_Data/maniskill_fewcam_datalist/balanced_segmentation_train_list.pkl",
        cameras_per_scene=5,
        stride = 4,
        traj_per_episode=8,
        traj_length=2,
        dataset_type=0,
        num_given_observation = 2,
        include_target= 0,
        use_language_instruction = True,
        use_segmentation = True,
        aug_gripper_status_pose=0,
        use_euler=1,
    )
    abs_pose_min = torch.ones(10) * 1000
    abs_pose_max = torch.ones(10) * - 1000
    total = len(dataset)
    for i, item in enumerate(dataset):
        import ipdb;ipdb.set_trace()
        trajectory = torch.cat([item['action']['abs_tar_pose'], item['action']['gripper_closedness_action'], item['action']['terminate_episode']],dim=-1)
        abs_pose_min = torch.minimum(abs_pose_min, trajectory.amin(dim=(0, 1)))
        abs_pose_max = torch.maximum(abs_pose_max, trajectory.amin(dim=(0, 1)))
        
        if i % 100 == 0:
            print(i, total, "wv_min: ", abs_pose_min)
            print(i, total,"wv_max: ", abs_pose_max)
# 400 300000 wv_min:  tensor([-0.7838, -0.5789, -0.3618, -1.8806, -0.8397, -3.1414, -1.0000,  0.0000,
#          0.0000,  0.0000])
# 400 300000 wv_max:  tensor([ 0.3869,  0.1667,  0.6956, -0.0086,  0.7997,  3.1260,  1.0000,  0.0000,
#          1.0000,  0.0000])

# wv_min:  tensor([-0.8146, -0.6030, -0.3618, -0.7054, -0.7065, -0.6001, -0.6994, -1.0000,
#          0.0000,  0.0000,  0.0000])
# wv_max:  tensor([0.3480, 0.3380, 0.8182, 0.9968, 0.8107, 0.8066, 0.9988, 1.0800, 0.0000,
#         1.0000, 0.0000])
            
        pass
    import ipdb;ipdb.set_trace()
    end = time.time()
    print(f"Dataset Init Time: {end - start}s", flush=True)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)
    # start_time = time.time()
    # for i, batch in enumerate(dataloader):
    #     # print(batch['action']['world_vector'].dtype)
    #     # print(batch['action']['rotation_delta'].dtype)
    #     # print(batch['action']['gripper_closedness_action'].dtype)
    #     # print(batch['action']['terminate_episode'].dtype)
    #     # print(batch['observation']['natural_language_embedding'].dtype)
    #     # print(batch['observation']['image'].dtype)
    #     # print(batch['observation']['camera_extrinsic_cv'].dtype)
    #     # print(batch['action']['base_displacement_vertical_rotation'].shape)
    #     # print(batch['action']['base_displacement_vector'].shape)
    #     end_time = time.time()
    #     print(f"data_time: {end_time - start_time}s", flush=True)
    #     start_time = time.time()
    #     # print(batch)
    wv_min = torch.ones(3) * 1000
    wv_max = torch.ones(3) * -1000
    rt_min = torch.ones(4) * 1000
    rt_max = torch.ones(4) * -1000
    # import ipdb

    # ipdb.set_trace()
    total_iter_num = 0
    for i in range(10):
        for ii, batch in enumerate(dataloader):

            import ipdb;ipdb.set_trace()
            total_iter_num += 1
            wv_min = torch.minimum(wv_min, batch["action"]["world_vector"].amin(dim=(0, 1, 2)))
            wv_max = torch.maximum(wv_max, batch["action"]["world_vector"].amax(dim=(0, 1, 2)))
            rt_min = torch.minimum(rt_min, batch["action"]["rotation_delta"].amin(dim=(0, 1, 2)))
            rt_max = torch.maximum(rt_max, batch["action"]["rotation_delta"].amax(dim=(0, 1, 2)))

            import ipdb;ipdb.set_trace()
            print("ii: ", ii, " ok!", flush=True)
            print("wv_min: ", wv_min)
            print("wv_max: ", wv_max)
            print("rt_min: ", rt_min)
            print("rt_max: ", rt_max)

            # if total_iter_num % 100 == 1:
            #     labels = []
            #     for _ in range(9):
            #         labels.append(f"0.0{_}~0.0{_+1}")
            #     labels.append("0.09~0.1")
            #     labels.append(">0.1")
            #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 4))
            #     bars1 = ax1.bar(labels, np.round(data_hist_world_vector / data_hist_world_vector.sum(), decimals=4), color="blue")
            #     ax1.set_title("World vector distribution")
            #     bars2 = ax2.bar(labels, np.round(data_hist_rotation_delta / data_hist_rotation_delta.sum(), decimals=4), color="blue")
            #     ax2.set_title("Rotation delta distribution")
            #     for bar in bars1:
            #         yval = bar.get_height()
            #         ax1.text(bar.get_x() + bar.get_width() / 2, yval, f"{yval}", va="bottom", ha="center")
            #     for bar in bars2:
            #         yval = bar.get_height()
            #         ax2.text(bar.get_x() + bar.get_width() / 2, yval, f"{yval}", va="bottom", ha="center")
            #     plt.tight_layout()
            #     plt.savefig("/xxx/xxx/Anonymous/Robotics/RT-1-X/data_distrubutions.png")
            #     plt.show()

        print("i: ", i, " ok!", flush=True)

    print("wv_min: ", wv_min)
    print("wv_max: ", wv_max)
    print("rt_min: ", rt_min)
    print("rt_max: ", rt_max)

    # data_url = '/xxx/xxx/share_data/Anonymous/Dataset/Sim_Data/PickCube-v0/pickcube_traj_110_camera_0.pkl'
    # with open(data_url, 'rb') as f:
    #     data = pickle.load(f)
    # print(data['step'][0]['prev_ee_pose'])
