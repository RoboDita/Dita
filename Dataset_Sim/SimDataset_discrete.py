import copy
import os
import pickle
import random
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pytorch3d.transforms as Pose3d
import torch
import torchvision.transforms
try:
    from petrel_client.client import Client
except:
    raise ValueError("Import your own client if using s3")
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
)

from torch.utils.data import DataLoader, Dataset
import copy
import io
def get_action_spec(DEVICE):

    action_spec = {
        "world_vector": {
            "tensor": torch.empty((3,), dtype=torch.float32).to(DEVICE),
            "minimum": torch.tensor([-0.0768], dtype=torch.float32).to(DEVICE),
            "maximum": torch.tensor([0.0768], dtype=torch.float32).to(DEVICE),
        },
        "rotation_delta": {
            "tensor": torch.empty((4,), dtype=torch.float32).to(DEVICE),
            "minimum": torch.tensor([-0.0768], dtype=torch.float32).to(DEVICE),
            "maximum": torch.tensor([0.0768], dtype=torch.float32).to(DEVICE),
        },
        "gripper_closedness_action": {
            "tensor": torch.empty((1,), dtype=torch.float32).to(DEVICE),
            "minimum": torch.tensor([-1.0], dtype=torch.float32).to(DEVICE),
            "maximum": torch.tensor([1.0], dtype=torch.float32).to(DEVICE),
        },
        "terminate_episode": {
            "tensor": torch.empty((3,), dtype=torch.int32).to(DEVICE),
            "minimum": torch.tensor([0], dtype=torch.int32).to(DEVICE),
            "maximum": torch.tensor([1], dtype=torch.int32).to(DEVICE),
        },
    }

    return action_spec


def normalize(data, data_min, data_max, norm_max, norm_min):

    # print(data)
    rescale_data = (data - data_min) / (data_max - data_min) * (norm_max - norm_min) + norm_min
    rescale_data = torch.clip(rescale_data, min=norm_min, max=norm_max)
    # print(rescale_data)
    # sys.exit()
    return rescale_data


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


def quaternion_to_euler_radians(w, x, y, z):
    roll = np.arctan2(2 * (w * x + y * z), w**2 + z**2 - (x**2 + y**2))

    sinpitch = 2 * (w * y - z * x)
    pitch = np.arcsin(sinpitch)

    yaw = np.arctan2(2 * (w * z + x * y), w**2 + x**2 - (y**2 + z**2))

    return torch.tensor([roll, pitch, yaw], dtype=torch.float32)


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

class SimDataset(Dataset):

    def __init__(
        self,
        data_path = "s3://your own data path",
        language_embedding_path="s3://Your preprocessing language embedding path",
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

        self.data_path = data_path
        self.traj_per_episode = traj_per_episode
        self.traj_length = traj_length
        self.cameras_per_scene = cameras_per_scene
        self.use_baseframe_action = use_baseframe_action
        self.split_type = split_type
        self.language_embedding_path = language_embedding_path
        self.client = Client()
        self.stride = stride
        self.dataset_type = dataset_type
        self.include_target = include_target
        print('include_target:', self.include_target, traj_length, 'traj_per_episode', traj_per_episode, num_given_observation)
        if self.split_type == "overfit":
            self.cache_list = {}

        with open(data_cam_list, "rb") as f:
            self.data_cam_list = pickle.load(f)

        self.data_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize(OPENAI_CLIP_MEAN, OPENAI_CLIP_STD)
                torchvision.transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ]
        )

        self.num_given_observation = num_given_observation

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

        
        gripper_change = np.concatenate([gripper_change, gripper_change + 1])
        gripper_change.sort()

        

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

        
        steps = len(episode["step"])
        if self.include_target:
            start_frame = np.random.permutation(steps - self.num_given_observation if self.num_given_observation is not None else steps - 2)[: self.traj_per_episode]
        else:
            start_frame = np.random.permutation(max(1, steps - self.traj_length + 1))[: self.traj_per_episode]
        if len(start_frame) < self.traj_per_episode:
            start_frame = np.random.choice(start_frame, self.traj_per_episode, replace=True)
       
        camera_extrinsic_cv = torch.tensor(episode["step"][0]["camera_extrinsic_cv"])
       
        trajs = {"observation": defaultdict(list), "action": defaultdict(list)}
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
                    observation["image"] = self.data_transform(
                            np.load(io.BytesIO(self.client.get(os.path.join(self.data_path, episode_path.replace('/data.pkl',''), episode["step"][current_frame_idx]["observation"]["image"]))))['data']
                        ).contiguous()   
                    image_shape = observation['image'].shape
                

                observation["natural_language_embedding"] = torch.tensor(language_embedding, dtype=torch.float32)
                observation["camera_extrinsic_cv"] = camera_extrinsic_cv

                if current_frame_idx < steps:
                    if episode["step"][min(steps - 1, current_frame_idx)]["is_terminal"] == True:
                        action["terminate_episode"] = torch.tensor([1, 0, 0], dtype=torch.int32)
                    else:
                        action["terminate_episode"] = torch.tensor([0, 1, 0], dtype=torch.int32)
                    action["gripper_closedness_action"] = torch.tensor(
                        episode["step"][min(steps - 1, current_frame_idx)]["action"][-1],
                        dtype=torch.float32,
                    ).unsqueeze(-1)
                else:
                    action["terminate_episode"] = torch.tensor([1, 0, 0], dtype=torch.int32)
                    action["gripper_closedness_action"] = torch.tensor([1], dtype=torch.float32)
                    pass
                

                if current_frame_idx < steps - 1:
                    action["loss_weight"] = torch.ones((9))
                    pose1 = torch.tensor(base_episode["step"][current_frame_idx]["prev_ee_pose"]).clone()
                    pose2 = torch.tensor(base_episode["step"][current_frame_idx + 1]["prev_ee_pose"]).clone()
                    
                    pose1[0] -= 0.615  # base to world
                    pose2[0] -= 0.615  # base to world
                    action["world_vector"], action["rotation_delta"] = process_traj_v3(
                        (camera_extrinsic_cv if self.use_baseframe_action == False else torch.eye(4)),
                        pose1,
                        pose2,
                    )
                    action["rotation_delta"][0] -= 1.0

                    
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
                else:
                    action["loss_weight"] = torch.zeros((9))
                    action["world_vector"] = torch.tensor([0, 0, 0], dtype=torch.float32)
                    action["rotation_delta"] = torch.tensor([0, 0, 0, 0], dtype=torch.float32)

                

                for k in observation.keys():
                    traj["observation"][k].append(observation[k])
                    if j == self.traj_length - 1:
                        traj["observation"][k] = torch.stack(traj["observation"][k], dim=0)
                if j == self.traj_length - 1 and 'image' not in observation.keys():
                        traj["observation"]['image'] = torch.stack(traj["observation"]['image'], dim=0)
                for k in action.keys():
                    traj["action"][k].append(action[k])
                    if j == self.traj_length - 1:
                        traj["action"][k] = torch.stack(traj["action"][k], dim=0)

            for k in traj["observation"].keys():
                trajs["observation"][k].append(traj["observation"][k])
                if i == self.traj_per_episode - 1:
                    trajs["observation"][k] = torch.stack(trajs["observation"][k], dim=0)
            for k in traj["action"].keys():
                trajs["action"][k].append(traj["action"][k])
                if i == self.traj_per_episode - 1:
                    trajs["action"][k] = torch.stack(trajs["action"][k], dim=0)

        return trajs

    @torch.no_grad()
    def __getitem__(self, index):
        while True:

            try:
                if self.split_type == "overfit":
                    index %= len(self.data_cam_list)
               
                data_url = os.path.join(self.data_path, self.data_cam_list[index])
                
                if self.split_type == "overfit":
                    if index in self.cache_list:
                        data_pkl = self.cache_list[index]
                    else:
                        data_pkl = pickle.loads(self.client.get(data_url))
                        self.cache_list[index] = data_pkl
                        
                else:
                    data_pkl = pickle.loads(self.client.get(data_url))
                   
                trajs = self.construct_traj(data_pkl, self.data_cam_list[index])
                break
            except Exception as e:
                    print("Write your own getitem based on your data path if you first use this codebase", flush=True)
                    index = random.randint(0, len(self.data_cam_list)-1)

        return trajs


if __name__ == "__main__":

    # import ipdb

    # ipdb.set_trace()

    pose1 = torch.tensor([-0.0777, 0.2005, 0.7166, 0.6656, 0.5059, 0.5025, -0.2205])
    pose2 = torch.tensor([-0.0757, 0.2048, 0.7143, 0.6678, 0.4971, 0.5037, -0.2305])
    world2cam = torch.tensor(
        [
            [-4.4721e-01, 8.9443e-01, -1.4901e-08, -4.4721e-02],
            [6.6667e-01, 3.3333e-01, -6.6667e-01, 1.3333e-01],
            [-5.9628e-01, -2.9814e-01, -7.4536e-01, 6.8573e-01],
            [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
        ]
    )

    print(process_traj_v3(world2cam, pose1, pose2), flush=True)

    import time

    start = time.time()

    dataset = SimDataset(
        data_path="your data path",
        language_embedding_path="your preprocessed languaged embedding path",
        data_cam_list="datalist pkl file",
        cameras_per_scene=20,
        stride = 2,
        traj_per_episode=2,
        traj_length=32,
        dataset_type=0,
        num_given_observation = 2,
        include_target= 1,
    )

    end = time.time()
    print(f"Dataset Init Time: {end - start}s", flush=True)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    wv_min = torch.ones(3) * 1000
    wv_max = torch.ones(3) * -1000
    rt_min = torch.ones(4) * 1000
    rt_max = torch.ones(4) * -1000
    # import ipdb

    # ipdb.set_trace()
    total_iter_num = 0
    for i in range(10):
        data_iter = iter(dataloader)
        for ii, batch in enumerate(data_iter):
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

            if total_iter_num % 100 == 1:
                labels = []
                for _ in range(9):
                    labels.append(f"0.0{_}~0.0{_+1}")
                labels.append("0.09~0.1")
                labels.append(">0.1")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 4))
                bars1 = ax1.bar(labels, np.round(data_hist_world_vector / data_hist_world_vector.sum(), decimals=4), color="blue")
                ax1.set_title("World vector distribution")
                bars2 = ax2.bar(labels, np.round(data_hist_rotation_delta / data_hist_rotation_delta.sum(), decimals=4), color="blue")
                ax2.set_title("Rotation delta distribution")
                for bar in bars1:
                    yval = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width() / 2, yval, f"{yval}", va="bottom", ha="center")
                for bar in bars2:
                    yval = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width() / 2, yval, f"{yval}", va="bottom", ha="center")
                plt.tight_layout()
                plt.savefig("Your save path")
                plt.show()

        print("i: ", i, " ok!", flush=True)

    print("wv_min: ", wv_min)
    print("wv_max: ", wv_max)
    print("rt_min: ", rt_min)
    print("rt_max: ", rt_max)

   