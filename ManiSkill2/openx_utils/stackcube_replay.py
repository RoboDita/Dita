import os
import sys
import os.path as osp
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import h5py
import pickle
from tqdm.auto import tqdm
from scipy.spatial.transform import Rotation
from robot_utils import eef_pose, cal_action_from_pose, transform_pose_cv, transform_pose_to_uv
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.io_utils import load_json
from mani_skill2.utils.sapien_utils import look_at, vectorize_pose
from mani_skill2.utils.registration import register_env
from mani_skill2.envs.pick_and_place.stack_cube import StackCubeEnv


# param
FILE_PATH = f"outputs"
TARGET_CONTROL_MODE = 'pd_ee_delta_pose' # param can be one of ['pd_ee_delta_pose', 'pd_ee_target_delta_pose']

CAMERA_POOL_FILE = "configs/camera_pool_300k.npz"
NUM_CAMERAS = 10
CAMERA_W = 224
CAMERA_H = 224
CAMERA_RESOLUTION_SCALE = 1


# basic info
env_id = "StackCube-v0"
traj_path = f"demos/v0/rigid_body/{env_id}/trajectory.h5"
h5_file = h5py.File(traj_path, "r")
json_path = traj_path.replace(".h5", ".json")
json_data = load_json(json_path)
traj_size = len(json_data["episodes"])

# camera pool
camera_pool = np.load(CAMERA_POOL_FILE)["cameras"]
camera_pool_size = camera_pool.shape[0]


# read *.pkl file
def read_record_data(file_name):
    with open(file_name, 'rb') as file:
        data = pickle.load(file)
    return data


# multi camera register
@register_env("StackCube-v0-MoreCamera", max_episode_steps=200, override=True)
class StackCubeMoreCameraEnv(StackCubeEnv):
    def __init__(self, *args, camera_params=None, **kwargs):
        self.camera_params = camera_params
        super().__init__(*args, **kwargs)
    
    def _register_cameras(self):
        cameras = []
        for camera_name in self.camera_params.keys():
            pose = self.camera_params[camera_name]
            camera = CameraConfig(
                camera_name, pose.p, pose.q, CAMERA_W*CAMERA_RESOLUTION_SCALE, CAMERA_H*CAMERA_RESOLUTION_SCALE, np.pi / 2, 0.01, 10, texture_names=["Color", "Position", "Segmentation"]
            )
            cameras.append(camera)
        return cameras

    def _register_render_cameras(self):
        return []
    
    def check_frame_goal_visibility(self, obs, min_num_pixel=10):
        names_of_actors_to_be_visible = ["cubeA", "cubeB"]
        
        ids_of_actors_to_be_visible = []
        for actor in self.unwrapped.get_actors():
            if actor.name in names_of_actors_to_be_visible:
                ids_of_actors_to_be_visible.append(actor.id)
        
        valid_camera_names = []
        camera_names = list(self.camera_params.keys())
        for camera_name in camera_names:
            actor_id_mask = obs["image"][camera_name]["Segmentation"][:, :, 1]
            is_valid_camera = True
            for actor_id in ids_of_actors_to_be_visible:
                num_actor_pixels = np.sum(actor_id_mask == actor_id)
                if num_actor_pixels < min_num_pixel:
                    is_valid_camera = False
                    break
            if is_valid_camera:
                valid_camera_names.append(camera_name)
        
        return valid_camera_names


# process openX format data
def process(episode_idx, camera_idx, h5_file, json_data):
    # replay
    # get action
    file_name_base = env_id + "_traj_" + str(episode_idx) + "_camera_"
    file_name = osp.join(FILE_PATH, env_id, file_name_base + str(camera_idx) + ".pkl")
    data = read_record_data(file_name)
    
    camera_poses = {}
    camera_key = "hand_camera" if camera_idx == 0 else f"camera_{camera_idx}"
    if camera_key != "hand_camera":
        camera_index_in_pool = data["camera_index_in_pool"]
        camera_poses[camera_key] = look_at(camera_pool[camera_index_in_pool, :3], camera_pool[camera_index_in_pool, 3:6], camera_pool[camera_index_in_pool, 6:9])
    
    episodes = json_data["episodes"]
    ep = episodes[episode_idx]
    episode_id = ep["episode_id"]
    traj = h5_file[f"traj_{episode_id}"]
    multi_camear_env_id = env_id + "-MoreCamera"

    env_kwargs = json_data["env_info"]["env_kwargs"]
    env_kwargs["obs_mode"] = "rgbd"
    env_kwargs["control_mode"] = TARGET_CONTROL_MODE
    env = gym.make(multi_camear_env_id, camera_params=camera_poses, render_mode="human", **env_kwargs)

    reset_kwargs = ep["reset_kwargs"].copy()
    if "seed" in reset_kwargs:
        assert reset_kwargs["seed"] == ep["episode_seed"]
    else:
        reset_kwargs["seed"] = ep["episode_seed"]
    seed = reset_kwargs.pop("seed")
    obs, _ = env.reset(seed=seed, options=reset_kwargs)

    intrinsic_cv = data['camera_intrinsic_cv']
    record_steps = data['step']
    actions = []
    observations = []
    target_ee_poses = []
    for record_step in record_steps:
        actions.append(record_step['action'])
        observations.append(record_step['observation'])
        target_ee_poses.append(record_step['target_ee_pose'])

    for i in tqdm(range(len(actions))):
        target_ee_pose = target_ee_poses[i]
        gripper = actions[i][-1]
        target_ee_pose_with_gripper = np.hstack([target_ee_pose, gripper])
        action = cal_action_from_pose(env, pose=target_ee_pose_with_gripper, extrinsic_cv=np.eye(4), camera_coord=False)
        
        img_sim, dep_sim, img_record, dep_record = obs['image'][camera_key]['rgb'], obs['image'][camera_key]['depth'], observations[i]['image'], observations[i]['depth']
        plt.clf()
        plt.subplot(2,2,1)
        plt.title(f"env-RGB")
        plt.imshow(img_sim)
        
        plt.subplot(2,2,2)
        plt.title(f"env-DEPTH")
        plt.imshow(dep_sim[:,:,0], cmap="gray")
        
        ax = plt.subplot(2,2,3)
        plt.title(f"record-RGB")
        # cv_pose = look_at(target_ee_pose[:3], [0, 0, 0])
        # cv_pose.set_q(target_ee_pose[3:])
        # uv_prev_pose, dx, dy, dz = transform_pose_to_uv(cv_pose, intrinsic_cv)
        # h, w, _ = img_record.shape
        plt.imshow(img_record)
        # if uv_prev_pose[0] >= 0 and uv_prev_pose[0] < h and uv_prev_pose[1] >= 0 and uv_prev_pose[1] < w \
        #       and dx[0] >= 0 and dx[0] < h and dx[1] >= 0 and dx[1] < w \
        #       and dy[0] >= 0 and dy[0] < h and dy[1] >= 0 and dy[1] < w \
        #       and dz[0] >= 0 and dz[0] < h and dz[1] >= 0 and dz[1] < w:
        #     # ax.scatter(uv_prev_pose[0], uv_prev_pose[1], color = 'yellow', s = 10)
        #     udx, udy = dx[0] - uv_prev_pose[0], dx[1] - uv_prev_pose[1]
        #     ax.arrow(uv_prev_pose[0], uv_prev_pose[1], udx, udy, width=0.1, head_width=1.5, head_length=0.5, ec='red')
        #     udx, udy = dy[0] - uv_prev_pose[0], dy[1] - uv_prev_pose[1]
        #     ax.arrow(uv_prev_pose[0], uv_prev_pose[1], udx, udy, width=0.1, head_width=1.5, head_length=0.5, ec='green')
        #     udx, udy = dz[0] - uv_prev_pose[0], dz[1] - uv_prev_pose[1]
        #     ax.arrow(uv_prev_pose[0], uv_prev_pose[1], udx, udy, width=0.1, head_width=1.5, head_length=0.5, ec='blue')

        plt.subplot(2,2,4)
        plt.title(f"record-DEPTH")
        plt.imshow(dep_record[:,:,0], cmap="gray")
        
        plt.tight_layout()
        plt.pause(0.001)
        
        obs, _, _, _, _ = env.step(action)

    env.close()
    del env


# outer script
def main():
    traj_idx = int(sys.argv[1])
    cam_idx = int(sys.argv[2])
    print("Pick trajectory ", traj_idx, " and camera ", cam_idx)
    process(traj_idx, cam_idx, h5_file, json_data)


if __name__ == "__main__":
    main()

