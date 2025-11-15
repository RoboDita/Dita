import multiprocessing as mp
import os
import os.path as osp
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import h5py
import mani_skill2.envs
import pickle
import time
from tqdm.auto import tqdm
from scipy.spatial.transform import Rotation
from geometry import from_pd_joint_pos, from_pd_joint_delta_pos
from robot_utils import transform_pose_cv
from transforms3d.euler import euler2quat, quat2euler, mat2euler
from transforms3d.quaternions import mat2quat, quat2mat
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.io_utils import load_json
from mani_skill2.utils.sapien_utils import look_at, vectorize_pose
from mani_skill2.utils.registration import register_env
from mani_skill2.envs.assembly.plug_charger import PlugChargerEnv
from mani_skill2.utils.common import clip_and_scale_action, inv_scale_action

# param
FILE_PATH = f"outputs"
TARGET_CONTROL_MODE = 'pd_ee_delta_pose' # param can be one of ['pd_ee_delta_pose', 'pd_ee_target_delta_pose']

CAL_DELTA_METHOD = 0 # 0:direct 1:tf

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

NUM_PROC = 10

# basic info
env_id = "PlugCharger-v0"
traj_path = f"demos/v0/rigid_body/{env_id}/trajectory.h5"
h5_file = h5py.File(traj_path, "r")
json_path = traj_path.replace(".h5", ".json")
json_data = load_json(json_path)
traj_size = len(json_data["episodes"])
process_traj_size = traj_size
natural_instruction = "plug the charger into the wall receptacle"

# write to *.pkl file
def write_record_data(file_name, data):
    save_dir = osp.dirname(file_name)
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)

# read *.pkl file
def read_record_data(file_name):
    with open(file_name, 'rb') as file:
        data = pickle.load(file)
    return data

def cal_delta(prev, target, gripper, method):
    delta_pos = target.p - prev.p
    if method == 0:
        delta_euler = np.array(quat2euler(target.q)) - np.array(quat2euler(prev.q)) # xyz
    elif method == 1:
        r_target = quat2mat(target.q)
        r_prev = quat2mat(prev.q)
        r_diff = r_target @ r_prev.T
        delta_euler = np.array(mat2euler(r_diff))
    else:
        print("cal_delta: invaild method!")

    delta = np.hstack([delta_pos, delta_euler, gripper])
    return delta

# multi camera register
@register_env("PlugCharger-v0-MoreCamera", max_episode_steps=200, override=True)
class PlugChargerMoreCameraEnv(PlugChargerEnv):
    def _register_cameras(self):
        cameras = []
        for camera_name in CAMERA_NAMES:
            if camera_name == "hand_camera":
                continue
            pose = CAMERA_POSES[camera_name]
            camera = CameraConfig(
                camera_name, pose.p, pose.q, CAMERA_W, CAMERA_H, np.pi / 2, 0.01, 10
            )
            cameras.append(camera)
        return cameras

    def _register_render_cameras(self):
        return []

# process openX format data
def process(episode_idx):
    print("[START] processing episode ", episode_idx + 1, " / ", process_traj_size)
    
    file_name_base = env_id + "_traj_" + str(episode_idx) + "_camera_"

    episodes = json_data["episodes"]
    ep = episodes[episode_idx]
    episode_id = ep["episode_id"]
    traj = h5_file[f"traj_{episode_id}"]
    multi_camear_env_id = env_id + "-MoreCamera"

    ori_env_kwargs = json_data["env_info"]["env_kwargs"]
    ori_env = gym.make(multi_camear_env_id, **ori_env_kwargs)

    env_kwargs = ori_env_kwargs.copy()
    env_kwargs["obs_mode"] = "rgbd"
    env_kwargs["control_mode"] = TARGET_CONTROL_MODE
    env = gym.make(multi_camear_env_id, render_mode="cameras", **env_kwargs)

    reset_kwargs = ep["reset_kwargs"].copy()
    if "seed" in reset_kwargs:
        assert reset_kwargs["seed"] == ep["episode_seed"]
    else:
        reset_kwargs["seed"] = ep["episode_seed"]
    seed = reset_kwargs.pop("seed")
    ori_env.reset(seed=seed, options=reset_kwargs)
    
    reset_kwargs = ep["reset_kwargs"].copy()
    if "seed" in reset_kwargs:
        assert reset_kwargs["seed"] == ep["episode_seed"]
    else:
        reset_kwargs["seed"] = ep["episode_seed"]
    seed = reset_kwargs.pop("seed")
    env.reset(seed=seed, options=reset_kwargs)

    ori_control_mode = ep["control_mode"]
    ori_actions = traj["actions"][:]
    if ori_control_mode == "pd_joint_pos":
        _, actions, target_ee_poses, prev_ee_poses, ee_poses = from_pd_joint_pos(
            TARGET_CONTROL_MODE,
            ori_actions,
            ori_env,
            env)
    elif ori_control_mode == 'pd_joint_delta_pose':
        _, actions, target_ee_poses, prev_ee_poses, ee_poses = from_pd_joint_delta_pos(
            TARGET_CONTROL_MODE,
            ori_actions,
            ori_env,
            env)
    else:
        raise ValueError(f"Unsupported original control mode: {ori_control_mode}")
        
    ori_env.close()
    del ori_env
    reset_kwargs = ep["reset_kwargs"].copy()
    if "seed" in reset_kwargs:
        assert reset_kwargs["seed"] == ep["episode_seed"]
    else:
        reset_kwargs["seed"] = ep["episode_seed"]
    seed = reset_kwargs.pop("seed")
    env.reset(seed=seed, options=reset_kwargs)
    
    step_lists = {x: [] for x in CAMERA_NAMES}
    base_pose = env.agent.robot.pose
    for i in tqdm(range(len(actions))):
        action = actions[i]
        obs, reward, terminated, _, _ = env.step(action)
        
        if i == 0:
            intrinsic_cv = {x: obs['camera_param'][x]['intrinsic_cv'] for x in CAMERA_NAMES}
            extrinsic_cv = {x: obs['camera_param'][x]['extrinsic_cv'] for x in CAMERA_NAMES}
        
        for camera_name in CAMERA_NAMES:
            step = {}
            step['is_first'] = i == 0
            step['is_last'] = step['is_terminal'] = i == len(actions) - 1 # terminated not work
            step['reward'] = reward
            step['discount'] = 1.0
            
            observation = {}
            observation['image'] = obs['image'][camera_name]['rgb']
            observation['depth'] = obs['image'][camera_name]['depth']
            observation['natural_instruction'] = natural_instruction
            step['observation'] = observation
            
            gripper = action[-1]
            if camera_name == "hand_camera":
                step['target_ee_pose'] = vectorize_pose(target_ee_poses[i])
                step['prev_ee_pose'] = vectorize_pose(prev_ee_poses[i])
                step['action'] = action
            else:
                target_ee_pose = transform_pose_cv(target_ee_poses[i], base_pose, extrinsic_cv[camera_name])
                prev_ee_pose = transform_pose_cv(prev_ee_poses[i], base_pose, extrinsic_cv[camera_name])
                step['target_ee_pose'] = vectorize_pose(target_ee_pose)
                step['prev_ee_pose'] = vectorize_pose(prev_ee_pose)
                step['action'] = cal_delta(prev_ee_pose, target_ee_pose, gripper, CAL_DELTA_METHOD)
            
            step_lists[camera_name].append(step)
        
    for camera_save_idx, camera_name in enumerate(CAMERA_NAMES):
        data = {
            'episode_id' : episode_idx * len(CAMERA_NAMES) + camera_save_idx,
            'agent_id' : 0,
            'base_pose' : vectorize_pose(base_pose),
            'camera_intrinsic_cv' : intrinsic_cv[camera_name],
            'camera_extrinsic_cv' : extrinsic_cv[camera_name],
            'step' : step_lists[camera_name]
        }
        file_name = osp.join(FILE_PATH, env_id, file_name_base + str(camera_save_idx) + ".pkl")
        write_record_data(file_name, data)
    
    env.close()
    del env
    
    print("[END] processing episode ", episode_idx + 1, " / ", process_traj_size)
    

# outer script
def main():
    start_time = time.time()
    print("Processing {} trajectories".format(process_traj_size))
    if NUM_PROC > 1:
        print("Using multiprocessing to process trajectories: {} processes".format(NUM_PROC))
        pool = mp.Pool(NUM_PROC)
        pool.map(process, range(process_traj_size))
        pool.close()
    else:
        print("Processing trajectories sequentially")
        for i in range(process_traj_size):
            print("processing trajectory ", i + 1, " / ", process_traj_size)
            process(i)

    end_time = time.time()
    print("Execution time: ", end_time - start_time, "s")


if __name__ == "__main__":
    # spawn is needed due to warp init issue
    mp.set_start_method("spawn")
    main()
