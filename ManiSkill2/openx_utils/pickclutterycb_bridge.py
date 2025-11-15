import multiprocessing as mp
import os
import os.path as osp
import gymnasium as gym
import matplotlib.pyplot as plt
import argparse
import random
import numpy as np
import h5py
import cv2
import pickle
import time
import json
from tqdm.auto import tqdm
from scipy.spatial.transform import Rotation
import datetime
from io import BytesIO
from petrel_client.client import Client

from geometry import from_pd_joint_pos, from_pd_joint_delta_pos
from robot_utils import eef_pose, cal_action_from_pose
from transforms3d.euler import euler2quat, quat2euler, mat2euler
from transforms3d.quaternions import mat2quat, quat2mat
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.io_utils import load_json
from mani_skill2.utils.sapien_utils import look_at, vectorize_pose, set_actor_visibility
from mani_skill2.utils.registration import register_env
from mani_skill2.envs.pick_and_place.pick_clutter import PickClutterYCBEnv

SAVE_TO_CEPH = os.getenv("SAVE_TO_CEPH", "False").lower() == "true"
CEPH_SAVE_BUCKET = os.getenv("CEPH_SAVE_BUCKET", "s3://your default save path")
print(CEPH_SAVE_BUCKET)
# param
FILE_PATH = f"outputs"
TARGET_CONTROL_MODE = 'pd_ee_delta_pose' # param can be one of ['pd_ee_delta_pose', 'pd_ee_target_delta_pose']

CAMERA_POOL_FILE = "configs/camera_pool_300k.npz"
NUM_CAMERAS = 20
CAMERA_W = 224
CAMERA_H = 224
CAMERA_RESOLUTION_SCALE = 1
MIN_NUM_PIXELS_TO_BE_VISIBLE = 16

TRAIN_CAMERA_RATIO = 0.95
TRAIN_TRAJECTORY_RATIO = 0.95

MAX_NUM_TRY = 10
MAX_EMPTY_COUNT = 1000
SEED = 42
NUM_PROC = 10

GEN_FLAG_NEW = 0
GEN_FLAG_EXIST = 1
GEN_FLAG_PARTIAL = 2
GEN_FLAG_FAIL = 3


# basic info
env_id = "PickClutterYCB-v0"
traj_path = f"demos/v0/rigid_body/{env_id}/trajectory.h5"
h5_file = h5py.File(traj_path, "r")
json_path = traj_path.replace(".h5", ".json")
json_data = load_json(json_path)
traj_size = len(json_data["episodes"])
natural_instruction_template = "pick up the {} and move it to the green point"
train_traj_range = (0, int(traj_size * TRAIN_TRAJECTORY_RATIO))
val_traj_range = (train_traj_range[1], traj_size)

# camera pool
camera_pool = np.load(CAMERA_POOL_FILE)["cameras"]
camera_pool_size = camera_pool.shape[0]
train_camera_pool_range = (0, int(camera_pool_size * TRAIN_CAMERA_RATIO))
val_camera_pool_range = (train_camera_pool_range[1], camera_pool_size)

train_camera_pool_range = (11,12)
val_camera_pool_range = (11, 12)

def convert_fn_to_model_category(fn):
    cat = fn[4:]
    ls = cat.split("_")
    if len(ls[0]) == 1:
        ls = ls[1:]
    label = " ".join(ls)
    return label


def log_print(*args):
    pid = os.getpid()
    time_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{time_string}] [PID {pid}] ", *args)


def format_data(data_dict):
    new_data_dict = {}
    new_data_dict['episode_id'] = data_dict['episode_id']
    new_data_dict['agent_id'] = data_dict['agent_id']
    new_data_dict['base_pose'] = data_dict['base_pose']
    new_data_dict['camera_index_in_pool'] = data_dict['camera_index_in_pool']
    new_data_dict['camera_intrinsic_cv'] = data_dict['camera_intrinsic_cv']
    new_data_dict['step'] = []
    
    rgb_dict = {}
    depth_dict = {}
    for i_step, step in enumerate(data_dict['step']):
        new_step = {}
        new_step['is_first'] = step['is_first']
        new_step['is_last'] = step['is_last']
        new_step['is_terminal'] = step['is_terminal']
        new_step['reward'] = step['reward']
        new_step['discount'] = step['discount']
        new_step['camera_extrinsic_cv'] = step['camera_extrinsic_cv']
        new_step['target_ee_pose'] = step['target_ee_pose']
        new_step['prev_ee_pose'] = step['prev_ee_pose']
        new_step['action'] = step['action']
        
        new_observation = {}
        rgb_image = step['observation']['image']
        depth_image = step['observation']['depth']
        rgb_fn = f"image_{i_step}.npz"
        depth_fn = f"depth_{i_step}.npz"
        rgb_dict[rgb_fn] = rgb_image
        depth_dict[depth_fn] = depth_image
        new_observation['image'] = rgb_fn
        new_observation['depth'] = depth_fn
        new_observation['natural_instruction'] = step['observation']['natural_instruction']
        new_observation['is_goal_visible'] = step['observation']['is_goal_visible']
        new_step['observation'] = new_observation
        
        new_data_dict['step'].append(new_step)
    
    return new_data_dict, rgb_dict, depth_dict


def save_format_data(save_dir, data_dict, rgb_dict, depth_dict):
    # save_dir: osp.join(FILE_PATH, env_id, file_name_base + str(camera_save_idx))  previous *.pkl filename with no suffix
    os.makedirs(save_dir, exist_ok=False)
    with open(osp.join(save_dir, "data.pkl"), 'wb') as file:
        pickle.dump(data_dict, file)
    
    for fn, img in rgb_dict.items():
        np.savez_compressed(osp.join(save_dir, fn), data=img)
    for fn, img in depth_dict.items():
        np.savez_compressed(osp.join(save_dir, fn), data=img)


def save_to_ceph(file_name, data_bytes, client):
    client.put(file_name, data_bytes)


def save_format_data_ceph(save_dir, client, data_dict, rgb_dict, depth_dict):
    pkl_fn = osp.join(save_dir, "data.pkl")
    data_bytes = pickle.dumps(data_dict)
    save_to_ceph(pkl_fn, data_bytes, client)
    
    for fn, img in rgb_dict.items():
        buffer = BytesIO()
        np.savez_compressed(buffer, data=img)
        buffer.seek(0)
        data_bytes = buffer.getvalue()
        npz_fn = osp.join(save_dir, fn)
        save_to_ceph(npz_fn, data_bytes, client)

    for fn, img in depth_dict.items():
        buffer = BytesIO()
        np.savez_compressed(buffer, data=img)
        buffer.seek(0)
        data_bytes = buffer.getvalue()
        npz_fn = osp.join(save_dir, fn)
        save_to_ceph(npz_fn, data_bytes, client)


def check_ceph_file_exist(file_name, client):
    return client.contains(file_name)

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
        log_print("cal_delta: invaild method!")

    delta = np.hstack([delta_pos, delta_euler, gripper])
    return delta

# multi camera register
@register_env("PickClutterYCB-v0-MoreCamera", max_episode_steps=200, override=True)
class PickClutterYCBMoreCameraEnv(PickClutterYCBEnv):
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
        names_of_actors_to_be_visible = ["_goal_site", self.obj.name]
        
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
def process(params):
    episode_idx, random_seed, camera_pool_range, client = params
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    log_print("[START] processing episode ", episode_idx + 1)
    
    file_name_base = env_id + "_traj_" + str(episode_idx) + "_camera_"
    
    if SAVE_TO_CEPH:
        num_file_exist = 0
        for camera_save_idx in range(NUM_CAMERAS+1):
            file_name = osp.join(CEPH_SAVE_BUCKET, env_id, file_name_base + str(camera_save_idx), "data.pkl")
            if check_ceph_file_exist(file_name, client):
                num_file_exist += 1
        if num_file_exist == NUM_CAMERAS+1:
            log_print("Episode {} already exists".format(episode_idx + 1))
            log_print("[END] processing episode ", episode_idx + 1)
            return GEN_FLAG_EXIST
        if num_file_exist > 0:
            log_print("Episode {} partially exists, not complete!".format(episode_idx + 1))
            log_print("[END] processing episode ", episode_idx + 1)
            return GEN_FLAG_PARTIAL
    else:
        num_file_exist = 0
        for camera_save_idx in range(NUM_CAMERAS):
            file_name = osp.join(FILE_PATH, env_id, file_name_base + str(camera_save_idx) + ".pkl")
            if osp.exists(file_name):
                num_file_exist += 1
        if num_file_exist == NUM_CAMERAS:
            log_print("Episode {} already exists".format(episode_idx + 1))
            log_print("[END] processing episode ", episode_idx + 1)
            return GEN_FLAG_EXIST
        if num_file_exist > 0:
            log_print("Episode {} partially exists, not complete!".format(episode_idx + 1))
            log_print("[END] processing episode ", episode_idx + 1)
            return GEN_FLAG_PARTIAL
    
    try_count = 0
    while try_count < MAX_NUM_TRY:
        try:
            episodes = json_data["episodes"]
            ep = episodes[episode_idx]
            episode_id = ep["episode_id"]
            traj = h5_file[f"traj_{episode_id}"]
            multi_camear_env_id = env_id + "-MoreCamera"
            
            ori_env_kwargs = json_data["env_info"]["env_kwargs"]
            ori_env = gym.make(env_id, renderer_kwargs={"offscreen_only": True}, **ori_env_kwargs)
            env_kwargs = ori_env_kwargs.copy()
            env_kwargs["obs_mode"] = "rgbd"
            env_kwargs["control_mode"] = TARGET_CONTROL_MODE
            env = gym.make(env_id, render_mode="cameras",renderer_kwargs={"offscreen_only": True}, **env_kwargs)
            
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
            info = {}
            if ori_control_mode == "pd_joint_pos":
                info, actions, target_ee_poses, prev_ee_poses, ee_poses = from_pd_joint_pos(
                    TARGET_CONTROL_MODE,
                    ori_actions,
                    ori_env,
                    env)
            elif ori_control_mode == 'pd_joint_delta_pose':
                info, actions, target_ee_poses, prev_ee_poses, ee_poses = from_pd_joint_delta_pos(
                    TARGET_CONTROL_MODE,
                    ori_actions,
                    ori_env,
                    env)
            else:
                raise ValueError(f"Unsupported original control mode: {ori_control_mode}")
            
            ori_env.close()
            del ori_env
            env.close()
            del env
            
            success = info.get("success", False)
            if not success:
                raise RuntimeError("Failed to convert actions")
            
            camera_param_indexs = list(range(camera_pool_range[0], camera_pool_range[1]))
            random.shuffle(camera_param_indexs)
            valid_camera_params = []
            last_num_valid_cameras = -1
            empty_count = 0
            while len(valid_camera_params) < NUM_CAMERAS:
                camera_poses = {}
                num_cameras = min(NUM_CAMERAS, len(camera_param_indexs))
                
                if num_cameras == 0:
                    raise RuntimeError("No enough valid cameras found")
                
                for i in range(num_cameras):
                    camera_poses["camera_" + str(i+1)] = look_at(camera_pool[camera_param_indexs[i]][:3], camera_pool[camera_param_indexs[i]][3:6], camera_pool[camera_param_indexs[i]][6:9])
                
                ori_env_kwargs = json_data["env_info"]["env_kwargs"]
                ori_env = gym.make(multi_camear_env_id, camera_params=camera_poses,renderer_kwargs={"offscreen_only": True}, **ori_env_kwargs)
                env_kwargs = ori_env_kwargs.copy()
                env_kwargs["obs_mode"] = "rgbd"
                env_kwargs["control_mode"] = TARGET_CONTROL_MODE
                env = gym.make(multi_camear_env_id, camera_params=camera_poses, renderer_kwargs={"offscreen_only": True}, render_mode="cameras", **env_kwargs)
                
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
                set_actor_visibility(env.target_site, 0.0)
                set_actor_visibility(env.goal_site, 1.0)
                obs = env.observation(env.get_obs())
                
                valid_camera_names = env.check_frame_goal_visibility(obs, min_num_pixel=MIN_NUM_PIXELS_TO_BE_VISIBLE*(CAMERA_RESOLUTION_SCALE**2))
                for camera_name in valid_camera_names:
                    valid_camera_params.append((camera_param_indexs[int(camera_name.split("_")[-1])-1], camera_poses[camera_name]))
                log_print("{} valid cameras found".format(len(valid_camera_names)))
                
                if len(valid_camera_names) == 0 and last_num_valid_cameras == 0:
                    empty_count += 1
                    if empty_count == MAX_EMPTY_COUNT:
                        raise RuntimeError("No valid cameras found for too many times")
                if last_num_valid_cameras > 0:
                    empty_count = 0
                
                last_num_valid_cameras = len(valid_camera_names)
                
                camera_param_indexs = camera_param_indexs[num_cameras:]
                
                ori_env.close()
                del ori_env
                env.close()
                del env
            valid_camera_params = valid_camera_params[:NUM_CAMERAS]
            
            ori_env_kwargs = json_data["env_info"]["env_kwargs"]
            env_kwargs = ori_env_kwargs.copy()
            env_kwargs["obs_mode"] = "rgbd"
            env_kwargs["control_mode"] = TARGET_CONTROL_MODE
            camera_poses = {}
            for i in range(NUM_CAMERAS):
                camera_poses["camera_" + str(i+1)] = valid_camera_params[i][1]
            env = gym.make(multi_camear_env_id, camera_params=camera_poses, render_mode="cameras",renderer_kwargs={"offscreen_only": True}, **env_kwargs)
            
            reset_kwargs = ep["reset_kwargs"].copy()
            if "seed" in reset_kwargs:
                assert reset_kwargs["seed"] == ep["episode_seed"]
            else:
                reset_kwargs["seed"] = ep["episode_seed"]
            seed = reset_kwargs.pop("seed")
            env.reset(seed=seed, options=reset_kwargs)
            set_actor_visibility(env.target_site, 0.0)
            set_actor_visibility(env.goal_site, 1.0)
            obs = env.observation(env.get_obs())
            reward = 0.0
            
            camera_names = ["hand_camera"] + sorted(list(camera_poses.keys()), key=lambda x: int(x.split("_")[-1]))
            natural_instruction = natural_instruction_template.format(convert_fn_to_model_category(env.obj.name))
            
            step_lists = {x: [] for x in camera_names}
            base_pose = env.agent.robot.pose
            intrinsic_cvs = {}
            info = {}
            for i in tqdm(range(len(target_ee_poses))):
                if i == 0:
                    for camera_name in camera_names:
                        intrinsic_cv = obs['camera_param'][camera_name]['intrinsic_cv']
                        intrinsic_cv[0, 0] /= CAMERA_RESOLUTION_SCALE
                        intrinsic_cv[0, 2] /= CAMERA_RESOLUTION_SCALE
                        intrinsic_cv[1, 1] /= CAMERA_RESOLUTION_SCALE
                        intrinsic_cv[1, 2] /= CAMERA_RESOLUTION_SCALE
                        intrinsic_cvs[camera_name] = intrinsic_cv
                
                goal_visible_camera_names = env.check_frame_goal_visibility(obs, min_num_pixel=MIN_NUM_PIXELS_TO_BE_VISIBLE*(CAMERA_RESOLUTION_SCALE**2))
                goal_visible_camera_names = set(goal_visible_camera_names)
                
                target_ee_pose = vectorize_pose(target_ee_poses[i])
                prev_ee_pose = vectorize_pose(eef_pose(env, camera_coord=False))
                gripper = actions[i][-1]
                target_ee_pose_with_gripper = np.hstack([target_ee_pose, gripper])
                action = cal_action_from_pose(env, pose=target_ee_pose_with_gripper, extrinsic_cv=np.eye(4), camera_coord=False)
                
                for camera_name in camera_names:
                    step = {}
                    step['is_first'] = i == 0
                    step['is_last'] = step['is_terminal'] = i == (len(target_ee_poses) - 1) # terminated not work
                    step['reward'] = reward
                    step['discount'] = 1.0
                    
                    extrinsic_cv = obs['camera_param'][camera_name]['extrinsic_cv']
                    step['camera_extrinsic_cv'] = extrinsic_cv
                    
                    observation = {}
                    rgb = obs['image'][camera_name]['rgb']
                    depth = obs['image'][camera_name]['depth']
                    if camera_name != "hand_camera":
                        # plt.imshow(rgb)
                        # plt.show()
                        rgb = cv2.resize(rgb, (CAMERA_W, CAMERA_H), interpolation=cv2.INTER_NEAREST)
                        # plt.imshow(rgb)
                        # plt.show()
                        
                        # plt.imshow(depth)
                        # plt.show()
                        depth = cv2.resize(depth[:,:,0], (CAMERA_W, CAMERA_H), interpolation=cv2.INTER_NEAREST)[:,:,np.newaxis]
                        # plt.imshow(depth)
                        # plt.show()
                    observation['image'] = rgb
                    observation['depth'] = depth
                    observation['natural_instruction'] = natural_instruction
                    observation['is_goal_visible'] = camera_name in goal_visible_camera_names if camera_name != "hand_camera" else None
                    step['observation'] = observation
                    
                    step['target_ee_pose'] = target_ee_pose
                    step['prev_ee_pose'] = prev_ee_pose
                    step['action'] = action
                    
                    step_lists[camera_name].append(step)
                
                set_actor_visibility(env.target_site, 0.0)
                set_actor_visibility(env.goal_site, 1.0)
                obs, reward, terminated, _, info = env.step(action)

            env.close()
            del env
            
            success = info.get("success", False)
            if not success:
                raise RuntimeError("Failed to replay actions")
            
            for camera_save_idx, camera_name in enumerate(camera_names):
                data = {
                    'episode_id' : episode_idx * len(camera_names) + camera_save_idx,
                    'agent_id' : 0,
                    'base_pose' : vectorize_pose(base_pose),
                    'camera_index_in_pool' : -1 if camera_name == "hand_camera" else valid_camera_params[int(camera_name.split("_")[-1])-1][0],
                    'camera_intrinsic_cv' : intrinsic_cvs[camera_name],
                    'step' : step_lists[camera_name]
                }
                if not SAVE_TO_CEPH:
                    file_name = osp.join(FILE_PATH, env_id, file_name_base + str(camera_save_idx))
                    save_format_data(file_name, *format_data(data))
                else:
                    file_name = osp.join(CEPH_SAVE_BUCKET, env_id, file_name_base + str(camera_save_idx))
                    save_format_data_ceph(file_name, client, *format_data(data))
        
        except RuntimeError as e:
            log_print("Failed to process episode ", episode_idx + 1)
            log_print(e)
            log_print("Retrying {} times".format(MAX_NUM_TRY - try_count - 1))
            try_count += 1
            continue
        
        except Exception as e:
            raise e
        
        else:
            log_print("[END] processing episode ", episode_idx + 1)
            return GEN_FLAG_NEW
    
    if try_count == MAX_NUM_TRY:
        log_print("After {} retries, failed to process episode {}".format(MAX_NUM_TRY, episode_idx + 1))
        log_print("[END] processing episode ", episode_idx + 1)
        return GEN_FLAG_FAIL
    
    raise RuntimeError("Should not reach here")


def process_wrapper(params):
    train_traj_ranges_this_proc, random_seeds_this_proc, camera_pool_range = params
    
    if SAVE_TO_CEPH:
        client = Client()
    else:
        client = None
    
    new_traj_ids = []
    exist_traj_ids = []
    partial_traj_ids = []
    failed_traj_ids = []
    num_trajs = len(train_traj_ranges_this_proc)
    log_print("Processing {} trajectories".format(num_trajs))
    
    # for i in range(num_trajs):
    for i in range(num_trajs-1, -1, -1):
        flag = process((train_traj_ranges_this_proc[i], random_seeds_this_proc[i], camera_pool_range, client))
        if flag == GEN_FLAG_NEW:
            label = "new"
            new_traj_ids.append(train_traj_ranges_this_proc[i])
        elif flag == GEN_FLAG_EXIST:
            label = "exist"
            exist_traj_ids.append(train_traj_ranges_this_proc[i])
        elif flag == GEN_FLAG_PARTIAL:
            label = "partial"
            partial_traj_ids.append(train_traj_ranges_this_proc[i])
        elif flag == GEN_FLAG_FAIL:
            label = "failed"
            failed_traj_ids.append(train_traj_ranges_this_proc[i])
        else:
            raise ValueError("Invalid flag")
        log_print("Trajectory {} processed, flag: {}".format(train_traj_ranges_this_proc[i] + 1, label))
    
    return new_traj_ids, exist_traj_ids, partial_traj_ids, failed_traj_ids


# outer script
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_proc", type=int, default=100)
    args = parser.parse_args()
    
    NUM_PROC = args.num_proc
    
    random.seed(SEED)
    np.random.seed(SEED)
    
    if SAVE_TO_CEPH:
        log_print("Saving data to Ceph bucket: ", CEPH_SAVE_BUCKET)
    else:
        log_print("Saving data to local path: ", FILE_PATH)
    
    # generate random seeds
    random_seeds = np.random.choice(np.arange(1000000), traj_size, replace=False)
    random_seeds = random_seeds.tolist()
    random.shuffle(random_seeds)
    
    data_split_info = {}
    
    # train split
    # log_print("Generating train split data")
    # new_traj_ids_all = []
    # exist_traj_ids_all = []
    # partial_traj_ids_all = []
    # failed_traj_ids_all = []
    # start_time = time.time()
    # log_print("Processing {} trajectories".format(train_traj_range[1] - train_traj_range[0]))
    # if NUM_PROC > 1:
    #     log_print("Using multiprocessing to process trajectories: {} processes".format(NUM_PROC))
    #     # split the trajectory range into NUM_PROC parts
    #     train_traj_range_parts = np.array_split(np.arange(train_traj_range[0], train_traj_range[1]), NUM_PROC)
    #     random_seeds_parts = np.array_split(random_seeds[train_traj_range[0]:train_traj_range[1]], NUM_PROC)
    #     pool = mp.Pool(NUM_PROC)
    #     results = []
    #     for i_proc in range(NUM_PROC):
    #         result = pool.apply_async(process_wrapper, args=((train_traj_range_parts[i_proc], random_seeds_parts[i_proc], train_camera_pool_range), ))
    #         results.append(result)
    #     for result in results:
    #         output = result.get()
    #         new_traj_ids, exist_traj_ids, partial_traj_ids, failed_traj_ids = output
    #         new_traj_ids_all.extend(new_traj_ids)
    #         exist_traj_ids_all.extend(exist_traj_ids)
    #         partial_traj_ids_all.extend(partial_traj_ids)
    #         failed_traj_ids_all.extend(failed_traj_ids)
    #     pool.close()
    #     pool.join()
    # else:
    #     log_print("Processing trajectories sequentially")
    #     new_traj_ids, exist_traj_ids, partial_traj_ids, failed_traj_ids = process_wrapper((np.arange(train_traj_range[0], train_traj_range[1]), random_seeds[train_traj_range[0]:train_traj_range[1]], train_camera_pool_range))
    #     new_traj_ids_all.extend(new_traj_ids)
    #     exist_traj_ids_all.extend(exist_traj_ids)
    #     partial_traj_ids_all.extend(partial_traj_ids)
    #     failed_traj_ids_all.extend(failed_traj_ids)
    # end_time = time.time()
    # log_print("Execution time: ", end_time - start_time, "s")
    # log_print("Train split data generated")
    
    # data_split_info["train"] = {
    #     "new": np.array(new_traj_ids_all).tolist(),
    #     "exist": np.array(exist_traj_ids_all).tolist(),
    #     "partial": np.array(partial_traj_ids_all).tolist(),
    #     "failed": np.array(failed_traj_ids_all).tolist()
    # }
    
    # val split
    log_print("Generating val split data")
    new_traj_ids_all = []
    exist_traj_ids_all = []
    partial_traj_ids_all = []
    failed_traj_ids_all = []
    start_time = time.time()
    log_print("Processing {} trajectories".format(val_traj_range[1] - val_traj_range[0]))
    if NUM_PROC > 1:
        log_print("Using multiprocessing to process trajectories: {} processes".format(NUM_PROC))
        # split the trajectory range into NUM_PROC parts
        val_traj_range_parts = np.array_split(np.arange(val_traj_range[0], val_traj_range[1]), NUM_PROC)
        random_seeds_parts = np.array_split(random_seeds[val_traj_range[0]:val_traj_range[1]], NUM_PROC)
        pool = mp.Pool(NUM_PROC)
        results = []
        for i_proc in range(NUM_PROC):
            result = pool.apply_async(process_wrapper, args=((val_traj_range_parts[i_proc], random_seeds_parts[i_proc], val_camera_pool_range), ))
            results.append(result)
        for result in results:
            output = result.get()
            new_traj_ids, exist_traj_ids, partial_traj_ids, failed_traj_ids = output
            new_traj_ids_all.extend(new_traj_ids)
            exist_traj_ids_all.extend(exist_traj_ids)
            partial_traj_ids_all.extend(partial_traj_ids)
            failed_traj_ids_all.extend(failed_traj_ids)
        pool.close()
        pool.join()
    else:
        log_print("Processing trajectories sequentially")
        new_traj_ids, exist_traj_ids, partial_traj_ids, failed_traj_ids = process_wrapper((np.arange(val_traj_range[0], val_traj_range[1]), random_seeds[val_traj_range[0]:val_traj_range[1]], val_camera_pool_range))
        new_traj_ids_all.extend(new_traj_ids)
        exist_traj_ids_all.extend(exist_traj_ids)
        partial_traj_ids_all.extend(partial_traj_ids)
        failed_traj_ids_all.extend(failed_traj_ids)
    end_time = time.time()
    log_print("Execution time: ", end_time - start_time, "s")
    log_print("Val split data generated")
    
    data_split_info["val"] = {
        "new": np.array(new_traj_ids_all).tolist(),
        "exist": np.array(exist_traj_ids_all).tolist(),
        "partial": np.array(partial_traj_ids_all).tolist(),
        "failed": np.array(failed_traj_ids_all).tolist()
    }
    
    with open(f"{env_id}_gen_data_info.json", 'w') as f:
        json.dump(data_split_info, f, indent=4)
    
    log_print("Data split info saved")
    
    log_print("All data generated")


if __name__ == "__main__":
    # spawn is needed due to warp init issue
    mp.set_start_method("spawn")
    main()

