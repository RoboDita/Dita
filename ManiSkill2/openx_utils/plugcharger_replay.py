import os
import os.path as osp
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import sapien.core as sapien
import h5py
import mani_skill2.envs
import pickle
import sys
from tqdm.auto import tqdm
from scipy.spatial.transform import Rotation
from pickcube_bridge import read_record_data
from robot_utils import transform_pose_cv, transform_pose_to_uv
from geometry import compact_axis_angle_from_quaternion, inv_scale_action
from transforms3d.euler import euler2quat, quat2euler, euler2mat
from transforms3d.quaternions import quat2mat, mat2quat
from mani_skill2.agents.base_controller import CombinedController
from mani_skill2.agents.controllers import *
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.io_utils import load_json
from mani_skill2.utils.sapien_utils import look_at
from mani_skill2.utils.registration import register_env
from mani_skill2.envs.assembly.plug_charger import PlugChargerEnv

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

# basic info
env_id = "PlugCharger-v0"
traj_path = f"demos/v0/rigid_body/{env_id}/trajectory.h5"
h5_file = h5py.File(traj_path, "r")
json_path = traj_path.replace(".h5", ".json")
json_data = load_json(json_path)
traj_size = len(json_data["episodes"])

def cal_action(env, delta, extrinsic_cv, method):
    assert len(delta) == 7
    target_mode = "target" in TARGET_CONTROL_MODE
    controller: CombinedController = env.agent.controller
    arm_controller: PDEEPoseController = controller.controllers["arm"]
    assert arm_controller.config.frame == "ee"
    ee_link: sapien.Link = arm_controller.ee_link
    base_pose = env.agent.robot.pose
    if target_mode:
        prev_ee_pose_at_base = arm_controller._target_pose
    else:
        prev_ee_pose_at_base = base_pose.inv() * ee_link.pose
    prev_ee_pose_at_camera = transform_pose_cv(prev_ee_pose_at_base, base_pose, extrinsic_cv)
    target_ee_pose_at_camera = look_at(prev_ee_pose_at_camera.p + delta[:3], [0, 0, 0])

    if method == 0:
        e_prev = quat2euler(prev_ee_pose_at_camera.q)
        e_target = np.array(e_prev) + delta[3:6]
        target_ee_pose_at_camera.set_q(euler2quat(e_target[0], e_target[1], e_target[2]))
    elif method == 1:
        r_prev = quat2mat(prev_ee_pose_at_camera.q)
        r_diff = euler2mat(delta[3], delta[4], delta[5])
        r_target = r_diff @ r_prev
        target_ee_pose_at_camera.set_q(mat2quat(r_target))
    else:
        print("cal_action: invaild method!")

    target_ee_pose = transform_pose_cv(target_ee_pose_at_camera, base_pose, extrinsic_cv, True)
    ee_pose_at_ee = prev_ee_pose_at_base.inv() * target_ee_pose
    ee_pose_at_ee = np.r_[
        ee_pose_at_ee.p,
        compact_axis_angle_from_quaternion(ee_pose_at_ee.q),
    ]
    arm_action = inv_scale_action(ee_pose_at_ee, -0.1, 0.1)
    action = np.hstack([arm_action, delta[-1]])
    return action

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
def process(episode_idx, camera_idx, h5_file, json_data):
    episodes = json_data["episodes"]
    ep = episodes[episode_idx]
    episode_id = ep["episode_id"]
    traj = h5_file[f"traj_{episode_id}"]
    multi_camear_env_id = env_id + "-MoreCamera"

    env_kwargs = json_data["env_info"]["env_kwargs"]
    env_kwargs["obs_mode"] = "rgbd"
    env_kwargs["control_mode"] = TARGET_CONTROL_MODE
    env = gym.make(multi_camear_env_id, render_mode="human", **env_kwargs)

    reset_kwargs = ep["reset_kwargs"].copy()
    if "seed" in reset_kwargs:
        assert reset_kwargs["seed"] == ep["episode_seed"]
    else:
        reset_kwargs["seed"] = ep["episode_seed"]
    seed = reset_kwargs.pop("seed")
    env.reset(seed=seed, options=reset_kwargs)
    
    # replay
    # get action
    file_name_base = env_id + "_traj_" + str(episode_idx) + "_camera_"
    file_name = osp.join(FILE_PATH, env_id, file_name_base + str(camera_idx) + ".pkl")
    is_base = camera_idx == 0
    camera_key = 'camera_' + str(camera_idx)
    if is_base: camera_key = 'hand_camera'
    data = read_record_data(file_name)
    extrinsic_cv = data['camera_extrinsic_cv']
    intrinsic_cv = data['camera_intrinsic_cv']
    record_steps = data['step']
    actions = []
    observations = []
    poses = []
    for record_step in record_steps:
        actions.append(record_step['action'])
        observations.append(record_step['observation'])
        poses.append(record_step['target_ee_pose'])

    for i in tqdm(range(len(actions))):
        if not is_base:
            delta = actions[i]
            action = cal_action(env, delta, extrinsic_cv, CAL_DELTA_METHOD)
        else:
            action = actions[i]
        obs, _, _, _, _ = env.step(action)
        env.render()

        img_sim, dep_sim, img_record, dep_record = obs['image'][camera_key]['rgb'], obs['image'][camera_key]['depth'], observations[i]['image'], observations[i]['depth']
        plt.clf()
        plt.subplot(2,2,1)
        plt.title(f"env-RGB")
        plt.imshow(img_sim)
        plt.subplot(2,2,2)
        plt.title(f"env-DEPTH")
        plt.imshow(dep_sim[:,:, 0], cmap="gray")
        ax = plt.subplot(2,2,3)
        plt.title(f"record-RGB")

        cv_pose = look_at(poses[i][:3], [0, 0, 0])
        cv_pose.set_q(poses[i][3:])
        uv_prev_pose, dx, dy, dz = transform_pose_to_uv(cv_pose, intrinsic_cv)
        h, w, _ = img_record.shape
        plt.imshow(img_record)
        if uv_prev_pose[0] >= 0 and uv_prev_pose[0] < h and uv_prev_pose[1] >= 0 and uv_prev_pose[1] < w \
              and dx[0] >= 0 and dx[0] < h and dx[1] >= 0 and dx[1] < w \
              and dy[0] >= 0 and dy[0] < h and dy[1] >= 0 and dy[1] < w \
              and dz[0] >= 0 and dz[0] < h and dz[1] >= 0 and dz[1] < w:
            # ax.scatter(uv_prev_pose[0], uv_prev_pose[1], color = 'yellow', s = 10)
            udx, udy = dx[0] - uv_prev_pose[0], dx[1] - uv_prev_pose[1]
            ax.arrow(uv_prev_pose[0], uv_prev_pose[1], udx, udy, width=0.1, head_width=1.5, head_length=0.5, ec='red')
            udx, udy = dy[0] - uv_prev_pose[0], dy[1] - uv_prev_pose[1]
            ax.arrow(uv_prev_pose[0], uv_prev_pose[1], udx, udy, width=0.1, head_width=1.5, head_length=0.5, ec='green')
            udx, udy = dz[0] - uv_prev_pose[0], dz[1] - uv_prev_pose[1]
            ax.arrow(uv_prev_pose[0], uv_prev_pose[1], udx, udy, width=0.1, head_width=1.5, head_length=0.5, ec='blue')

        plt.subplot(2,2,4)
        plt.title(f"record-DEPTH")
        plt.imshow(dep_record[:,:, 0], cmap="gray")
        plt.tight_layout()
        plt.pause(0.001)

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
