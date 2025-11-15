import argparse
import os

import gymnasium as gym
import numpy as np
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.utils.sapien_utils import look_at
from mani_skill2.utils.visualization.cv2_utils import OpenCVViewer
from mani_skill2.utils.wrappers import RecordEpisode
from PIL import Image
from pytorch_utils import PytorchInference
from robot_utils import cal_action, cal_action_from_pose, eef_pose
from tqdm import tqdm

# param
MAX_EPISODE_STEPS = 200 * 4
TARGET_CONTROL_MODE = "pd_ee_delta_pose"  # param can be one of ['pd_ee_delta_pose', 'pd_ee_target_delta_pose']
CAL_DELTA_METHOD = 2  # 0:direct 1:tf 2:model
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

NATURAL_INSTRUCTIONS = {
    "PickCube-v0": "pick up the red cube and move it to the green point",
    "StackCube-v0": "stack the red cube on the green cube",
    "PickSingleYCB-v0": "pick up the ",
    # "PickSingleEGAD-v0": "Pick up an EGAD object and move it to a goal position",
    "PegInsertionSide-v0": "insert the peg into the horizontal hole in the box",
    # "PlugCharger-v0": "Plug a charger into a wall receptacle",
    "AssemblingKits-v0": "insert the objects into the corresponding holes on the plate",
    # "TurnFaucet-v0": "Turn on a faucet by rotating its handle",
    # "PandaAvoidObstacles-v0": "Navigate the (Panda) robot arm through a region of dense obstacles and move the end-effector to a goal pose",
    # "PickClutterYCB-v0": "Pick up an object from a clutter of 4-8 YCB objects",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PickCube-v0")
    parser.add_argument("-o", "--obs-mode", type=str, default="rgbd")
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("-c", "--control-mode", type=str, default=TARGET_CONTROL_MODE)
    parser.add_argument("--render-mode", type=str, default="cameras")
    parser.add_argument("--record-dir", type=str)
    parser.add_argument("--render-goal-point", type=bool, default=True)
    parser.add_argument("-t", "--test-episodes-num", type=int, default=100)

    args, opts = parser.parse_known_args()

    # Parse env kwargs
    # print("opts:", opts)
    eval_str = lambda x: eval(x[1:]) if x.startswith("@") else x
    env_kwargs = dict((x, eval_str(y)) for x, y in zip(opts[0::2], opts[1::2]))
    # print("env_kwargs:", env_kwargs)
    args.env_kwargs = env_kwargs
    # print("args:", args)

    return args


def main():
    np.set_printoptions(suppress=True, precision=3)
    args = parse_args()
    instruction = NATURAL_INSTRUCTIONS[args.env_id]

    cam = "camera_3"

    env: BaseEnv = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        render_camera_cfgs=dict(width=2 * CAMERA_W, height=2 * CAMERA_H),
        camera_cfgs=dict(
            base_camera=dict(p=CAMERA_POSES[cam].p, q=CAMERA_POSES[cam].q, width=CAMERA_W, height=CAMERA_H),
            hand_camera=dict(width=128, height=128),
        ),
        max_episode_steps=MAX_EPISODE_STEPS * 100,
        **args.env_kwargs,
    )

    record_dir = args.record_dir
    if record_dir:
        record_dir = record_dir.format(env_id=args.env_id)
        env = RecordEpisode(env, record_dir, render_mode=args.render_mode)

    print("Observation space", env.observation_space)
    print("Action space", env.action_space)
    print("Control mode", env.control_mode)
    print("Reward mode", env.reward_mode)

    if args.render_goal_point and hasattr(env, "goal_site"):
        env.goal_site.unhide_visual()
    obs, _ = env.reset(seed=0, options={"model_id": "002_master_chef_can"})
    # obs, _ = env.reset(seed=0)

    assert (obs["image"]["base_camera"]["rgb"] >= 0).all() and (obs["image"]["base_camera"]["rgb"] <= 255).all()

    # opencv_viewer = OpenCVViewer(exit_on_esc=False)

    successes = []
    i = 0
    pbar = tqdm(total=args.test_episodes_num, leave=False)

    # Load model
   
   
    model = PytorchInference(
       
       
    )  
    model.set_natural_instruction(instruction)
    camera_coord = True

    # root_folder = "results_cam3_15frame_256token_18000"
    # root_folder = "results_cam3_5hz_15frame_prev_ee_pose_dropout_4000"
    # root_folder = "results_cam5_5hz_15frame_prev_ee_pose_dropout_clean_run2_8000"
    # root_folder = "results_cam3_5hz_15frame_prev_ee_pose_dropout_clean_22000"
    # root_folder = "results_cam5_5hz_15frame_prev_ee_pose_dropout_clean_base_22000"
    # root_folder = "results_cam3_5hz_15frame_no_wrist_prev_ee_pose_13000_AssemblingKits"
    root_folder = "results_gt_PickSingle_YCB"
    os.makedirs(root_folder, exist_ok=True)

    # path = f"{root_folder}/{i:03d}"
    # os.makedirs(path, exist_ok=True)

    while i < args.test_episodes_num:
        # -------------------------------------------------------------------------- #
        # Visualization
        # -------------------------------------------------------------------------- #
        # render_frame = env.render()
        # opencv_viewer.imshow(render_frame, delay=1)

        # -------------------------------------------------------------------------- #
        # Post-process action
        # -------------------------------------------------------------------------- #

        # Image.fromarray(obs["image"]["base_camera"]["rgb"]).save(f"{path}/{model.frame:03d}.png")

        model.set_eef_pose_base(eef_pose(env))
        model.set_eef_pose(eef_pose(env, obs["camera_param"]["base_camera"]["extrinsic_cv"], camera_coord=camera_coord))
        model.set_observation(rgb=obs["image"]["base_camera"]["rgb"], wrist=obs["image"]["hand_camera"]["rgb"])
        model_output = model.inference(obs["camera_param"]["base_camera"]["extrinsic_cv"])
        """!!! get target pose from model !!!"""
        model_terminate = model_output[-1]
        # delta = cal_action(env, model_output[:8], obs["camera_param"]["base_camera"]["extrinsic_cv"], CAL_DELTA_METHOD)
        # delta = cal_action(env, model_output[:8], np.eye(4), CAL_DELTA_METHOD)
        delta, loop = np.array([1, 1, 1, 1, 1, 1, 1], dtype=float), 1
        while np.max(np.abs(delta[:3])) > 1e-4 and loop > 0:
            loop -= 1
            delta = cal_action_from_pose(env, model_output[:8], obs["camera_param"]["base_camera"]["extrinsic_cv"], camera_coord=camera_coord)
            ee_action = tuple(delta[:6])
            gripper_action = delta[-1]
            action_dict = dict(arm=ee_action, gripper=gripper_action)
            action = env.agent.controller.from_action_dict(action_dict)

            if args.render_goal_point and hasattr(env, "goal_site"):
                env.goal_site.unhide_visual()
            obs, reward, terminated, truncated, info = env.step(action)
            # print("reward", reward)
            # print("terminated", terminated, "truncated", truncated)
            # print("info", info)

        truncated = model.frame >= MAX_EPISODE_STEPS

        if terminated or truncated or model_terminate:
            print(terminated, truncated, model_terminate, flush=True)
            print("success: ", i, info["success"], flush=True)
            successes.append(info["success"])
            model.save_video(os.path.join(root_folder, f'{i:03d}_{info["success"]}.mp4'))
            i += 1
            # path = f"{root_folder}/{i:03d}"
            # os.makedirs(path, exist_ok=True)
            if args.render_goal_point and hasattr(env, "goal_site"):
                env.goal_site.unhide_visual()
            obs, _ = env.reset(seed=i)
            pbar.update(1)
            model.reset_observation()
            # user_instruction = input("Please input the natural instruction:")
            # set_natural_instruction(user_instruction)

    success_rate = np.mean(successes)
    print(args.env_id, "success rate: ", success_rate, flush=True)
    env.close()


if __name__ == "__main__":
    main()
