import argparse

import gymnasium as gym
import numpy as np

from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.utils.sapien_utils import look_at
from mani_skill2.utils.visualization.cv2_utils import OpenCVViewer
from mani_skill2.utils.wrappers import RecordEpisode

from robot_utils import cal_action
from pytorch_utils import PytorchInference

# param
TEST_NUM_EPISODES = 100
MAX_EPISODE_STEPS = 200
TARGET_CONTROL_MODE = "pd_ee_delta_pose" # param can be one of ['pd_ee_delta_pose', 'pd_ee_target_delta_pose']
CAL_DELTA_METHOD = 2 # 0:direct 1:tf 2:model
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
    "StackCube-v0": "Pick up a red cube and place it onto a green one",
    "PickSingleYCB-v0": "Pick up a YCB object and move it to a goal position",
    "PickSingleEGAD-v0": "Pick up an EGAD object and move it to a goal position",
    "PegInsertionSide-v0": "Insert a peg into the horizontal hole in a box",
    "PlugCharger-v0": "Plug a charger into a wall receptacle",
    "AssemblingKits-v0": "Insert an object into the corresponding slot on a board",
    "TurnFaucet-v0": "Turn on a faucet by rotating its handle",
    "PandaAvoidObstacles-v0": "Navigate the (Panda) robot arm through a region of dense obstacles and move the end-effector to a goal pose",
    "PickClutterYCB-v0": "Pick up an object from a clutter of 4-8 YCB objects",
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PickCube-v0")
    parser.add_argument("-o", "--obs-mode", type=str, default='rgbd')
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("-c", "--control-mode", type=str, default=TARGET_CONTROL_MODE)
    parser.add_argument("--render-mode", type=str, default="cameras")
    parser.add_argument("--enable-sapien-viewer", action="store_true")
    parser.add_argument("--record-dir", type=str)
    parser.add_argument("--render-goal-point", type=bool, default=True)
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

    env: BaseEnv = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        render_camera_cfgs=dict(width=2*CAMERA_W, height=2*CAMERA_H),
        camera_cfgs=dict(base_camera=dict(p=CAMERA_POSES['camera_5'].p,\
                                          q=CAMERA_POSES['camera_5'].q,\
                                          width=CAMERA_W, height=CAMERA_H),\
                         hand_camera=dict(width=CAMERA_W, height=CAMERA_H)),
        max_episode_steps=MAX_EPISODE_STEPS,
        **args.env_kwargs
    )

    record_dir = args.record_dir
    if record_dir:
        record_dir = record_dir.format(env_id=args.env_id)
        env = RecordEpisode(env, record_dir, render_mode=args.render_mode)

    print("Observation space", env.observation_space)
    print("Action space", env.action_space)
    print("Control mode", env.control_mode)
    print("Reward mode", env.reward_mode)

    if args.render_goal_point and hasattr(env, 'goal_site'): env.goal_site.unhide_visual()
    obs, _ = env.reset(seed=0)
    after_reset = True

    # Viewer
    if args.enable_sapien_viewer:
        env.render_human()
    opencv_viewer = OpenCVViewer(exit_on_esc=False)

    def render_wait():
        if not args.enable_sapien_viewer:
            return
        while True:
            env.render_human()
            sapien_viewer = env.viewer
            if sapien_viewer.window.key_down("0"):
                break

    # Embodiment
    num_arms = sum("arm" in x for x in env.agent.controller.configs)
    has_gripper = any("gripper" in x for x in env.agent.controller.configs)
    gripper_action = 1
    EE_ACTION = 0.1

    # Load model
    model = PytorchInference(model_path="") # todo: real model path
    model.set_natural_instruction(instruction)

    while True:
        # -------------------------------------------------------------------------- #
        # Visualization
        # -------------------------------------------------------------------------- #
        if args.enable_sapien_viewer:
            env.render_human()

        render_frame = env.render()

        if after_reset:
            after_reset = False
            # Re-focus on opencv viewer
            if args.enable_sapien_viewer:
                opencv_viewer.close()
                opencv_viewer = OpenCVViewer(exit_on_esc=False)

        # -------------------------------------------------------------------------- #
        # Interaction
        # -------------------------------------------------------------------------- #
        # Input
        key = opencv_viewer.imshow(render_frame, delay=100)

        # Parse end-effector action
        ee_action = np.zeros([6])

        if key == None:
            model.set_observation(rgb=obs['image']['base_camera']['rgb'])
            model_output = model.inference()
            model_terminate = model_output[-1]
            if model_terminate:
                print("model has terminated!")
            delta = cal_action(env, model_output[:8],\
                               obs['camera_param']['base_camera']['extrinsic_cv'],\
                               CAL_DELTA_METHOD)
            ee_action = tuple(delta[:6])
            gripper_action = delta[-1]

        # End-effector
        if num_arms > 0:
            # Position
            if key == "i":  # +x
                ee_action[0] = EE_ACTION
            elif key == "k":  # -x
                ee_action[0] = -EE_ACTION
            elif key == "j":  # +y
                ee_action[1] = EE_ACTION
            elif key == "l":  # -y
                ee_action[1] = -EE_ACTION
            elif key == "u":  # +z
                ee_action[2] = EE_ACTION
            elif key == "o":  # -z
                ee_action[2] = -EE_ACTION

            # Rotation (axis-angle)
            if key == "1":
                ee_action[3:6] = (1, 0, 0)
            elif key == "2":
                ee_action[3:6] = (-1, 0, 0)
            elif key == "3":
                ee_action[3:6] = (0, 1, 0)
            elif key == "4":
                ee_action[3:6] = (0, -1, 0)
            elif key == "5":
                ee_action[3:6] = (0, 0, 1)
            elif key == "6":
                ee_action[3:6] = (0, 0, -1)

        # Gripper
        if has_gripper:
            if key == "f":  # open gripper
                gripper_action = 1
            elif key == "g":  # close gripper
                gripper_action = -1

        # Other functions
        if key == "0":  # switch to SAPIEN viewer
            render_wait()
        elif key == "r":  # reset env
            random_episode_idx = np.random.randint(0, TEST_NUM_EPISODES - 1)
            print(args.env_id, "reset to episode ", random_episode_idx)
            if args.render_goal_point and hasattr(env, 'goal_site'): env.goal_site.unhide_visual()
            obs, _ = env.reset(seed=random_episode_idx)
            gripper_action = 1
            after_reset = True
            # user_instruction = input("Please input the natural instruction:")
            # set_natural_instruction(user_instruction)
            continue
        elif key == 'q':  # exit
            break

        # -------------------------------------------------------------------------- #
        # Post-process action
        # -------------------------------------------------------------------------- #
        action_dict = dict(arm=ee_action, gripper=gripper_action)
        action = env.agent.controller.from_action_dict(action_dict)

        if args.render_goal_point and hasattr(env, 'goal_site'): env.goal_site.unhide_visual()
        obs, reward, terminated, truncated, info = env.step(action)
        # print("reward", reward)
        # print("terminated", terminated, "truncated", truncated)
        # print("info", info)

    env.close()


if __name__ == "__main__":
    main()
