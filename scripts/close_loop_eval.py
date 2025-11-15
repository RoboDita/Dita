import argparse
import logging
import os
os.environ["MS2_ASSET_DIR"] = "/xxx/xxx/share_data/Anonymous/maniskill2/assets" # maniskill2 assets path
import pickle
import sys
import time

import gymnasium as gym
import numpy as np
import sapien.core as sapien
import torch
import torch.nn as nn
import torchvision
from Dataset_Sim.SimDataset import process_traj_v3
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.utils.io_utils import load_json
from mani_skill2.utils.sapien_utils import look_at
from mani_skill2.utils.visualization.cv2_utils import OpenCVViewer
from mani_skill2.utils.wrappers import RecordEpisode
from moviepy.editor import ImageSequenceClip
from openvla.prismatic.util.robot_utils import cal_action, cal_action_from_pose, eef_pose
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPModel, CLIPProcessor
from transforms3d.quaternions import mat2quat, quat2mat
from moviepy.editor import ImageSequenceClip
from petrel_client.client import Client
from collections import defaultdict
import nltk
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 



extrinsics_pool = np.asarray([[[-4.4721359e-01, 8.9442706e-01,-1.4901161e-08, -4.4721358e-02], [ 6.6666663e-01, 3.3333331e-01,-6.6666663e-01, 1.3333333e-01], 
                                   [-5.9628463e-01,-2.9814237e-01,-7.4535596e-01, 6.8572754e-01], [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00,]],
                                   [[ 4.4721359e-01, 8.9442706e-01, 1.4901161e-08,  4.4721358e-02], [ 6.6666663e-01,-3.3333331e-01,-6.6666663e-01, 1.3333333e-01],
                                     [-5.9628463e-01, 2.9814237e-01,-7.4535596e-01, 6.8572754e-01],[ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00,]],
                                     [[-4.4721359e-01, 8.9442718e-01, 3.7252903e-09, -4.4721343e-02],[ 1.9518001e-01, 9.7590014e-02, -9.7590005e-01, 3.1228808e-01],
                                      [-8.7287164e-01,-4.3643576e-01,-2.1821789e-01, 4.3643579e-01],[ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00,]],
                                      [[ 3.1622776e-01, 9.4868326e-01,-7.4505806e-09,  3.1622782e-02],[ 7.0392162e-01,-2.3464054e-01,-6.7040157e-01, 1.3743240e-01],
                                       [-6.3599873e-01, 2.1199957e-01,-7.4199849e-01, 9.5399815e-01],[ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],
                                       [[-3.9391929e-01, 9.1914511e-01, 7.4505806e-09, -7.8783855e-02],[ 5.0444633e-01, 2.1619129e-01,-8.3593971e-01, 1.8448329e-01],
                                        [-7.6834989e-01,-3.2929277e-01,-5.4882127e-01, 8.1225550e-01],[ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]]])


# param
MAX_EPISODE_STEPS = 300
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
CAMERA_POOL_FILE = "/xxx/xxx/share_data/Anonymous/maniskill2/camera_pool_300k.npz" # The Camera pool discribed in the OC-VLA paper
camera_pool = np.load(CAMERA_POOL_FILE)["cameras"]


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


class PytorchInference(nn.Module):
    def __init__(self, model, stride=1, exec_steps = None, use_language_instruction = False):
        super().__init__()

        self.device = "cuda:" + str(os.environ["LOCAL_RANK"]) if torch.cuda.is_available() else "cpu"

        self.model = model
        try:
            self.use_segmentation = self.model.module.use_segmentation
        except:
            self.use_segmentation = False
        self.sequence_length = self.model.module.time_sequence_length
        self.use_wrist_img = self.model.module.use_wrist_img
        self.use_depth_img = self.model.module.use_depth_img
        try:
            self.text_max_length = self.model.module.text_max_length
        except:
            self.text_max_length = 77
        self.model.eval()
        self.model_input = []
        self.model_input_tranformed = []
        # self.img_orig = []
        self.observation = []
        self.model_input_wrist = []
        self.model_input_depth = []
        # self.wrist_observation = []
        self.instruction = ""
        self.stride = stride
        self.use_language_instruction = use_language_instruction
        self.data_transform = torchvision.transforms.Compose(
            [
                # torchvision.transforms.ToTensor(),
                # torchvision.transforms.Resize((448,448),antialias=True),
                torchvision.transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                # torchvision.transforms.Normalize(OPENAI_CLIP_MEAN, OPENAI_CLIP_STD)
            ]
        )
        self.clip_tokenizer = AutoTokenizer.from_pretrained(
            "/mnt/petrelfs/share_data/houzhi/clip-vit-large-patch14/", use_fast=False
        )
        self.clip_text_encoder = CLIPModel.from_pretrained(
            "/mnt/petrelfs/share_data/houzhi/clip-vit-large-patch14/"
        ).text_model

        self.to(self.device)
        self.frame = 0

        self.exec_steps = exec_steps
        # model_output: dx dy dz dqw dqx dqy dqz terminate

        if self.use_segmentation:
            self.segmentation_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base", local_files_only = True).to(self.device)
            self.segmentation_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base", trust_remote_code=True, local_files_only = True)

    def set_natural_instruction(self, instruction: str, seg_obj = None):
        inputs = self.clip_tokenizer(text=instruction, return_tensors="pt", max_length=self.text_max_length, padding="max_length")
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        with torch.no_grad():
            text_embeddings = self.clip_text_encoder(**inputs)[0].squeeze(0)
        self.instruction = text_embeddings
        # self.natural_instruction = preprocess_instruction(instruction)
        self.natural_instruction = None
        if seg_obj is not None:
            self.natural_instruction = seg_obj + '.'

    def set_eef_pose(self, eef_pose):
        self.eef_pose = eef_pose

    def set_observation(self, rgb, depth=None, wrist=None):
        assert (rgb >= 0).all() and (rgb <= 255).all()
        self.observation.append(rgb)
        # import ipdb;ipdb.set_trace()
        # self.img_orig.append(torch.tensor(rgb).to(self.device, non_blocking=True))
        if self.model_input == []:

            rgb = torch.tensor(rgb).to(self.device, non_blocking=True)
            if not self.use_language_instruction:
                rgb_data = self.data_transform((rgb / 255.0).permute(2, 0, 1).contiguous())
            else:
                rgb_data = rgb
                rgb_data_transformed = self.data_transform((rgb / 255.0).permute(2, 0, 1).contiguous())
            self.model_input = rgb_data.unsqueeze(0)
            if self.use_language_instruction:
                self.model_input_tranformed = rgb_data_transformed.unsqueeze(0)
        else:

            rgb = torch.tensor(rgb).to(self.device, non_blocking=True)
            if not self.use_language_instruction:
                rgb_data = self.data_transform((rgb / 255.0).permute(2, 0, 1).contiguous())
            else:
                rgb_data = rgb
                rgb_data_transformed = self.data_transform((rgb / 255.0).permute(2, 0, 1).contiguous())
            self.model_input = torch.cat((self.model_input, rgb_data.unsqueeze(0)), dim=0)
            self.model_input = self.model_input[-self.sequence_length :]

            if self.use_language_instruction:
                self.model_input_tranformed = torch.cat((self.model_input_tranformed, rgb_data_transformed.unsqueeze(0)), dim = 0)
                self.model_input_tranformed = self.model_input_tranformed[-self.sequence_length:]

        if wrist is not None and self.use_wrist_img:
            if self.model_input_wrist == []:

                wrist_data = torch.tensor(wrist).to(self.device, non_blocking=True)

                wrist_data = self.data_transform((wrist_data / 255.0).permute(2, 0, 1).contiguous())

                self.model_input_wrist = wrist_data.unsqueeze(0)
            else:

                wrist_data = torch.tensor(wrist).to(self.device, non_blocking=True)

                wrist_data = self.data_transform((wrist_data / 255.0).permute(2, 0, 1).contiguous())

                self.model_input_wrist = torch.cat((self.model_input_wrist, wrist_data.unsqueeze(0)), dim=0)
                self.model_input_wrist = self.model_input_wrist[-self.sequence_length :]
            wrist = (
                nn.functional.interpolate(torch.tensor(wrist).permute(2, 0, 1).unsqueeze(0), size=(224, 224), mode="nearest")
                .squeeze()
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            self.observation[-1] = np.concatenate([self.observation[-1], wrist], axis=1)
        if depth is not None and self.use_depth_img:
            if self.model_input_depth == []:

                depth_data = torch.tensor(depth / 10).to(self.device, non_blocking=True)
                self.model_input_depth = depth_data.unsqueeze(0)
            else:
                depth_data = torch.tensor(depth / 10).to(self.device, non_blocking=True)
                self.model_input_depth = torch.cat((self.model_input_depth, depth_data.unsqueeze(0)), dim=0)
                self.model_input_depth = self.model_input_depth[-self.sequence_length :]
            depth = torch.tensor(depth / 10 * 255).repeat(1, 1, 3).byte().cpu().numpy()
            self.observation[-1] = np.concatenate([self.observation[-1], depth], axis=1)

    def reset_observation(self):
        self.model_input = []
        self.model_input_tranformed = []
        self.observation = []
        self.model_input_wrist = []
        self.model_input_depth = []
        # self.wrist_observation = []
        # self.img_orig = []
        self.frame = 0

    def save_video(self, fpath):
        
        clip = ImageSequenceClip(self.observation, fps=10 / self.stride)
        clip.write_videofile(fpath, codec="libx264", audio=False, logger=None) 


    def calc_act(self, base_episode, camera_extrinsic_cv, current_frame_idx):
        try:
            pose1 = torch.tensor(base_episode["step"][current_frame_idx]["prev_ee_pose"]).clone()
            # pose2 = torch.tensor(base_episode["step"][current_frame_idx]["target_ee_pose"]).clone()
            pose2 = torch.tensor(base_episode["step"][current_frame_idx + self.stride]["prev_ee_pose"]).clone()
        except:
            current_frame_idx = min(current_frame_idx, len(base_episode["step"]) - 1)
            pose1 = torch.tensor(base_episode["step"][current_frame_idx]["prev_ee_pose"]).clone()
            # pose2 = torch.tensor(base_episode["step"][current_frame_idx]["target_ee_pose"]).clone()
            pose2 = torch.tensor(base_episode["step"][-1]["prev_ee_pose"]).clone()

        pose1[0] -= 0.615  # base to world
        pose2[0] -= 0.615  # base to world
        action = {}
        action["world_vector"], action["rotation_delta"] = process_traj_v3(
            (camera_extrinsic_cv),
            pose1,
            pose2,
        )

        if base_episode["step"][current_frame_idx]["is_terminal"] == True:
            action["terminate_episode"] = torch.tensor([1, 0, 0], dtype=torch.int32)
        else:
            action["terminate_episode"] = torch.tensor([0, 1, 0], dtype=torch.int32)
        action["gripper_closedness_action"] = torch.tensor(
            base_episode["step"][current_frame_idx]["action"][-1],
            dtype=torch.float32,
        ).unsqueeze(-1)

        return action

    def get_target_pose(self, delta_pos, delta_rot):
        target_ee_pose_at_camera = sapien.Pose(p=self.eef_pose.p + delta_pos)
        r_prev = quat2mat(self.eef_pose.q)
        r_diff = quat2mat(delta_rot)
        r_target = r_diff @ r_prev
        target_ee_pose_at_camera.set_q(mat2quat(r_target))

        return target_ee_pose_at_camera

    def inference(self, extrinsics=None):

        obs = {"image": self.model_input[-self.sequence_length :].unsqueeze(0)}
        if self.use_wrist_img:
            obs["wrist_image"] = self.model_input_wrist[-self.sequence_length :].unsqueeze(0)
        if self.use_depth_img:
            obs["depth_image"] = self.model_input_depth[-self.sequence_length :].unsqueeze(0)
        if self.use_language_instruction:
            obs["image_transformed"] = self.model_input_tranformed[-self.sequence_length:].unsqueeze(0)
        obs["natural_language_embedding"] = self.instruction[None, None, ...].repeat(1, obs["image"].shape[1], 1, 1)
        obs["natural_language_instruction"] = [self.natural_instruction] * obs["image"].shape[1]
        if self.use_segmentation:

            # import ipdb;ipdb.set_trace()
            seg_img = torch.tensor(self.observation[-self.sequence_length:])

            seg_inputs = self.segmentation_processor(
                # images = obs["image"].reshape(-1, *obs['image'].shape[2:]), 
                images = seg_img,
                text = obs["natural_language_instruction"], 
                return_tensors = 'pt'
            ).to(self.device)
            with torch.no_grad():
                seg_outputs = self.segmentation_model(**seg_inputs)
            
            seg_result = self.segmentation_processor.post_process_grounded_object_detection(
                seg_outputs,
                seg_inputs.input_ids,
                box_threshold=0.4,
                text_threshold=0.3,
                target_sizes=[(obs['image'].shape[-2], obs['image'].shape[-3])] * obs["image"].shape[1]
            )
            
            boxes = [result['boxes'][0,:].tolist() if result['boxes'].shape[0]!=0 else torch.tensor([0,0,0,0], device = self.device).tolist()  for result in seg_result]
            boxes = torch.tensor(boxes, dtype = torch.int32, device = self.device).unsqueeze(0)
            obs['segmentation'] = boxes

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                # 1 x T x L x C
                if self.exec_steps is None:
                    model_output = self.model.module.inference(obs)
                else:
                    model_output = self.model.module.inference(obs, exec_steps = self.exec_steps)

        # 1 x L x C
        if model_output.dim() == 4:
            model_output = model_output.flatten(0,1)
        detokenize_output, _, _ = self.model.module.action_tokenizer.detokenize(model_output)
        detokenize_output["rotation_delta"][:,0] += 1
        outputs = []
        # import pdb;pdb.set_trace()
        for i in range(model_output.shape[0]):
            # target_pose = self.get_target_pose(detokenize_output["world_vector"][i].cpu().numpy(), detokenize_output["rotation_delta"][i].cpu().numpy())

            output = torch.cat(
                [
                    # torch.tensor(target_pose.p),
                    # torch.tensor(target_pose.q),
                    detokenize_output["world_vector"][i].cpu(),
                    detokenize_output["rotation_delta"][i].cpu(),
                    detokenize_output["gripper_closedness_action"][i].cpu(),
                    detokenize_output["terminate_episode"][i][[0]].cpu(),
                ]
            )

            # add 1 to quat
            output[-2] = (output[-2] > 0.0).float() * 2 - 1
            output[-1] = (output[-1] > 0.5).float()
            # self.frame += self.stride
            outputs.append(output)
        
        output = torch.stack(outputs, dim = 0)

        # import pdb;pdb.set_trace()
        return output.cpu().numpy()



def analyze_traj_str(traj_str):

    env_id = traj_str.split("-")[0] + "-v0"
    select_camera = f"camera_{traj_str[-5]}"
    seed_start_pos = traj_str.find("traj") + 5
    seed = int(traj_str[seed_start_pos:-13])
    return env_id, select_camera, seed


def close_loop_eval_v2(
    obs_mode="rgbd",
    reward_mode=None,
    control_mode=TARGET_CONTROL_MODE,
    render_mode="cameras",
    record_dir=None,
    render_goal_point=True,
    test_episodes_num=100,
    model=None,
    eval_data_list=None,
    args=None,
    rand_seed=0,
    json_repo="/xxx/xxx/share_data/Anonymous/maniskill2/demos/v0/rigid_body/", # maniskill2 official files
    camera_coord=True,
    stride=1,
    root_folder = None,
    data_root_path = None,
    exec_steps = None,
    use_language_instruction = False,
    temp_fix = False,
):
    
    client = Client()
    np.set_printoptions(suppress=True, precision=3)
    eval_traj_list = pickle.load(open(eval_data_list, "rb"))
    np.random.seed(0 % 9973)
    eval_traj_index = np.random.permutation(len(eval_traj_list))[: min(args.world_size * test_episodes_num, len(eval_traj_list))]
    eval_traj_index = eval_traj_index[args.rank * test_episodes_num : min( (args.rank + 1) * test_episodes_num, len(eval_traj_list))]
    if len(eval_traj_index) == 0 :
        eval_traj_index = []
        eval_traj_index.append(0)
    eval_traj_index = sorted(eval_traj_index)
    

    success_num = {"PickCube-v0" : 0.0, "PickSingleYCB-v0" : 0.0, "StackCube-v0" : 0.0, "PickClutterYCB-v0": 0.0,
                   "AssemblingKits-v0" : 0.0, "PegInsertionSide-v0": 0.0, "PickSingleEGAD-v0": 0.0}
    success_list = []
    i = 0
    model = PytorchInference(model=model, stride=stride, exec_steps = exec_steps, use_language_instruction = use_language_instruction)
    traj_str = eval_traj_list[eval_traj_index[i]]
    if 'PickClutterYCB-v0' in traj_str:
        data_root_path = data_root_path.replace('camerabase1','camerabase1')
    else:
        data_root_path = data_root_path.replace('camerabase4','camerabase1')
    env_id, select_camera, seed, reset_kwargs, instruction, seg_obj = analyze_traj_str_v2(client, traj_str, json_repo, data_root_path)
    # env_id = 'PickCube-v0'; select_camera = 1; seed = 982; reset_kwargs = {}
  
      
    if temp_fix:
        data_temp = pickle.loads(client.get(os.path.join(data_root_path, traj_str)))
        for ii in range(5):
            if (extrinsics_pool[ii].astype(np.float32) == data_temp['step'][0]['camera_extrinsic_cv']).all():
                select_camera = ii + 1
                break
        camera_pose = CAMERA_POSES["camera_{}".format(select_camera)]

    else:

        camera_pose = look_at(camera_pool[select_camera][:3], camera_pool[select_camera][3:6], camera_pool[select_camera][6:9])
    # import pdb;pdb.set_trace()
    env: BaseEnv = gym.make(
        env_id,
        renderer_kwargs={"offscreen_only": True, "device": f"cuda:{args.local_rank}"},
        obs_mode=obs_mode,
        reward_mode=reward_mode,
        control_mode=control_mode,
        render_mode=render_mode,
        render_camera_cfgs=dict(width=2 * CAMERA_W, height=2 * CAMERA_H),
        camera_cfgs=dict(
            base_camera=dict(p=camera_pose.p, q=camera_pose.q, width=CAMERA_W, height=CAMERA_H),
            hand_camera=dict(width=128, height=128),
        ),
        max_episode_steps=MAX_EPISODE_STEPS * 100,
    )

    if record_dir != None:
        record_dir = record_dir.format(env_id=env_id)
        env = RecordEpisode(env, record_dir, render_mode=render_mode)
    if render_goal_point and hasattr(env, "goal_site"):
        env.goal_site.unhide_visual()

    obs, _ = env.reset(seed=seed, options=reset_kwargs)

    model.set_natural_instruction(instruction, seg_obj)
    # total_num = 0
    # start_time = time.time()

    if root_folder != None:
        os.makedirs(root_folder, exist_ok = True) 

    if exec_steps is None:
        exec_steps = 1

    model.set_eef_pose(eef_pose(env, obs["camera_param"]["base_camera"]["extrinsic_cv"], camera_coord=camera_coord))
    model.set_observation(rgb=obs["image"]["base_camera"]["rgb"], wrist=obs["image"]["hand_camera"]["rgb"])

    # step_cur = 0
    while i < test_episodes_num and i < len(eval_traj_index) :
        # total_num += 1
        # if total_num >500:
        #     break
       # if 'DEBUG' in os.environ: import ipdb;ipdb.set_trace()
        
        model_output_steps = model.inference(obs["camera_param"]["base_camera"]["extrinsic_cv"])
        # model_terminate = model_output[-1]
        
        for exec_i in range(exec_steps):
            model_output = model_output_steps[exec_i]
            target_pose = model.get_target_pose(model_output[:3], model_output[3:7])
            model_output[:3] = target_pose.p
            model_output[3:7] = target_pose.q
            
            delta, loop = np.array([1, 1, 1, 1, 1, 1, 1], dtype=float), 8
            while np.max(np.abs(delta[:3])) > 1e-4 and loop > 0:
                loop -= 1
                delta = cal_action_from_pose(env, model_output[:8], obs["camera_param"]["base_camera"]["extrinsic_cv"], camera_coord=camera_coord)
                ee_action = tuple(delta[:6])
                gripper_action = delta[-1]
                action_dict = dict(arm=ee_action, gripper=gripper_action)
                action = env.agent.controller.from_action_dict(action_dict)

                if render_goal_point and hasattr(env, "goal_site"):
                    env.goal_site.unhide_visual()
                obs, reward, terminated, truncated, info = env.step(action)

            model.frame += model.stride
            truncated = model.frame >= MAX_EPISODE_STEPS
            if terminated or truncated:
                success_list.append(info["success"])
                success_num[env_id] += info["success"]
                print(i, traj_str, info["success"], flush = True)
                if root_folder != None:
                    model.save_video(os.path.join(root_folder, f'{(i+args.rank * test_episodes_num):04d}_{instruction}_{info["success"]}.mp4'))
                i += 1
                if i >= test_episodes_num or i >= len(eval_traj_index):
                    break
                traj_str = eval_traj_list[eval_traj_index[i]]
                if 'PickClutterYCB-v0' in traj_str:
                    data_root_path = data_root_path.replace('camerabase1','camerabase1')
                else:
                    data_root_path = data_root_path.replace('camerabase4','camerabase1')
                env_id_new, select_camera_new, seed_new, reset_kwargs_new, instruction_new, seg_obj = analyze_traj_str_v2(client, traj_str, json_repo, data_root_path)
                if env_id != env_id_new or select_camera != select_camera_new or instruction_new != instruction or reset_kwargs_new != reset_kwargs:
                    env_id = env_id_new
                    select_camera = select_camera_new
                    seed = seed_new
                    reset_kwargs = reset_kwargs_new
                    instruction = instruction_new

                    if temp_fix:
                        data_temp = pickle.loads(client.get(os.path.join(data_root_path, traj_str)))
                        for ii in range(5):
                            if (extrinsics_pool[ii].astype(np.float32) == data_temp['step'][0]['camera_extrinsic_cv']).all():
                                select_camera = ii + 1
                                break
                        camera_pose = CAMERA_POSES["camera_{}".format(select_camera)]

                    else:
                        camera_pose = look_at(camera_pool[select_camera][:3], camera_pool[select_camera][3:6], camera_pool[select_camera][6:9])



                        
                    env: BaseEnv = gym.make(
                        env_id,
                        renderer_kwargs={"offscreen_only": True, "device": f"cuda:{args.local_rank}"},
                        obs_mode=obs_mode,
                        reward_mode=reward_mode,
                        control_mode=control_mode,
                        render_mode=render_mode,
                        render_camera_cfgs=dict(width=2 * CAMERA_W, height=2 * CAMERA_H),
                        camera_cfgs=dict(
                            base_camera=dict(p=camera_pose.p, q=camera_pose.q, width=CAMERA_W, height=CAMERA_H),
                            hand_camera=dict(width=128, height=128),
                        ),
                        max_episode_steps=MAX_EPISODE_STEPS * 100,
                    )
                    if record_dir != None:
                        record_dir = record_dir.format(env_id=env_id)
                        env = RecordEpisode(env, record_dir, render_mode=render_mode)
                    # instruction = NATURAL_INSTRUCTIONS[env_id]
                    model.set_natural_instruction(instruction, seg_obj)
                env_id = env_id_new
                select_camera = select_camera_new
                seed = seed_new
                reset_kwargs = reset_kwargs_new
                if render_goal_point and hasattr(env, "goal_site"):
                    env.goal_site.unhide_visual()
                obs, _ = env.reset(seed=seed, options=reset_kwargs)
                model.reset_observation()
                model.set_eef_pose(eef_pose(env, obs["camera_param"]["base_camera"]["extrinsic_cv"], camera_coord=camera_coord))
                model.set_observation(rgb=obs["image"]["base_camera"]["rgb"], wrist=obs["image"]["hand_camera"]["rgb"])
                break
            model.set_eef_pose(eef_pose(env, obs["camera_param"]["base_camera"]["extrinsic_cv"], camera_coord=camera_coord))
            model.set_observation(rgb=obs["image"]["base_camera"]["rgb"], wrist=obs["image"]["hand_camera"]["rgb"])
        if i >= test_episodes_num or i >= len(eval_traj_index):
            break

        
    
    total_success_rate = np.mean(success_list)
    env.close()
    del env
    
    print(success_num, total_success_rate, flush = True)
    
    return success_num, total_success_rate







def analyze_traj_str_v2(client, traj_str, json_repo, data_root_path):

    env_id = traj_str.split("/")[0]
    
    data = pickle.loads(client.get(os.path.join(data_root_path, traj_str)))
    select_camera = data["camera_index_in_pool"]
    json_root_path = os.path.join(json_repo, env_id)

    
    if env_id != "PickSingleYCB-v0":
        json_data = load_json(os.path.join(json_root_path, "trajectory.json"))
    elif env_id == "PickSingleYCB-v0":
        pkl_str = traj_str.split('/')[1]
        task_name_start_pos = len(env_id) + 1
        task_name_end_pos = pkl_str.find('_traj')
        task_name = pkl_str[task_name_start_pos:task_name_end_pos]
        json_data = load_json(os.path.join(json_root_path, task_name + '.json'))

    traj_start_pos = traj_str.find("traj")
    index = int(traj_str[traj_start_pos:].split("_")[1])
    reset_kwargs = json_data["episodes"][index]["reset_kwargs"]
    if "seed" in reset_kwargs:
        reset_kwargs["seed"] = json_data["episodes"][index]["episode_seed"]
    seed = reset_kwargs.pop("seed")
   
    instruction = data["step"][0]["observation"]["natural_instruction"]
    if 'segmentation' in data['step'][0]['observation']:
        seg = data["step"][0]["observation"]["segmentation"]
        seg_obj = None
        for k in seg.keys():
            if k in instruction and k != 'robot' and k!='site':
                seg_obj = k
                if k == '':
                    seg_obj = 'object'
                if "StackCube-v0" in traj_str:
                    seg_obj == 'red cube'
                break
        
        if seg_obj is None:
            if env_id == "PickCube-v0":
                seg_obj = 'cube'
            elif env_id == "StackCube-v0":
                seg_obj == 'red cube'
            elif env_id == "PickSingleEGAD":
                seg_obj = 'object'
            else:
                pos_end = instruction.find('and')
                pos_start = len('pick up the ')
                seg_obj = instruction[pos_start:pos_end - 1]
                
        return env_id, select_camera, seed, reset_kwargs, instruction, seg_obj   
    else:
        return env_id, select_camera, seed, reset_kwargs, instruction, None
