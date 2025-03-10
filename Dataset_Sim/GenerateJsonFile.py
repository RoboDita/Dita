# import a s3 package as Client
import pickle
import pdb
import numpy as np
import io
import json
import os
from collections import defaultdict
import torch
from SimDataset import process_traj_v3
import time

def main(task_name: str, stride: int, output_path: str):

    if not os.path.exists(os.path.join(output_path, task_name)):
        os.makedirs(os.path.join(output_path, task_name))
    client = Client()
    root_url = "s3://Anonymous/Sim_Data/ManiSkill2-0503/"
    task_url = os.path.join(root_url, task_name)
    contents = client.list(task_url)
    current_data_idx = 0
    last_time = time.time()
    for ii, content in enumerate(contents):
        if content.endswith('camera_0/'):
            continue
        cam_data_repo = os.path.join(task_url, content)
        base_data_repo = os.path.join(task_url, content[:content.find('camera')] + 'camera_0/')

        cam_metadata_path = os.path.join(cam_data_repo, 'data.pkl')
        base_metadata_path = os.path.join(base_data_repo, 'data.pkl')
        cam_metadata = pickle.loads(client.get(cam_metadata_path))
        base_metadata = pickle.loads(client.get(base_metadata_path))


        episode_length = len(base_metadata['step'])
        for i in range(episode_length):
            
            json_dict = {}
            json_dict['id'] = current_data_idx
            json_dict['image'] = os.path.join(cam_data_repo, cam_metadata['step'][i]['observation']['image']).replace('', "")
            json_dict['conversations'] = []
            camera_extrinsic_cv = torch.tensor(cam_metadata["step"][0]["camera_extrinsic_cv"])
            json_dict['extrinsics'] = camera_extrinsic_cv.tolist()

            current_idx = i
            next_idx = min(i + stride, episode_length - 1)
            pose1 = torch.tensor(base_metadata['step'][current_idx]['prev_ee_pose']).clone()
            pose2 = torch.tensor(base_metadata['step'][next_idx]['prev_ee_pose']).clone()
            pose1[0] -= 0.615
            pose2[0] -= 0.615
            world_vector, rotation_delta = process_traj_v3(
                camera_extrinsic_cv, pose1, pose2
            )
            world_vector = world_vector.tolist()
            rotation_delta = rotation_delta.tolist()
            rotation_delta[0] -= 1.0
            terminate_episode = int( not cam_metadata['step'][current_idx]['is_terminal'])
            gripper = cam_metadata['step'][current_idx]['action'][-1]

            human_conv = {}
            human_conv['from'] = "human"
            human_conv['value'] = '<image>\n' + 'What should the robot do to '+ cam_metadata['step'][current_idx]['observation']['natural_instruction'] + '?'
            gpt_conv = {}
            gpt_conv['from'] = 'gpt'
            # gpt_conv['value'] = str(world_vector[0]) + ' ' + str(world_vector[1]) + ' ' \
            #     + str(world_vector[2]) + ' ' + str(rotation_delta[0]) + ' '+ str(rotation_delta[1]) \
            #     + ' ' + str(rotation_delta[2]) + ' ' + str(rotation_delta[3]) + ' ' \
            #     + str(gripper) + ' ' + str(terminate_episode)
            gpt_conv['value'] = (
                f"{world_vector[0]:.10f} {world_vector[1]:.10f} {world_vector[2]:.10f} {rotation_delta[0]:.10f} {rotation_delta[1]:.10f} "
                f"{rotation_delta[2]:.10f} {rotation_delta[3]:.10f} "
                f"{gripper} " 
                f"{terminate_episode}"
            )
            json_dict['conversations'].append(human_conv)
            json_dict['conversations'].append(gpt_conv)

            json_path = os.path.join(output_path, task_name, f"{task_name}_{current_data_idx // 200000}.jsonl")
            with open(json_path, 'a') as file:
                json_line = json.dumps(json_dict)
                file.write(json_line + '\n')
        
            current_data_idx += 1
        if ii % 100 == 0:
            now_time = time.time()
            print(f"{ii} episodes done!, use time: {now_time - last_time}s", flush = True) 


if __name__ == "__main__":
    
    main(
        task_name = 'PickCube-v0', 
        stride = 4, 
        output_path = '/xxx/xxx/share_data/Anonymous/Dataset/Sim_Data_json',
    )

    