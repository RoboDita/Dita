import os
import os.path as osp
import numpy as np
import h5py
import json
from petrel_client.client import Client

from mani_skill2.utils.io_utils import load_json


CEPH_SAVE_BUCKET = os.getenv("CEPH_SAVE_BUCKET", "s3://your default save path")

NUM_CAMERAS = 20

TRAIN_CAMERA_RATIO = 0.95
TRAIN_TRAJECTORY_RATIO = 0.95

ENV_NAMES = ["PickCube-v0", "StackCube-v0", "PickSingleYCB-v0", "PegInsertionSide-v0", "AssemblingKits-v0", "PickSingleEGAD-v0", "PickClutterYCB-v0"]


def process_env(client, env_id):
    traj_path = f"demos/v0/rigid_body/{env_id}/trajectory.h5"
    h5_file = h5py.File(traj_path, "r")
    json_path = traj_path.replace(".h5", ".json")
    json_data = load_json(json_path)
    traj_size = len(json_data["episodes"])
    train_traj_range = (0, int(traj_size * TRAIN_TRAJECTORY_RATIO))
    val_traj_range = (train_traj_range[1], traj_size)
    
    outputs = {}
    for label, bound in [("train", train_traj_range), ("val", val_traj_range)]:
        exist_list = []
        partial_list = []
        fail_list = []
        ranges = np.arange(bound[0], bound[1])
        num_traj = len(ranges)
        for i in range(num_traj):
            print(f"Processing {env_id} {label} {i}/{num_traj}")
            episode_idx = ranges[i]
            ep = json_data["episodes"][episode_idx]
            episode_id = ep["episode_id"]
            traj = h5_file[f"traj_{episode_id}"]
            num_steps = len(traj["actions"][:])
            num_camera_exist = 0
            
            for camera_idx in range(NUM_CAMERAS+1):
                file_name_base = env_id + "_traj_" + str(episode_idx)
                file_name = file_name_base + "_camera_" + str(camera_idx)
                full_data = True
                
                # for i_step in range(num_steps):
                #     data_path = osp.join(CEPH_SAVE_BUCKET, env_id, file_name, f"image_{i_step}.npz")
                #     if not client.contains(data_path):
                #         full_data = False
                #         break
                #     data_path = osp.join(CEPH_SAVE_BUCKET, env_id, file_name, f"depth_{i_step}.npz")
                #     if not client.contains(data_path):
                #         full_data = False
                #         break
                data_path = osp.join(CEPH_SAVE_BUCKET, env_id, file_name, "data.pkl")
                if not client.contains(data_path):
                    full_data = False
                
                if full_data:
                    num_camera_exist += 1
            
            if num_camera_exist == NUM_CAMERAS+1:
                exist_list.append(file_name_base)
            elif num_camera_exist > 0:
                partial_list.append(file_name_base)
            elif num_camera_exist == 0:
                fail_list.append(file_name_base)
            else:
                raise ValueError("Invalid number of cameras")
        
        outputs[label] = (exist_list, partial_list, fail_list)
    
    return outputs


def process_picksingleycb(client):
    env_id = "PickSingleYCB-v0"
    traj_dir = f"demos/v0/rigid_body/{env_id}"
    traj_fns = sorted([fn.split(".")[0] for fn in os.listdir(traj_dir) if fn.endswith(".h5")])
    
    all_outputs = {
        "train": ([], [], []),
        "val": ([], [], [])
    }
    
    for fn_idx, fn in enumerate(traj_fns):
        print(f"Processing {env_id} {fn_idx}/{len(traj_fns)}")
        traj_path = osp.join(traj_dir, fn + ".h5")
        h5_file = h5py.File(traj_path, "r")
        json_path = traj_path.replace(".h5", ".json")
        json_data = load_json(json_path)
        traj_size = len(json_data["episodes"])
        train_traj_range = (0, int(traj_size * TRAIN_TRAJECTORY_RATIO))
        val_traj_range = (train_traj_range[1], traj_size)
        
        outputs = {}
        for label, bound in [("train", train_traj_range), ("val", val_traj_range)]:
            exist_list = []
            partial_list = []
            fail_list = []
            ranges = np.arange(bound[0], bound[1])
            num_traj = len(ranges)
            for i in range(num_traj):
                print(f"Processing {env_id} {fn} {label} {i}/{num_traj}")
                episode_idx = ranges[i]
                ep = json_data["episodes"][episode_idx]
                episode_id = ep["episode_id"]
                traj = h5_file[f"traj_{episode_id}"]
                num_steps = len(traj["actions"][:])
                num_camera_exist = 0
                
                for camera_idx in range(NUM_CAMERAS+1):
                    file_name_base = env_id + "_" + fn + "_traj_" + str(episode_idx)
                    file_name = file_name_base + "_camera_" + str(camera_idx)
                    full_data = True
                    
                    # for i_step in range(num_steps):
                    #     data_path = osp.join(CEPH_SAVE_BUCKET, env_id, file_name, f"image_{i_step}.npz")
                    #     if not client.contains(data_path):
                    #         full_data = False
                    #         break
                    #     data_path = osp.join(CEPH_SAVE_BUCKET, env_id, file_name, f"depth_{i_step}.npz")
                    #     if not client.contains(data_path):
                    #         full_data = False
                    #         break
                    data_path = osp.join(CEPH_SAVE_BUCKET, env_id, file_name, "data.pkl")
                    if not client.contains(data_path):
                        full_data = False
                    
                    if full_data:
                        num_camera_exist += 1
                
                if num_camera_exist == NUM_CAMERAS+1:
                    exist_list.append(file_name_base)
                elif num_camera_exist > 0:
                    partial_list.append(file_name_base)
                elif num_camera_exist == 0:
                    fail_list.append(file_name_base)
                else:
                    raise ValueError("Invalid number of cameras")
            
            outputs[label] = (exist_list, partial_list, fail_list)
        
        for label in ["train", "val"]:
            all_outputs[label][0].extend(outputs[label][0])
            all_outputs[label][1].extend(outputs[label][1])
            all_outputs[label][2].extend(outputs[label][2])
    
    return all_outputs


if __name__ == "__main__":
    client = Client()
    
    data_info_json = {}
    
    for env_id in ENV_NAMES:
        print(f"Processing {env_id}")
        if env_id == "PickSingleYCB-v0":
            output = process_picksingleycb(client)
        else:
            output = process_env(client, env_id)
        
        train_exist_file_names, train_partial_file_names, train_fail_file_names = output["train"]
        val_exist_file_names, val_partial_file_names, val_fail_file_names = output["val"]
        
        data_info_json[env_id] = {
            "train": {
                "exist": train_exist_file_names,
                "partial": train_partial_file_names,
                "fail": train_fail_file_names
            },
            "val": {
                "exist": val_exist_file_names,
                "partial": val_partial_file_names,
                "fail": val_fail_file_names
            }
        }
    
    for env_id in ENV_NAMES:
        print("Env: ", env_id)
        print("     Train:")
        print("             Exist: ", len(data_info_json[env_id]["train"]["exist"]))
        print("             Partial: ", len(data_info_json[env_id]["train"]["partial"]))
        print("             Fail: ", len(data_info_json[env_id]["train"]["fail"]))
        print("     Val:")
        print("             Exist: ", len(data_info_json[env_id]["val"]["exist"]))
        print("             Partial: ", len(data_info_json[env_id]["val"]["partial"]))
        print("             Fail: ", len(data_info_json[env_id]["val"]["fail"]))
        print()
    
    with open("data_info.json", "w") as f:
        json.dump(data_info_json, f, indent=4)
    
    print("All Done!")

