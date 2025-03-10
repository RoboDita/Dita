import pickle
import os
import random

base_cam_list_url = "/xxx/xxx/share_data/Anonymous/Dataset/Sim_Data/FilePath/s3_simdata_base_cam_list.pkl"
data_cam_list_url = "/xxx/xxx/share_data/Anonymous/Dataset/Sim_Data/FilePath/s3_simdata_data_cam_list.pkl"

with open(base_cam_list_url, 'rb') as f:
    base_cam_list = pickle.load(f)

with open(data_cam_list_url, 'rb') as f:
    data_cam_list = pickle.load(f)

base_cam_list = sorted(base_cam_list)
data_cam_list = sorted(data_cam_list)

#for fix cam
train_data = []
eval_data = []
for i in range(len(data_cam_list)):
    if i % 5 == 4:
        eval_data.append(data_cam_list[i])
    else:
        train_data.append(data_cam_list[i])

import pdb
pdb.set_trace()

save_path = '/xxx/xxx/share_data/Anonymous/Dataset/Sim_Data/FilePath'
with open(os.path.join(save_path, 'fix_cam_train_data.pkl'), 'wb') as f:
    pickle.dump(train_data, f)
with open(os.path.join(save_path, 'fix_cam_eval_data.pkl'), 'wb') as f:
    pickle.dump(eval_data, f)

#for fix traj
train_data = []
eval_data = []
random.seed(0)
random.shuffle(base_cam_list)
eval_traj_length = len(base_cam_list) // 10
for i in range(len(base_cam_list)):
    
    for j in range(5):
        if i < eval_traj_length:
            eval_data.append(base_cam_list[i].replace('camera_0', f'camera_{j+1}'))
        else:
            train_data.append(base_cam_list[i].replace('camera_0', f'camera_{j+1}'))

import pdb
pdb.set_trace()

with open(os.path.join(save_path, 'fix_traj_train_data.pkl'), 'wb') as f:
    pickle.dump(train_data, f)
with open(os.path.join(save_path, 'fix_traj_eval_data.pkl'), 'wb') as f:
    pickle.dump(eval_data, f)