import os
import pickle
import numpy as np

old_data_list_path = '/xxx/xxx/share_data/Anonymous/20240920/'
data_list = os.listdir(old_data_list_path)
new_data_list_path = '/xxx/xxx/share_data/Anonymous/Dataset/LabData__'
if not os.path.exists(new_data_list_path):
    os.makedirs(new_data_list_path)

data_num = len(data_list)

for i in range(data_num):

    data_path = os.path.join(old_data_list_path, data_list[i])
    data = pickle.load(open(data_path, 'rb'))
    episode_length = len(data['steps'])
    for j in range(episode_length):
        gripper = data['steps'][j]['observation']["gripper_position"]
        if gripper != 0:
            left_non_zero = j
            break
    
    for j in range(episode_length - 1, -1, -1):
        gripper = data['steps'][j]['observation']["gripper_position"]
        if gripper != 0:
            right_non_zero = j
            break
    
    for j in range(episode_length):
        gripper = data['steps'][j]['observation']["gripper_position"]
        if gripper == 0 and j > left_non_zero and j < right_non_zero:
            for k in range(j - 1, -1, -1):
                data1 = data['steps'][k]['observation']["gripper_position"]
                if data1 != 0:
                    break
            for k in range(j+1, episode_length):
                data2 = data['steps'][k]['observation']["gripper_position"]
                if data2 != 0:
                    break
            data['steps'][j]['observation']["gripper_position"] = (data1 + data2) / 2
            print(f"change traj {data_list[i]}, pos {j}, from {gripper} to {data['steps'][j]['observation']['gripper_position']}", flush = True)
    
    save_path = os.path.join(new_data_list_path, data_list[i])
    pickle.dump(data, open(save_path,'wb'))
    print(f"{i} finished!", flush = True)