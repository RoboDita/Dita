"""
traj_transforms.py

Contains trajectory transforms used in the orca data pipeline. Trajectory transforms operate on a dictionary
that represents a single trajectory, meaning each tensor has the same leading dimension (the trajectory length).
"""

import logging
from typing import Dict

import tensorflow as tf


def rotation_matrix_x(angles):  
    """Create a batch of rotation matrices around the x-axis."""  
    c, s = tf.cos(angles), tf.sin(angles)  
    zeros = tf.zeros_like(angles)  
    ones = tf.ones_like(angles)  
    tmp = tf.stack([  
        ones, zeros, zeros, zeros,  
        zeros, c, -s, zeros,  
        zeros, s, c, zeros,  
        zeros, zeros, zeros, ones  
    ], axis=-1)
    return tf.reshape(tmp, [-1, 4, 4]) 
  
def rotation_matrix_y(angles):  
    """Create a batch of rotation matrices around the y-axis."""  
    c, s = tf.cos(angles), tf.sin(angles)  
    zeros = tf.zeros_like(angles)  
    ones = tf.ones_like(angles)  
    tmp = tf.stack([  
        c, zeros, s, zeros,  
        zeros, ones, zeros, zeros,  
        -s, zeros, c, zeros,  
        zeros, zeros, zeros, ones  
    ], axis=-1)
    return tf.reshape(tmp, [-1, 4, 4]) 
  
def rotation_matrix_z(angles):  
    """Create a batch of rotation matrices around the z-axis."""  
    c, s = tf.cos(angles), tf.sin(angles)  
    zeros = tf.zeros_like(angles)  
    ones = tf.ones_like(angles)  
    tmp = tf.stack([  
        c, -s, zeros, zeros,  
        s, c, zeros, zeros,  
        zeros, zeros, ones, zeros,  
        zeros, zeros, zeros, ones  
    ], axis=-1)
    return tf.reshape(tmp, [-1, 4, 4])


def euler_to_matrix_xyz(roll, pitch, yaw, batch_size=None):  
    """Convert Euler angles (roll, pitch, yaw in XYZ order) and translation to a batch of 4x4 transformation matrices."""  
    if batch_size is not None:  
        # Expand Euler angles and translation to match the batch size  
        roll = tf.tile(tf.expand_dims(roll, 0), [batch_size, 1])  
        pitch = tf.tile(tf.expand_dims(pitch, 0), [batch_size, 1])  
        yaw = tf.tile(tf.expand_dims(yaw, 0), [batch_size, 1])  
        translation = tf.tile(tf.expand_dims(translation, 0), [batch_size, 1, 1])  

    # Convert Euler angles to rotation matrices  
    Rx = rotation_matrix_x(roll)  
    Ry = rotation_matrix_y(pitch)  
    Rz = rotation_matrix_z(yaw)  
  
    # Combine the rotations in XYZ order  
    # Note: TensorFlow uses matrix multiplication from the left, so the order is reversed compared to post-multiplication  
    R = tf.matmul(Rz, tf.matmul(Ry, Rx))
    return R

def nx4x4_matrix_to_euler_xyz(R):  
    """  
    Convert a batch of 4x4 rotation matrices to Euler angles in XYZ order.  
  
    Args:  
    R (tf.Tensor): Tensor of shape [N, 4, 4] containing 4x4 rotation matrices.  
  
    Returns:  
    tf.Tensor: Tensor of shape [N, 3] containing Euler angles (in radians) in XYZ order.  
    """  
    # 确保R是float32或float64类型  
    R = tf.cast(R, tf.float32)  
  
    # 提取旋转矩阵的旋转部分（忽略平移）  
    R_rot = R[:, :3, :3]  
  
    # 计算绕X轴的旋转（theta_x）  
    theta_x = tf.atan2(-R_rot[:, 2, 1], R_rot[:, 2, 2])  
  
    # 计算绕Y轴的旋转（theta_y）  
    c1 = tf.cos(theta_x)  
    s1 = tf.sin(theta_x)  
    theta_y = tf.atan2(s1 * R_rot[:, 0, 2] - c1 * R_rot[:, 1, 2],  
                       c1 * R_rot[:, 1, 1] + s1 * R_rot[:, 0, 1])  
  
    # 计算绕Z轴的旋转（theta_z）  
    c2 = tf.cos(theta_y)  
    s2 = tf.sin(theta_y)  
    c3 = tf.cos(theta_x)  
    s3 = tf.sin(theta_x)  
    theta_z = tf.atan2(s2 * s3 * R_rot[:, 0, 0] + c2 * c3 * R_rot[:, 0, 1] - s2 * c3 * R_rot[:, 1, 0],  
                       -c2 * s1 * R_rot[:, 0, 0] + s2 * s1 * R_rot[:, 0, 1] + c2 * c1 * R_rot[:, 1, 0])  
  
    # 返回结果  
    return tf.stack([theta_x, theta_y, theta_z], axis=-1) 

def euler_to_matrix_xyz_with_translation(roll, pitch, yaw, translation, batch_size=None):  
    """Convert Euler angles (roll, pitch, yaw in XYZ order) and translation to a batch of 4x4 transformation matrices."""  
    if batch_size is not None:  
        # Expand Euler angles and translation to match the batch size  
        roll = tf.tile(tf.expand_dims(roll, 0), [batch_size, 1])  
        pitch = tf.tile(tf.expand_dims(pitch, 0), [batch_size, 1])  
        yaw = tf.tile(tf.expand_dims(yaw, 0), [batch_size, 1])  
        translation = tf.tile(tf.expand_dims(translation, 0), [batch_size, 1, 1])  
  
    # Convert Euler angles to rotation matrices  
    Rx = rotation_matrix_x(roll)  
    Ry = rotation_matrix_y(pitch)  
    Rz = rotation_matrix_z(yaw)  
  
    # Combine the rotations in XYZ order  
    # Note: TensorFlow uses matrix multiplication from the left, so the order is reversed compared to post-multiplication  
    R = tf.matmul(Rz, tf.matmul(Ry, Rx))  
  
    # Create the 4x4 identity matrix with the rotation applied  
    T = tf.eye(4, batch_shape=[-1 if batch_size is None else batch_size])  
  
    # Append the translation to the rotation matrix  
    # TensorFlow doesn't have a direct function for this, so we manually set the last column of the first three rows  
    indices = tf.stack([  
        tf.repeat(tf.range(batch_size), 3),  # Batch indices  
        tf.tile(tf.constant([0, 1, 2]), [batch_size]),  # Row indices  
        tf.fill([batch_size * 3], 3)  # Column indices for the translation (always 3)  
    ], axis=1)  
    updates = tf.reshape(translation, [-1, 3])  # Flatten the translation tensor for updates  
  
    # Apply the translation updates  
    T = tf.tensor_scatter_nd_update(T, indices, updates)  
  
    # Set the rotation part of the matrix  
    indices_rotation = tf.stack([  
        tf.repeat(tf.range(batch_size), 9),  # Batch indices  
        tf.tile(tf.reshape(tf.range(3) * 3 + tf.range(3), [1, -1]), [batch_size, 1]),  # Row and column indices for the 3x3 rotation matrix  
    ], axis=1)  
    updates_rotation = tf.reshape(R, [-1, 3])  # Flatten the rotation matrix for updates  
  
    # Combine the rotation and translation  
    T = tf.tensor_scatter_nd_update(T, indices_rotation, updates_rotation)  
  
    return T

def calculate_delta_transform(pose1, pose2):  
    """  
    Calculate the delta transform (relative pose) between two poses.  
  
    Parameters:  
    - pose1: A TensorFlow tensor of shape [..., 4, 4] representing the first pose.  
    - pose2: A TensorFlow tensor of shape [..., 4, 4] representing the second pose.  
  
    Returns:  
    - delta_transform: A TensorFlow tensor of shape [..., 4, 4] representing the delta transform  
      between pose1 and pose2.  
    """  
    # Ensure the poses are tensors  
    pose1 = tf.convert_to_tensor(pose1)  
    pose2 = tf.convert_to_tensor(pose2)  
  
    # Check if the poses have the expected shape  
    # Note: We're assuming the last two dimensions are 4x4  
    # assert pose1.shape[-2:] == (4, 4), "pose1 must have shape [..., 4, 4]"  
    # assert pose2.shape[-2:] == (4, 4), "pose2 must have shape [..., 4, 4]"  
  
    # Calculate the inverse of pose1  
    # Note: tf.linalg.inv expects the last two dimensions to be square and invertible  
    inv_pose1 = tf.linalg.inv(pose1[..., tf.newaxis, :, :])  
    inv_pose1 = tf.squeeze(inv_pose1, axis=-3)  # Remove the added dimension  
  
    # Calculate the delta transform by multiplying the inverse of pose1 by pose2  
    delta_transform = tf.linalg.matmul(pose2, inv_pose1)  
  
    return delta_transform  

def chunk_act_obs(traj: Dict, window_size: int, future_action_window_size: int = 0) -> Dict:
    """
    Chunks actions and observations into the given window_size.

    "observation" keys are given a new axis (at index 1) of size `window_size` containing `window_size - 1`
    observations from the past and the current observation. "action" is given a new axis (at index 1) of size
    `window_size + future_action_window_size` containing `window_size - 1` actions from the past, the current
    action, and `future_action_window_size` actions from the future. "pad_mask" is added to "observation" and
    indicates whether an observation should be considered padding (i.e. if it had come from a timestep
    before the start of the trajectory).
    """
    traj_len = tf.shape(traj["action"])[0]
    action_dim = traj["action"].shape[-1]
    chunk_indices = tf.broadcast_to(tf.range(-window_size + 1, 1), [traj_len, window_size]) + tf.broadcast_to(
        tf.range(traj_len)[:, None], [traj_len, window_size]
    )

    action_chunk_indices = tf.broadcast_to(
        tf.range(-window_size + 1, 1 + future_action_window_size),
        [traj_len, window_size + future_action_window_size],
    ) + tf.broadcast_to(
        tf.range(traj_len)[:, None],
        [traj_len, window_size + future_action_window_size],
    )
    # import ipdb;ipdb.set_trace()

    floored_chunk_indices = tf.maximum(chunk_indices, 0)

    if "timestep" in traj["task"]:
        goal_timestep = traj["task"]["timestep"]
    else:
        goal_timestep = tf.fill([traj_len], traj_len - 1)

    # tf.print(traj_len, chunk_indices[:2])
    # tf.print('action_chunk_indices:', action_chunk_indices[:2])
    floored_action_chunk_indices = tf.minimum(tf.maximum(action_chunk_indices, 0), goal_timestep[:, None])

    # floored_action_chunk_indices = tf.maximum(action_chunk_indices, 0)
    # tf.print(goal_timestep, tf.shape(goal_timestep), goal_timestep[:2])
    # tf.print(floored_action_chunk_indices[:2])
    # pad_action_seq = tf.pad(traj['action'], paddings=[(0, window_size + future_action_window_size), (0, 0)])
    # import ipdb;ipdb.set_trace()
    # tf.zeros([window_size+future_action_window_size, ])
    if 'proprio' in traj['observation']:
        traj["proprio"] = traj["observation"]['proprio']
    traj["observation"] = tf.nest.map_structure(lambda x: tf.gather(x, floored_chunk_indices), traj["observation"])
    traj["action"] = tf.gather(traj["action"], floored_action_chunk_indices)

    if 'proprio' in traj:
        traj["proprio"] = tf.gather(traj["proprio"], floored_action_chunk_indices)

    if 'traj_index' in traj:
        traj["traj_index"] = tf.gather(traj["traj_index"], floored_action_chunk_indices)
    if 'idx_1' in traj:
        traj["idx_1"] = tf.gather(traj["idx_1"], floored_action_chunk_indices)
 
    # indicates whether an entire observation is padding
    traj["observation"]["pad_mask"] = chunk_indices >= 0

    # if no absolute_action_mask was provided, assume all actions are relative
    if "absolute_action_mask" not in traj and future_action_window_size > 0:
        logging.warning(
            "future_action_window_size > 0 but no absolute_action_mask was provided. "
            "Assuming all actions are relative for the purpose of making neutral actions."
        )
    absolute_action_mask = traj.get("absolute_action_mask", tf.zeros([traj_len, action_dim], dtype=tf.bool))
    neutral_actions = tf.where(
        absolute_action_mask[:, None, :],
        traj["action"],  # absolute actions are repeated (already done during chunking)
        tf.zeros_like(traj["action"]),  # relative actions are zeroed
    )
    # tf.print(traj['absolute_action_mask'])

    # actions past the goal timestep become neutral
    action_past_goal = action_chunk_indices > goal_timestep[:, None]
    traj["action"] = tf.where(action_past_goal[:, :, None], neutral_actions, traj["action"])
    if 'proprio' in traj:
        traj["proprio"] = tf.cast(traj["proprio"], tf.float32)
        traj["proprio"] = tf.where(action_past_goal[:, :, None], neutral_actions, traj["proprio"])
    traj['action_past_goal'] = action_past_goal
    # tf.print(tf.shape(traj['action']), 'shape')
    return traj

def filter_by_max_action(traj: Dict, max_action: float) -> Dict:
    # tf.print(max_action, tf.shape(traj['action']), )

    indices = tf.where(tf.math.reduce_all(tf.math.abs(traj["action"][:,...,:6]) <= max_action, axis=[1,2]))[:, 0]
    traj = tf.nest.map_structure(lambda x: tf.gather(x, indices), traj)
    # tf.print(max_action, tf.shape(traj['action']), )
    print(traj.keys())
    print(tf.shape(traj['observation']['image_primary']))
    return traj

def subsample(traj: Dict, subsample_length: int) -> Dict:
    """Subsamples trajectories to the given length."""
    traj_len = tf.shape(traj["action"])[0]
    if traj_len > subsample_length:
        indices = tf.random.shuffle(tf.range(traj_len))[:subsample_length]
        traj = tf.nest.map_structure(lambda x: tf.gather(x, indices), traj)

    return traj

def subsample_withgoal(traj: Dict, subsample_length: int) -> Dict:
    """Subsamples trajectories to the given length."""
    # traj_len = tf.shape(traj["action"])[0]
    gripper_change_status = traj['action'][..., 1:, -1] - traj['action'][..., :-1,  -1]
    # gripper_change_idx = tf.pad(gripper_change_status != 0, [(0, 0), (1, 0)])
    weighted = tf.reduce_sum(gripper_change_status, -1)
    
    epis = tf.random.normal(tf.shape(weighted), 0, 1, tf.float32) > 0.2
    epis = tf.cast(epis, tf.float32)
    gripper_change_indices = tf.where(weighted + epis > 0)[:, 0]
    # import ipdb;ipdb.set_trace()
    # goal_idx = tf.where(gripper_change_status)
    # tf.random.
    traj = tf.nest.map_structure(lambda x: tf.gather(x, gripper_change_indices), traj) 
    

    # weighted = weighted + 0.2
    # # 确保权重是归一化的  
    # normalized_weights = weighted / tf.reduce_sum(weighted)    
    # # 创建一个Categorical分布  
    # categorical = tf.distributions.Categorical(probs=normalized_weights)  
    # # 从分布中采样  
    # # 注意：sample方法默认返回的是一个shape为[1,]的Tensor，表示采样的结果  
    # # 如果你想要一次性采样多个样本，可以将sample方法的参数设置为所需的样本数，例如sample(5)  
    # sample = categorical.sample(subsample_length)  
    # traj = tf.nest.map_structure(lambda x: tf.gather(x, sample), traj) 
    return traj

def subsample_withgoalprop(traj: Dict, subsample_length: int) -> Dict:
    """Subsamples trajectories to the given length."""
    # traj_len = tf.shape(traj["action"])[0]
    traj_len = tf.shape(traj["action"])[0]
    if traj_len > subsample_length:
        # import ipdb;ipdb.set_trace()
        gripper_change_status = traj['action'][..., 1:, -1] - traj['action'][..., :-1,  -1]
        # gripper_change_idx = tf.pad(gripper_change_status != 0, [(0, 0), (1, 0)])
        # import ipdb;ipdb.set_trace()
        weighted = tf.reduce_sum(gripper_change_status, -1)
        weighted1 = tf.reduce_sum(gripper_change_status[..., :4], -1)
        # weighted1 = weighted1 + tf.reduce_sum(gripper_change_status[..., 1:4], -1) * 0.2 + tf.reduce_sum(gripper_change_status[..., 1:3], -1) * 0.2 + tf.reduce_sum(gripper_change_status[..., 1:2], -1) * 0.2
        # gripper_change_indices = tf.where(weighted)
        # traj = tf.nest.map_structure(lambda x: tf.gather(x, gripper_change_indices), traj) 
        
    
        weighted = weighted + 0.2 + weighted1
        # 确保权重是归一化的  
        normalized_weights = weighted / tf.reduce_sum(weighted)    
        # 创建一个Categorical分布  
        import tensorflow_probability as tfp
        categorical = tfp.distributions.Categorical(probs=normalized_weights)  
        # 从分布中采样  
        # 注意：sample方法默认返回的是一个shape为[1,]的Tensor，表示采样的结果  
        # 如果你想要一次性采样多个样本，可以将sample方法的参数设置为所需的样本数，例如sample(5)  
        sample = categorical.sample(subsample_length)  
        # tf.print(tf.shape(traj['action']), subsample_length, sample)
        traj = tf.nest.map_structure(lambda x: tf.gather(x, sample), traj) 
    return traj

def add_pad_mask_dict(traj: Dict) -> Dict:
    """
    Adds a dictionary indicating which elements of the observation/task should be treated as padding.
        =>> traj["observation"|"task"]["pad_mask_dict"] = {k: traj["observation"|"task"][k] is not padding}
    """
    traj_len = tf.shape(traj["action"])[0]

    for key in ["observation", "task"]:
        pad_mask_dict = {}
        for subkey in traj[key]:
            # Handles "language_instruction", "image_*", and "depth_*"
            if traj[key][subkey].dtype == tf.string:
                pad_mask_dict[subkey] = tf.strings.length(traj[key][subkey]) != 0

            # All other keys should not be treated as padding
            else:
                pad_mask_dict[subkey] = tf.ones([traj_len], dtype=tf.bool)

        traj[key]["pad_mask_dict"] = pad_mask_dict

    return traj