"""
transforms.py

Defines a registry of per-dataset standardization transforms for each dataset in Open-X Embodiment.

Transforms adopt the following structure:
    Input: Dictionary of *batched* features (i.e., has leading time dimension)
    Output: Dictionary `step` =>> {
        "observation": {
            <image_keys, depth_image_keys>
            State (in chosen state representation)
        },
        "action": Action (in chosen action representation),
        "language_instruction": str
    }
"""

from typing import Any, Dict

import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tft
from prismatic.vla.datasets.rlds.oxe.utils.droid_utils import droid_baseact_transform, droid_baseact_transform_delta, droid_baseact_transform_delta_state, droid_baseact_transform_delta_state_with_trajid, droid_baseact_transform_delta_state1, droid_finetuning_transform, rand_swap_exterior_images, Calvin_finetuning_transform
from prismatic.vla.datasets.rlds.utils.data_utils import (
    binarize_gripper_actions,
    invert_gripper_actions,
    rel2abs_gripper_actions,
    relabel_bridge_actions,
)

EMPTY_IMG = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x01,\x01,\x00\x00\xff\xdb\x00C\x00\x02\x01\x01\x01\x01\x01\x02\x01\x01\x01\x02\x02\x02\x02\x02\x04\x03\x02\x02\x02\x02\x05\x04\x04\x03\x04\x06\x05\x06\x06\x06\x05\x06\x06\x06\x07\t\x08\x06\x07\t\x07\x06\x06\x08\x0b\x08\t\n\n\n\n\n\x06\x08\x0b\x0c\x0b\n\x0c\t\n\n\n\xff\xdb\x00C\x01\x02\x02\x02\x02\x02\x02\x05\x03\x03\x05\n\x07\x06\x07\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\xff\xc0\x00\x11\x08\x01\x00\x01\x00\x03\x01"\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n\x16\x17\x18\x19\x1a%&\'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xc4\x00\x1f\x01\x00\x03\x01\x01\x01\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x11\x00\x02\x01\x02\x04\x04\x03\x04\x07\x05\x04\x04\x00\x01\x02w\x00\x01\x02\x03\x11\x04\x05!1\x06\x12AQ\x07aq\x13"2\x81\x08\x14B\x91\xa1\xb1\xc1\t#3R\xf0\x15br\xd1\n\x16$4\xe1%\xf1\x17\x18\x19\x1a&\'()*56789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x82\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00?\x00\xfe\x7f\xe8\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00\xff\xd9'        


def joints_to_pose_batch_tf(joints_batch):
    """
    Apply forward kinematics method for franka panda robot.

    Args:
    joints_batch (tf.Tensor): Tensor of shape (batch_size, 7) with target joints values

    Returns:
    tf.Tensor: Tensor of shape (batch_size, 7) with target eef poses in the format (x, y, z, w, i, j, k).
    """
    assert joints_batch.shape[1] == 7, "Each element in the batch must have 7 joint values"
    import tensorflow as tf
    import numpy as np
    def mat2euler(matrix):
        """
        Convert rotation matrices to Euler angles (XYZ order).

        Args:
        matrix (tf.Tensor): Tensor of shape (batch_size, 3, 3) representing rotation matrices.

        Returns:
        tf.Tensor: Tensor of shape (batch_size, 3) representing Euler angles (yaw, pitch, roll).
        """
        batch_size = tf.shape(matrix)[0]
        
        # Calculate yaw (Z)
        yaw = tf.atan2(matrix[:, 1, 0], matrix[:, 0, 0])
        
        # Calculate pitch (Y)
        pitch = tf.atan2(-matrix[:, 2, 0], tf.sqrt(tf.square(matrix[:, 2, 1]) + tf.square(matrix[:, 2, 2])))
        
        # Calculate roll (X)
        roll = tf.atan2(matrix[:, 2, 1], matrix[:, 2, 2])
        
        return tf.stack([roll, pitch, yaw], axis=1)

    
    
    def dh_params(joints_batch):
        M_PI = np.pi

        batch_size = tf.shape(joints_batch)[0]
        zeros = tf.zeros([batch_size], dtype=tf.float32)
        ones = tf.ones([batch_size], dtype=tf.float32)

        dh = tf.stack(
            [
                tf.stack([zeros, zeros, 0.333 * ones, joints_batch[:, 0]], axis=1),
                tf.stack([-M_PI / 2 * ones, zeros, zeros, joints_batch[:, 1]], axis=1),
                tf.stack([M_PI / 2 * ones, zeros, 0.316 * ones, joints_batch[:, 2]], axis=1),
                tf.stack([M_PI / 2 * ones, 0.0825 * ones, zeros, joints_batch[:, 3]], axis=1),
                tf.stack([-M_PI / 2 * ones, -0.0825 * ones, 0.384 * ones, joints_batch[:, 4]], axis=1),
                tf.stack([M_PI / 2 * ones, zeros, zeros, joints_batch[:, 5]], axis=1),
                tf.stack([M_PI / 2 * ones, 0.088 * ones, 0.107 * ones, joints_batch[:, 6]], axis=1),
            ], axis=1
        )
        return dh

    def TF_matrix(i, dh):
        # Define Transformation matrix based on DH params
        alpha = dh[:, i, 0]
        a = dh[:, i, 1]
        d = dh[:, i, 2]
        q = dh[:, i, 3]

        TF = tf.stack(
            [
                tf.stack([tf.cos(q), -tf.sin(q), tf.zeros_like(q), a], axis=1),
                tf.stack([tf.sin(q) * tf.cos(alpha), tf.cos(q) * tf.cos(alpha), -tf.sin(alpha), -tf.sin(alpha) * d], axis=1),
                tf.stack([tf.sin(q) * tf.sin(alpha), tf.cos(q) * tf.sin(alpha), tf.cos(alpha), tf.cos(alpha) * d], axis=1),
                tf.stack([tf.zeros_like(q), tf.zeros_like(q), tf.zeros_like(q), tf.ones_like(q)], axis=1)
            ], axis=1
        )
        return TF
    dh_parameters = dh_params(joints_batch)

    T_01 = TF_matrix(0, dh_parameters)
    T_12 = TF_matrix(1, dh_parameters)
    T_23 = TF_matrix(2, dh_parameters)
    T_34 = TF_matrix(3, dh_parameters)
    T_45 = TF_matrix(4, dh_parameters)
    T_56 = TF_matrix(5, dh_parameters)
    T_67 = TF_matrix(6, dh_parameters)

    T_07 = tf.linalg.matmul(T_01, T_12)
    T_07 = tf.linalg.matmul(T_07, T_23)
    T_07 = tf.linalg.matmul(T_07, T_34)
    T_07 = tf.linalg.matmul(T_07, T_45)
    T_07 = tf.linalg.matmul(T_07, T_56)
    T_07 = tf.linalg.matmul(T_07, T_67)

    translation = T_07[:, :3, 3]
    
    euler = mat2euler(T_07[:, :3, :3])
    return tf.concat([translation, euler], axis=1)


def bridge_oxe_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Applies to version of Bridge V2 in Open X-Embodiment mixture.

    Note =>> In original Bridge V2 dataset, the first timestep has an all-zero action, so we remove it!
    """
    for key in trajectory.keys():
        if key == "traj_metadata":
            continue
        elif key in ["observation", "action"]:
            for key2 in trajectory[key]:
                trajectory[key][key2] = trajectory[key][key2][1:]
        else:
            trajectory[key] = trajectory[key][1:]

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            tf.cast(trajectory["action"]["open_gripper"][:, None], tf.float32),
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    trajectory = relabel_bridge_actions(trajectory)
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    return trajectory


def bridge_orig_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Applies to original version of Bridge V2 from the official project website.

    Note =>> In original Bridge V2 dataset, the first timestep has an all-zero action, so we remove it!
    """
    for key in trajectory.keys():
        if key == "traj_metadata":
            continue
        elif key == "observation":
            for key2 in trajectory[key]:
                trajectory[key][key2] = trajectory[key][key2][1:]
        else:
            trajectory[key] = trajectory[key][1:]

    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            binarize_gripper_actions(trajectory["action"][:, -1])[:, None],
        ],
        axis=1,
    )
    trajectory = relabel_bridge_actions(trajectory)
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]

    trajectory["observation"]["proprio"] = tf.concat([trajectory["observation"]["state"][:, :6], trajectory["observation"]["state"][:, -1:]], axis=1)

    trajectory['observation']['image_1'] = tf.cond((trajectory['observation']['image_1'] == EMPTY_IMG)[0], lambda : trajectory['observation']['image_0'], lambda :trajectory['observation']['image_1'])
    trajectory['observation']['image_2'] = tf.cond((trajectory['observation']['image_2'] == EMPTY_IMG)[0], lambda :trajectory['observation']['image_0'], lambda :trajectory['observation']['image_2'])
    trajectory['observation']['image_3'] = tf.cond((trajectory['observation']['image_3'] == EMPTY_IMG)[0], lambda :trajectory['observation']['image_0'], lambda :trajectory['observation']['image_3'])

    trajectory['observation']['image_0'], trajectory['observation']['image_1'], trajectory['observation']['image_2'],trajectory['observation']['image_3'] = \
        rand_swap_exterior_images4(trajectory['observation']['image_0'], trajectory['observation']['image_1'], trajectory['observation']['image_2'],trajectory['observation']['image_3'])

    # bridge_orig
    return trajectory

def bridge_orig_dataset_transform_state(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Applies to original version of Bridge V2 from the official project website.

    Note =>> In original Bridge V2 dataset, the first timestep has an all-zero action, so we remove it!
    """
    import tensorflow_graphics as tfg
    from openvla.prismatic.vla.datasets.rlds import traj_transforms
    for key in trajectory.keys():
        if key == "traj_metadata":
            continue
        elif key == "observation":
            for key2 in trajectory[key]:
                trajectory[key][key2] = tf.cond(tf.shape(trajectory[key][key2][:])[0] > 1, lambda: trajectory[key][key2][1:], lambda: trajectory[key][key2][:], )
        else:
            # trajectory[key] = trajectory[key][1:]
            trajectory[key] = tf.cond(tf.shape(trajectory[key][:])[0] > 1, lambda:  trajectory[key][1:], lambda:  trajectory[key][:], )

    rot_matrix = tfg.geometry.transformation.rotation_matrix_3d.from_euler(trajectory["observation"]["state"][:-1,3:6]) # roll pitch yaw
    
    rot_matrix_next = tfg.geometry.transformation.rotation_matrix_3d.from_euler(trajectory["observation"]["state"][1:,3:6]) # roll pitch yaw
    rot_delta_matrix = traj_transforms.calculate_delta_transform(rot_matrix, rot_matrix_next)
    # dR = traj_transforms.nx4x4_matrix_to_euler_xyz(rot_delta_matrix)
    dR = tfg.geometry.transformation.euler.from_rotation_matrix(rot_delta_matrix)


    dR = tf.cast(dR, dtype=tf.float64)
    # tf.print(dR[0])
    dt = trajectory["observation"]["state"][1:,:3] - trajectory["observation"]["state"][:-1,:3]
    # trajectory['action'] = tf.concat([trans_delta, rot_delta_euler, trajectory['action'][5:6]])
    
    # dt1 = trajectory["action_dict"]["cartesian_velocity"][:, :3]
    # dR1 = trajectory["action_dict"]["cartesian_velocity"][:, 3:6]
    
    # print(dR, dt)
    trajectory["observation"]["proprio"] = tf.concat(
        [
            trajectory["observation"]["state"][:, :6],
            binarize_gripper_actions(trajectory["action"][:, -1])[:, None],
        ], 
        axis=1,
        )
    tmp_proprio = trajectory["observation"]["proprio"]
    trajectory["observation"]["state"] = trajectory["observation"]["proprio"]
    trajectory = tf.nest.map_structure(lambda x: x[:-1], trajectory)
    trajectory["observation"]["proprio"] = tmp_proprio[1:]
    trajectory["action"] = tf.concat(
        [
            dt,
            tf.cast(dR, tf.float32),
            binarize_gripper_actions(trajectory["action"][:, -1])[:, None],
        ],
        axis=-1,
    )
    # tf.print(tf.shape(trajectory["observation"]["state"][:, :6]), tf.shape(trajectory["action"][:, -1]), tf.shape(binarize_gripper_actions(trajectory["action"][:, -1])[:, None]),)

    # trajectory["observation"]["state"] = tf.concat(
    #     [
    #         trajectory["observation"]["state"][:-1, :6],
    #         binarize_gripper_actions(trajectory["action"][:-1, -1])[:, None],
    #     ], 
    #     axis=1,
    #     )
    # tf.print(tf.shape(trajectory["action"]), tf.shape(trajectory["observation"]["proprio"]))
    trajectory["action"] = tf.cast(trajectory["action"], tf.float32)
    trajectory["observation"]["state"] = tf.cast(trajectory["observation"]["state"], tf.float32)
    trajectory["observation"]["proprio"] = tf.cast(trajectory["observation"]["proprio"], tf.float32)
        
    trajectory['observation']['image_0'] = trajectory['observation']['image_0']
    trajectory['observation']['image_1'] = trajectory['observation']['image_1']
    trajectory['observation']['image_2'] = trajectory['observation']['image_2']
    trajectory['observation']['image_3'] = trajectory['observation']['image_3']

    trajectory['observation']['image_1'] = tf.cond((trajectory['observation']['image_1'] == EMPTY_IMG)[0], lambda : trajectory['observation']['image_0'], lambda :trajectory['observation']['image_1'])
    trajectory['observation']['image_2'] = tf.cond((trajectory['observation']['image_2'] == EMPTY_IMG)[0], lambda :trajectory['observation']['image_0'], lambda :trajectory['observation']['image_2'])
    trajectory['observation']['image_3'] = tf.cond((trajectory['observation']['image_3'] == EMPTY_IMG)[0], lambda :trajectory['observation']['image_0'], lambda :trajectory['observation']['image_3'])

    trajectory['observation']['image_0'], trajectory['observation']['image_1'], trajectory['observation']['image_2'],trajectory['observation']['image_3'] = \
        rand_swap_exterior_images4(trajectory['observation']['image_0'], trajectory['observation']['image_1'], trajectory['observation']['image_2'],trajectory['observation']['image_3'])
    # trajectory["observation"].pop('state')
    
    # bridge_orig
    # trajectory.pop('reward')
    # trajectory.pop('is_last')
    # trajectory.pop('is_terminal')
    # trajectory.pop('discount')
    # trajectory.pop('is_first')
    # trajectory.pop('language_embedding')
    # trajectory.pop('traj_metadata')
    # trajectory.pop('_len')
    # trajectory.pop('_traj_index')
    # trajectory.pop('_frame_index')
    # trajectory["observation"].pop('state')

    # tf.print(trajectory.keys(), trajectory['observation'].keys(), tf.shape(trajectory['observation']['image_3']), tf.shape(trajectory["action"]), tf.shape(trajectory["observation"]["proprio"]), tf.shape(trajectory["observation"]["state"]))

    # dict_keys(['reward', 'is_last', 'action', 'is_terminal', 'language_instruction', 'discount', 'is_first', 'observation', 'language_embedding', 'traj_metadata', '_len', '_traj_index', '_frame_index'])
    return trajectory


def ppgm_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            binarize_gripper_actions(trajectory["action"][:, -1])[:, None],
        ],
        axis=1,
    )
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["cartesian_position"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["gripper_position"][:, -1:]
    return trajectory


def rt1_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)


    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def kuka_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    # decode compressed state
    eef_value = tf.io.decode_compressed(
        trajectory["observation"]["clip_function_input/base_pose_tool_reached"],
        compression_type="ZLIB",
    )
    eef_value = tf.io.decode_raw(eef_value, tf.float32)
    trajectory["observation"]["clip_function_input/base_pose_tool_reached"] = tf.reshape(eef_value, (-1, 7))
    
    gripper_value = tf.io.decode_compressed(trajectory["observation"]["gripper_closed"], compression_type="ZLIB")
    gripper_value = tf.io.decode_raw(gripper_value, tf.float32)

    
    trajectory["observation"]["gripper_closed"] = tf.reshape(gripper_value, (-1, 1))
    
    quat = trajectory["observation"]["clip_function_input/base_pose_tool_reached"]
    trajectory["observation"]["proprio"] = tf.concat([quat[:, :3],
                                                    tft.euler.from_quaternion(quat[:, 3:7]),
                                                      trajectory["observation"]["gripper_closed"]], axis=-1)
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    # )  # delete uninformative language instruction
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def kuka_dataset_transform_state(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    # decode compressed state
    eef_value = tf.io.decode_compressed(
        trajectory["observation"]["clip_function_input/base_pose_tool_reached"],
        compression_type="ZLIB",
    )
    eef_value = tf.io.decode_raw(eef_value, tf.float32)
    trajectory["observation"]["clip_function_input/base_pose_tool_reached"] = tf.reshape(eef_value, (-1, 7))
    
    gripper_value = tf.io.decode_compressed(trajectory["observation"]["gripper_closed"], compression_type="ZLIB")
    gripper_value = tf.io.decode_raw(gripper_value, tf.float32)

    
    trajectory["observation"]["gripper_closed"] = tf.reshape(gripper_value, (-1, 1))
    import tensorflow_graphics.geometry.transformation as tft
    quat = trajectory["observation"]["clip_function_input/base_pose_tool_reached"]
    trajectory["observation"]["proprio"] = tf.concat([quat[:, :3],
                                                    tft.euler.from_quaternion(quat[:, 3:7]),
                                                      trajectory["observation"]["gripper_closed"]], axis=-1)
    trajectory["observation"]["state"] = trajectory["observation"]["proprio"]
    tmp_proprio = trajectory["observation"]["proprio"]
    trajectory = tf.nest.map_structure(lambda x: x[:-1], trajectory)
    trajectory["observation"]["proprio"] = tmp_proprio[1:]
    
    trajectory["action"] = tf.cast(trajectory["action"], tf.float32)
    trajectory["observation"]["state"] = tf.cast(trajectory["observation"]["state"], tf.float32)
    trajectory["observation"]["proprio"] = tf.cast(trajectory["observation"]["proprio"], tf.float32)
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    # )  # delete uninformative language instruction
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def taco_play_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["state_eef"] = trajectory["observation"]["robot_obs"][:, :6]
    trajectory["observation"]["state_gripper"] = trajectory["observation"]["robot_obs"][:, 7:8]
    trajectory["action"] = trajectory["action"]["rel_actions_world"]

    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            tf.clip_by_value(trajectory["action"][:, -1:], 0, 1),
        ),
        axis=-1,
    )

    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def jaco_play_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["state_eef"] = trajectory["observation"]["end_effector_cartesian_pos"][:, :6]
    trajectory["observation"]["state_gripper"] = trajectory["observation"]["end_effector_cartesian_pos"][:, -1:]

    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            tf.zeros_like(trajectory["action"]["world_vector"]),
            gripper_action[:, None],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def berkeley_cable_routing_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            tf.zeros_like(trajectory["action"]["world_vector"][:, :1]),
        ),
        axis=-1,
    )
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    # )  # delete uninformative language instruction
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def roboturk_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert absolute gripper action, +1 = open, 0 = close
    gripper_action = invert_gripper_actions(tf.clip_by_value(trajectory["action"]["gripper_closedness_action"], 0, 1))

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action,
        ),
        axis=-1,
    )
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    # )  # delete uninformative language instruction
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def nyu_door_opening_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    # )  # delete uninformative language instruction
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def viola_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # make gripper action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, None]
    gripper_action = tf.clip_by_value(gripper_action, 0, 1)
    gripper_action = invert_gripper_actions(gripper_action)
    # ee_states
    import tensorflow_graphics as tfg
    dR = tfg.geometry.transformation.euler.from_rotation_matrix(tf.reshape(trajectory["observation"]['ee_states'], [-1, 4, 4])[:, :3, :3])
    dt = tf.reshape(trajectory["observation"]['ee_states'], [-1, 4, 4])[:, 3, :3]
    # tf.print(dt, 'dt')

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action,
        ),
        axis=-1,
    )
    trajectory['observation']['proprio'] = tf.concat([ dt, dR], axis=-1)
    # tf.print(tf.shape(trajectory['observation']['proprio']))
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    # )  # delete uninformative language instruction
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def berkeley_autolab_ur5_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    import tensorflow_graphics as tfg
    dt = trajectory["observation"]['robot_state'][:, 6:9]
    dR = tft.euler.from_quaternion(trajectory["observation"]['robot_state'][:, 9:13])


    trajectory["observation"]["state"] = tf.concat([
        dt, dR, trajectory["observation"]['robot_state'][:, 13:14]
    ], axis=-1)
    trajectory["observation"]["depth"] = trajectory["observation"].pop("image_with_depth")

    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def toto_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            tf.cast(trajectory["action"]["open_gripper"][:, None], tf.float32),
        ),
        axis=-1,
    )
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    # )  # delete uninformative language instruction
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def language_table_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # default to "open" gripper
    trajectory["action"] = tf.concat(
        (
            trajectory["action"],
            tf.zeros_like(trajectory["action"]),
            tf.zeros_like(trajectory["action"]),
            tf.ones_like(trajectory["action"][:, :1]),
        ),
        axis=-1,
    )

    # decode language instruction
    instruction_bytes = trajectory["observation"]["instruction"]
    instruction_encoded = tf.strings.unicode_encode(instruction_bytes, output_encoding="UTF-8")
    # Remove trailing padding --> convert RaggedTensor to regular Tensor.
    trajectory["language_instruction"] = tf.strings.split(instruction_encoded, "\x00")[:, :1].to_tensor()[:, 0]
    return trajectory


def pusht_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            trajectory["action"]["gripper_closedness_action"][:, None],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def stanford_kuka_multimodal_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["depth_image"] = trajectory["observation"]["depth_image"][..., 0]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tf.zeros_like(trajectory["action"][:, :3]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def nyu_rot_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][..., :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][..., -1:]
    trajectory["action"] = trajectory["action"][..., :7]
    return trajectory


def stanford_hydra_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert gripper action, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(trajectory["action"][:, -1:]),
        ),
        axis=-1,
    )

    trajectory["observation"]["eef_state"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :3],
            trajectory["observation"]["state"][:, 7:10],
        ),
        axis=-1,
    )
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -3:-2]
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["language_instruction"]), ""
    # )  # delete uninformative language instruction
    return trajectory

def stanford_hydra_dataset_transform_state(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert gripper action, +1 = open, 0 = close

    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :3],
            trajectory["observation"]["state"][:, 7:10],
            invert_gripper_actions(trajectory["action"][:, -1:]),
        ),
        axis=-1,
    )
    trajectory["observation"]["eef_state"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :3],
            trajectory["observation"]["state"][:, 7:10],
        ),
        axis=-1,
    )

    import tensorflow_graphics as tfg
    from openvla.prismatic.vla.datasets.rlds import traj_transforms

    rot_matrix = tfg.geometry.transformation.rotation_matrix_3d.from_euler(trajectory["observation"]["eef_state"][:-1, -6:-3]) # roll pitch yaw
    
    rot_matrix_next = tfg.geometry.transformation.rotation_matrix_3d.from_euler(trajectory["observation"]["eef_state"][1:, -6:-3] ) # roll pitch yaw
    rot_delta_matrix = traj_transforms.calculate_delta_transform(rot_matrix, rot_matrix_next)
    # dR = traj_transforms.nx4x4_matrix_to_euler_xyz(rot_delta_matrix)
    dR = tfg.geometry.transformation.euler.from_rotation_matrix(rot_delta_matrix)
 
    dR = tf.cast(dR, dtype=tf.float32)
    dt = trajectory["observation"]["eef_state"][1:, -3:] - trajectory["observation"]["eef_state"][:-1, -3:]

    tmp_proprio = trajectory["observation"]["proprio"]
    trajectory["observation"]["state"] = tmp_proprio
    trajectory = tf.nest.map_structure(lambda x: x[:-1], trajectory)
    trajectory["observation"]["proprio"] = tmp_proprio[1:]
    trajectory["action"] = tf.concat(
        (
            dt,
            dR,
            invert_gripper_actions(trajectory["action"][:, -1:]),
        ),
        axis=-1,
    )
    trajectory["action"] = tf.cast(trajectory["action"], tf.float32)
    trajectory["observation"]["state"] = tf.cast(trajectory["observation"]["state"], tf.float32)
    trajectory["observation"]["proprio"] = tf.cast(trajectory["observation"]["proprio"], tf.float32)
        
    # trajectory["observation"]["eef_state"] = trajectory["observation"]["eef_state"][:-1]

    # trajectory["observation"]["image"] = trajectory["observation"]["image"][:-1]
    # trajectory["observation"]["wrist_image"] = trajectory["observation"]["wrist_image"][:-1]
    # trajectory["observation"]["state"] = trajectory["observation"]["state"][1:]


    return trajectory


def austin_buds_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)),
        ),
        axis=-1,
    )

    import tensorflow_graphics as tfg
    dR = tfg.geometry.transformation.euler.from_rotation_matrix(tf.reshape(trajectory["observation"]['state'][:, 8:], [-1, 4, 4])[:, :3, :3])
    dt = tf.reshape(trajectory["observation"]['state'][:, 8:], [-1, 4, 4])[:, 3, :3]


    trajectory["observation"]["state"] = tf.concat([
        dt, dR, trajectory["observation"]['state'][:, 7:8]
    ], axis=-1
    )
    # trajectory["observation"]["state"][:, :8]

    # trajectory["observation"]["state"] = trajectory["observation"]["state"][:, 8:]
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["language_instruction"]), ""
    # )  # delete uninformative language instruction
    return trajectory


def nyu_franka_play_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["depth"] = tf.cast(trajectory["observation"]["depth"][..., 0], tf.float32)
    trajectory["observation"]["depth_additional_view"] = tf.cast(
        trajectory["observation"]["depth_additional_view"][..., 0], tf.float32
    )
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, -6:]

    # clip gripper action, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, -8:-2],
            tf.clip_by_value(trajectory["action"][:, -2:-1], 0, 1),
        ),
        axis=-1,
    )

    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["language_instruction"]), ""
    # )  # delete uninformative language instruction
    return trajectory



def nyu_franka_play_dataset_transform_state(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # trajectory["observation"]["depth"] = tf.cast(trajectory["observation"]["depth"][..., 0], tf.float32)
    # trajectory["observation"]["depth_additional_view"] = tf.cast(
    #     trajectory["observation"]["depth_additional_view"][1:,..., 0], tf.float32
    # )
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"][:, -6:]


    import tensorflow_graphics as tfg
    from openvla.prismatic.vla.datasets.rlds import traj_transforms

    rot_matrix = tfg.geometry.transformation.rotation_matrix_3d.from_euler(trajectory["observation"]["state"][:-1, -3:]) # roll pitch yaw
    
    rot_matrix_next = tfg.geometry.transformation.rotation_matrix_3d.from_euler(trajectory["observation"]["state"][1:, -3:] ) # roll pitch yaw
    rot_delta_matrix = traj_transforms.calculate_delta_transform(rot_matrix, rot_matrix_next)
    # dR = traj_transforms.nx4x4_matrix_to_euler_xyz(rot_delta_matrix)
    dR = tfg.geometry.transformation.euler.from_rotation_matrix(rot_delta_matrix)
 
    dR = tf.cast(dR, dtype=tf.float32)
    dt = trajectory["observation"]["state"][1:, -6:-3] - trajectory["observation"]["state"][:-1, -6:-3]
    
    tmp_proprio = tf.concat([trajectory["observation"]["proprio"], tf.clip_by_value(trajectory["action"][:, -2:-1], 0, 1)], axis=-1)
    trajectory["observation"]["state"] = tmp_proprio
    trajectory = tf.nest.map_structure(lambda x: x[:-1], trajectory)
    trajectory["observation"]["proprio"] = tf.cast(tmp_proprio[1:], tf.float32)
    trajectory["action"] = tf.concat(
        (
            dt,
            dR,
            tf.clip_by_value(trajectory["action"][:, -2:-1], 0, 1),
        ),
        axis=-1,
    )
    trajectory["action"] = tf.cast(trajectory["action"], tf.float32)
    trajectory["observation"]["state"] = tf.cast(trajectory["observation"]["state"], tf.float32)
    trajectory["observation"]["proprio"] = tf.cast(trajectory["observation"]["proprio"], tf.float32)
    
    trajectory["observation"]["image"], trajectory["observation"]["image_additional_view"] = (
        rand_swap_exterior_images(
            trajectory["observation"]["image"],
            trajectory["observation"]["image_additional_view"],
        )
    )
    return trajectory



def maniskill_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][..., 7:8]
    return trajectory


def furniture_bench_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    import tensorflow_graphics.geometry.transformation as tft

    trajectory["observation"]["state"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :3],
            tft.euler.from_quaternion(trajectory["observation"]["state"][:, 3:7]),
            trajectory["observation"]["state"][:, -1:],
        ),
        axis=-1,
    )

    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tft.euler.from_quaternion(trajectory["action"][:, 3:7]),
            invert_gripper_actions(tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)),
        ),
        axis=-1,
    )
    return trajectory


def cmu_franka_exploration_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., :-1]
    return trajectory


def ucsd_kitchen_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["proprio"] = joints_to_pose_batch_tf(trajectory["observation"]["state"][...,:7])
    trajectory["observation"]["joint_state"] = trajectory["observation"]["state"][:, :7]
    
    trajectory["observation"]["gripper_state"] = tf.concat([tf.zeros_like(trajectory["action"][:1, -2:-1]), trajectory["action"][:-1, -2:-1]], axis=0)

    trajectory["action"] = trajectory["action"][..., :-1]
    return trajectory


def ucsd_pick_place_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tf.zeros_like(trajectory["action"][:, :3]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def austin_sailor_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)),
        ),
        axis=-1,
    )
    import tensorflow_graphics as tfg
    dR = tfg.geometry.transformation.euler.from_rotation_matrix(tf.reshape(trajectory["observation"]['state_ee'], [-1, 4, 4])[:, :3, :3])
    dt = tf.reshape(trajectory["observation"]['state_ee'], [-1, 4, 4])[:, 3, :3]
    # tf.print(dt, 'dt')

    trajectory["observation"]['proprio'] = tf.concat(
        (
            dt,
            dR,
        ),
        axis=-1,
    )
#             trajectory["observation"]['state_gripper'],

    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["language_instruction"]), ""
    # )  # delete uninformative language instruction
    return trajectory


def austin_sirius_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)),
        ),
        axis=-1,
    )
    import tensorflow_graphics as tfg
    dR = tfg.geometry.transformation.euler.from_rotation_matrix(tf.reshape(trajectory["observation"]['state_ee'], [-1, 4, 4])[:, :3, :3])
    dt = tf.reshape(trajectory["observation"]['state_ee'], [-1, 4, 4])[:, 3, :3]
    # tf.print(dt, 'dt')

    trajectory["observation"]['proprio'] = tf.concat(
        (
            dt,
            dR,
        ),
        axis=-1,
    )
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["language_instruction"]), ""
    # )  # delete uninformative language instruction
    return trajectory


def bc_z_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["future/xyz_residual"][:, :3],
            trajectory["action"]["future/axis_angle_residual"][:, :3],
            invert_gripper_actions(tf.cast(trajectory["action"]["future/target_close"][:, :1], tf.float32)),
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def tokyo_pr2_opening_fridge_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    trajectory["action"] = trajectory["action"][..., :-1]
    return trajectory


def tokyo_pr2_tabletop_manipulation_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    trajectory["action"] = trajectory["action"][..., :-1]
    return trajectory


def utokyo_xarm_pick_place_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    return trajectory


def utokyo_xarm_bimanual_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., -7:]
    return trajectory


def robo_net_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :4],
            tf.zeros_like(trajectory["observation"]["state"][:, :2]),
        ),
        axis=-1,
    )
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :4],
            tf.zeros_like(trajectory["action"][:, :2]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def berkeley_mvp_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    return trajectory


def berkeley_rpt_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    return trajectory


def kaist_nonprehensible_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["state"] = trajectory["observation"]["state"][:, -7:]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            tf.zeros_like(trajectory["action"][:, :1]),
        ),
        axis=-1,
    )
    return trajectory


def stanford_mask_vit_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = tf.concat(
        (
            trajectory["observation"]["end_effector_pose"][:, :4],
            tf.zeros_like(trajectory["observation"]["end_effector_pose"][:, :2]),
        ),
        axis=-1,
    )
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["end_effector_pose"][:, -1:]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :4],
            tf.zeros_like(trajectory["action"][:, :2]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def tokyo_lsmo_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    return trajectory


def dlr_sara_pour_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    return trajectory


def dlr_sara_grid_clamp_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["state"] = trajectory["observation"]["state"][:, :6]
    return trajectory


def dlr_edan_shared_control_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert gripper action, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(trajectory["action"][:, -1:]),
        ),
        axis=-1,
    )
    return trajectory


def dlr_edan_shared_control_dataset_transform_state(trajectory: Dict[str, Any]) -> Dict[str, Any]:

    import tensorflow_graphics as tfg
    from openvla.prismatic.vla.datasets.rlds import traj_transforms
    rot_matrix = tfg.geometry.transformation.rotation_matrix_3d.from_euler(trajectory["observation"]["state"][:-1, 3:6]) # roll pitch yaw
    
    rot_matrix_next = tfg.geometry.transformation.rotation_matrix_3d.from_euler(trajectory["observation"]["state"][1:, 3:6] ) # roll pitch yaw
    rot_delta_matrix = traj_transforms.calculate_delta_transform(rot_matrix, rot_matrix_next)
    # dR = traj_transforms.nx4x4_matrix_to_euler_xyz(rot_delta_matrix)
    dR = tfg.geometry.transformation.euler.from_rotation_matrix(rot_delta_matrix)
 
    dR = tf.cast(dR, dtype=tf.float32)
    dt = trajectory["observation"]["state"][1:, :3] - trajectory["observation"]["state"][:-1, :3]
    tmp_proprio = trajectory["observation"]["state"]  
    trajectory = tf.nest.map_structure(lambda x: x[:-1], trajectory) 
    trajectory["observation"]["proprio"] = tmp_proprio[1:]
    trajectory["action"] = tf.concat(
        (
            dt,
            dR,
            invert_gripper_actions(trajectory["action"][:, -1:]),
        ),
        axis=-1,
    )
    trajectory["action"] = tf.cast(trajectory["action"], tf.float32)
    trajectory["observation"]["state"] = tf.cast(trajectory["observation"]["state"], tf.float32)
    trajectory["observation"]["proprio"] = tf.cast(trajectory["observation"]["proprio"], tf.float32)
        


    return trajectory


def asu_table_top_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["ground_truth_states"]["EE"]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    return trajectory


def robocook_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    return trajectory


def imperial_wristcam_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., :-1]
    return trajectory


def iamlab_pick_insert_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    import tensorflow_graphics.geometry.transformation as tft

    trajectory["observation"]["joint_state"] = trajectory["observation"]["state"][:, :7]

    trajectory["observation"]["joint_state"] = joints_to_pose_batch_tf(trajectory["observation"]["joint_state"])

    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, 7:8]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tft.euler.from_quaternion(trajectory["action"][:, 3:7]),
            trajectory["action"][:, 7:8],
        ),
        axis=-1,
    )
    return trajectory


def uiuc_d3field_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"],
            tf.zeros_like(trajectory["action"]),
            tf.zeros_like(trajectory["action"][:, :1]),
        ),
        axis=-1,
    )
    return trajectory


def utaustin_mutex_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # trajectory["observation"]["state"] = trajectory["observation"]["state"][:, :8]
    import tensorflow_graphics as tfg
    dR = tfg.geometry.transformation.euler.from_rotation_matrix(tf.reshape(trajectory["observation"]['state'][:, 8:], [-1, 4, 4])[:, :3, :3])
    dt = tf.reshape(trajectory["observation"]['state'][:, 8:], [-1, 4, 4])[:, 3, :3]
    # tf.print(dt, 'dt')
    trajectory["observation"]["state"] = tf.concat([ dt,dR, trajectory["observation"]['state'][:, 7:8]], axis=-1)

#             'state': Tensor(shape=(24,), dtype=float32, description=Robot state, consists of [7x robot joint angles, 1x gripper position, 16x robot end-effector homogeneous matrix].),
    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)),
        ),
        axis=-1,
    )

    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["language_instruction"]), ""
    # )  # delete uninformative language instruction
    return trajectory


def berkeley_fanuc_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["joint_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, 6:7]

    # dataset does not store gripper actions, so use gripper state info, invert so +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"],
            invert_gripper_actions(trajectory["observation"]["gripper_state"]),
        ),
        axis=-1,
    )
    return trajectory


def berkeley_fanuc_dataset_transform_state(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, 6:7]

    import tensorflow_graphics as tfg
    from openvla.prismatic.vla.datasets.rlds import traj_transforms
    stride = 3

    rot_matrix = tfg.geometry.transformation.rotation_matrix_3d.from_euler(trajectory["observation"]["state"][:-1, 3:6]) # roll pitch yaw
    
    rot_matrix_next = tfg.geometry.transformation.rotation_matrix_3d.from_euler(trajectory["observation"]["state"][1:, 3:6] ) # roll pitch yaw
    rot_delta_matrix = traj_transforms.calculate_delta_transform(rot_matrix, rot_matrix_next)
    # dR = traj_transforms.nx4x4_matrix_to_euler_xyz(rot_delta_matrix)
    dR = tfg.geometry.transformation.euler.from_rotation_matrix(rot_delta_matrix)
 
    dR = tf.cast(dR, dtype=tf.float32)
    dt = trajectory["observation"]["state"][1:, :3] - trajectory["observation"]["state"][:-1, :3]
    
    tmp_proprio = tf.concat(
        (
            trajectory["observation"]["state"][:, :6],
            invert_gripper_actions(trajectory["observation"]["state"][:, 6:7]),
        ),
        axis=-1,
    )
    trajectory["observation"]["state"] = tmp_proprio
    trajectory = tf.nest.map_structure(lambda x: x[:-1], trajectory)
    trajectory["observation"]["proprio"] = tmp_proprio[1:]
    # tf.print(tf.shape(trajectory["observation"]["state"]), tf.shape(dt), tf.shape(trajectory["observation"]["gripper_state"]))
    trajectory["action"] = tf.concat(
        (
            dt,
            dR,
            invert_gripper_actions(trajectory["observation"]["gripper_state"]),
        ),
        axis=-1,
    )
    trajectory["action"] = tf.cast(trajectory["action"], tf.float32)
    trajectory["observation"]["state"] = tf.cast(trajectory["observation"]["state"], tf.float32)
    trajectory["observation"]["proprio"] = tf.cast(trajectory["observation"]["proprio"], tf.float32)
            

    # trajectory["observation"]["image"] = trajectory["observation"]["image"]
    # trajectory["observation"]["wrist_image"] = trajectory["observation"]["wrist_image"]
    # trajectory["observation"]["state"] = trajectory["observation"]["state"]


    return trajectory


def cmu_playing_with_food_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    import tensorflow_graphics.geometry.transformation as tft

    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tft.euler.from_quaternion(trajectory["action"][:, 3:7]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def playfusion_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            trajectory["action"][:, -4:],
        ),
        axis=-1,
    )
    return trajectory


def cmu_stretch_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :3],
            tf.zeros_like(trajectory["observation"]["state"][:, :3]),
        ),
        axis=-1,
    )
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    trajectory["action"] = trajectory["action"][..., :-1]
    return trajectory


def gnm_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["state"] = tf.concat(
        (
            trajectory["observation"]["position"],
            tf.zeros_like(trajectory["observation"]["state"][:, :3]),
            trajectory["observation"]["yaw"],
        ),
        axis=-1,
    )
    trajectory["action"] = tf.concat(
        (
            trajectory["action"],
            tf.zeros_like(trajectory["action"]),
            tf.zeros_like(trajectory["action"]),
            tf.zeros_like(trajectory["action"][:, :1]),
        ),
        axis=-1,
    )
    return trajectory


def fmb_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["eef_pose"][:, :3],
            tft.euler.from_quaternion(trajectory['observation']['eef_pose'][:, 3:7]),
            trajectory["observation"]["state_gripper_pose"][..., None],
        ),
        axis=-1,
    )
    return trajectory


def dobbe_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    return trajectory

##############################################################################################
def joints_to_pose_batch_tf(joints_batch):
    """
    Apply forward kinematics method for franka panda robot.

    Args:
    joints_batch (tf.Tensor): Tensor of shape (batch_size, 7) with target joints values

    Returns:
    tf.Tensor: Tensor of shape (batch_size, 7) with target eef poses in the format (x, y, z, w, i, j, k).
    """
    assert joints_batch.shape[1] == 7, "Each element in the batch must have 7 joint values"
    import tensorflow as tf
    import numpy as np
    def mat2euler(matrix):
        """
        Convert rotation matrices to Euler angles (XYZ order).

        Args:
        matrix (tf.Tensor): Tensor of shape (batch_size, 3, 3) representing rotation matrices.

        Returns:
        tf.Tensor: Tensor of shape (batch_size, 3) representing Euler angles (yaw, pitch, roll).
        """
        batch_size = tf.shape(matrix)[0]
        
        # Calculate yaw (Z)
        yaw = tf.atan2(matrix[:, 1, 0], matrix[:, 0, 0])
        
        # Calculate pitch (Y)
        pitch = tf.atan2(-matrix[:, 2, 0], tf.sqrt(tf.square(matrix[:, 2, 1]) + tf.square(matrix[:, 2, 2])))
        
        # Calculate roll (X)
        roll = tf.atan2(matrix[:, 2, 1], matrix[:, 2, 2])
        
        return tf.stack([roll, pitch, yaw], axis=1)

    
    
    def dh_params(joints_batch):
        M_PI = np.pi

        batch_size = tf.shape(joints_batch)[0]
        zeros = tf.zeros([batch_size], dtype=tf.float32)
        ones = tf.ones([batch_size], dtype=tf.float32)

        dh = tf.stack(
            [
                tf.stack([zeros, zeros, 0.333 * ones, joints_batch[:, 0]], axis=1),
                tf.stack([-M_PI / 2 * ones, zeros, zeros, joints_batch[:, 1]], axis=1),
                tf.stack([M_PI / 2 * ones, zeros, 0.316 * ones, joints_batch[:, 2]], axis=1),
                tf.stack([M_PI / 2 * ones, 0.0825 * ones, zeros, joints_batch[:, 3]], axis=1),
                tf.stack([-M_PI / 2 * ones, -0.0825 * ones, 0.384 * ones, joints_batch[:, 4]], axis=1),
                tf.stack([M_PI / 2 * ones, zeros, zeros, joints_batch[:, 5]], axis=1),
                tf.stack([M_PI / 2 * ones, 0.088 * ones, 0.107 * ones, joints_batch[:, 6]], axis=1),
            ], axis=1
        )
        return dh

    def TF_matrix(i, dh):
        # Define Transformation matrix based on DH params
        alpha = dh[:, i, 0]
        a = dh[:, i, 1]
        d = dh[:, i, 2]
        q = dh[:, i, 3]

        TF = tf.stack(
            [
                tf.stack([tf.cos(q), -tf.sin(q), tf.zeros_like(q), a], axis=1),
                tf.stack([tf.sin(q) * tf.cos(alpha), tf.cos(q) * tf.cos(alpha), -tf.sin(alpha), -tf.sin(alpha) * d], axis=1),
                tf.stack([tf.sin(q) * tf.sin(alpha), tf.cos(q) * tf.sin(alpha), tf.cos(alpha), tf.cos(alpha) * d], axis=1),
                tf.stack([tf.zeros_like(q), tf.zeros_like(q), tf.zeros_like(q), tf.ones_like(q)], axis=1)
            ], axis=1
        )
        return TF
    dh_parameters = dh_params(joints_batch)

    T_01 = TF_matrix(0, dh_parameters)
    T_12 = TF_matrix(1, dh_parameters)
    T_23 = TF_matrix(2, dh_parameters)
    T_34 = TF_matrix(3, dh_parameters)
    T_45 = TF_matrix(4, dh_parameters)
    T_56 = TF_matrix(5, dh_parameters)
    T_67 = TF_matrix(6, dh_parameters)

    T_07 = tf.linalg.matmul(T_01, T_12)
    T_07 = tf.linalg.matmul(T_07, T_23)
    T_07 = tf.linalg.matmul(T_07, T_34)
    T_07 = tf.linalg.matmul(T_07, T_45)
    T_07 = tf.linalg.matmul(T_07, T_56)
    T_07 = tf.linalg.matmul(T_07, T_67)

    translation = T_07[:, :3, 3]
    
    euler = mat2euler(T_07[:, :3, :3])
    return tf.concat([translation, euler], axis=1)


def rand_swap_exterior_images3(img1, img2, img3):
    """
    Randomly swaps the two exterior images (for training with single exterior input).
    """
    ccc = tf.random.uniform(shape=[])
    return tf.cond(ccc > 0.33, lambda: tf.cond(ccc > 0.33, lambda: (img1, img2, img3), lambda: (img2, img1, img3)), lambda: (img3, img2, img1))

def rand_swap_exterior_images4(img1, img2, img3, img4):
    """
    Randomly swaps the two exterior images (for training with single exterior input).
    """
    ccc = tf.random.uniform(shape=[])
    return tf.cond(ccc > 0.25, lambda: tf.cond(ccc > 0.5, lambda:  tf.cond(ccc > 0.75, lambda: (img1, img2, img3, img4), lambda: (img2, img3, img4, img1)), lambda: (img3, img4, img1, img2)), lambda: (img4, img1, img2, img3))


def roboset_dataset_transform_state(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension

    trajectory["observation"]["proprio"] = joints_to_pose_batch_tf(trajectory["observation"]["state"][...,:7])
    # stride 1
    # array([-0.03571236, -0.02748381, -0.03519425, -0.10359032, -0.08635712,
    #    -0.08265033,  0.        ]), 'q99': array([0.02108428, 0.02454048, 0.04115295, 0.11795005, 0.08118542,
    #    0.07721102, 1.        ])

    # stride 3
    # 'q01': array([-0.10639569, -0.08045862, -0.10174164, -0.29653627, -0.22435432,
    #    -0.24011278,  0.        ]), 'q99': array([0.05957478, 0.07090826, 0.120134  , 0.33000916, 0.1904533 ,
    #    0.21813695, 1.        ])

    # stride 2
    # 'q01': array([-0.06835699, -0.05336827, -0.06934792, -0.20084344, -0.16507584,
    #    -0.16242206,  0.        ]), 'q99': array([0.04127821, 0.04817784, 0.08139336, 0.22930118, 0.15800564,
    #    0.14983632, 1.        ]),

    # droid 3
    # 'q01': array([-0.03280808, -0.04527937, -0.04277795, -0.10660763, -0.1169551 ,
    #    -0.15010694,  0.        ]), 'q99': array([0.03989471, 0.04507686, 0.04635656, 0.10849163, 0.10751813,
    #    0.15049884, 1.        ]),
    # gripper action is in -1...1 --> clip to 0...1, flip
    gripper_action = trajectory["action"][:, -1:]
    gripper_action = invert_gripper_actions(tf.clip_by_value(gripper_action, 0, 1))

    import tensorflow_graphics as tfg
    from openvla.prismatic.vla.datasets.rlds import traj_transforms
    stride = 1

    rot_matrix = tfg.geometry.transformation.rotation_matrix_3d.from_euler(trajectory["observation"]["proprio"][:-stride:stride,3:6]) # roll pitch yaw
    
    rot_matrix_next = tfg.geometry.transformation.rotation_matrix_3d.from_euler(trajectory["observation"]["proprio"][stride::stride,3:6], ) # roll pitch yaw
    rot_delta_matrix = traj_transforms.calculate_delta_transform(rot_matrix, rot_matrix_next)
    # dR = traj_transforms.nx4x4_matrix_to_euler_xyz(rot_delta_matrix)
    dR = tfg.geometry.transformation.euler.from_rotation_matrix(rot_delta_matrix)

    # tf.print('3', trajectory["action_dict"]['cartesian_position'][:-stride:stride,3][0], trajectory["action_dict"]['cartesian_position'][stride::stride,3][0])
    # tf.print(trajectory["action_dict"]['cartesian_position'][:-stride:stride,4][0], trajectory["action_dict"]['cartesian_position'][stride::stride,4][0])
    # tf.print(trajectory["action_dict"]['cartesian_position'][:-stride:stride,5][0], trajectory["action_dict"]['cartesian_position'][stride::stride,5][0])
    dR = tf.cast(dR, dtype=tf.float32)
    # tf.print(dR[0])
    dt = trajectory["observation"]["proprio"][stride::stride, :3] - trajectory["observation"]["proprio"][:-stride:stride, :3]
    # trajectory['action'] = tf.concat([trans_delta, rot_delta_euler, trajectory['action'][5:6]])
    
    # dt1 = trajectory["action_dict"]["cartesian_velocity"][:, :3]
    # dR1 = trajectory["action_dict"]["cartesian_velocity"][:, 3:6]
    # tf.print(dR, [tf.shape(dR), tf.shape(dt), tf.shape(dt1), tf.shape(dR1)])
    # print(dR, dt)
    # tf.print(gripper_action)
    
    tmp_proprio = trajectory["observation"]["proprio"]
    trajectory["observation"]["state"] = trajectory["observation"]["proprio"]
    trajectory = tf.nest.map_structure(lambda x: x[:-1], trajectory)
    trajectory["observation"]["proprio"] = tmp_proprio[1:]
    trajectory["observation"]["proprio"] = tf.concat(
        (
            tmp_proprio[1:],
            gripper_action[stride::stride],
        ),
        axis=-1,
    )
    trajectory["observation"]["state"] = tf.concat(
        (
            trajectory["observation"]["state"],
            gripper_action[stride::stride],
        ),
        axis=-1,
    )
    trajectory["action"] = tf.concat(
        (
            dt,
            dR,
            gripper_action[stride::stride],
        ),
        axis=-1,
    )

    # trajectory["observation"]["proprio"] = proprio
    # print(trajectory.keys()) dict_keys(['language_instruction_2', 'action_dict', 
    # 'language_instruction', 'action', 'language_instruction_3', 'is_terminal', 'is_first', 'is_last', 
    # 'reward', 'observation', 'discount', 'traj_metadata', '_le
# n', '_traj_index', '_frame_index'])
    # trajectory.pop('action_dict')
    # trajectory.pop('language_instruction_3')
    trajectory.pop('_frame_index')
    trajectory.pop('_traj_index')
    trajectory.pop('_len')
    trajectory.pop('is_terminal')
    trajectory.pop('is_first')
    trajectory.pop('is_last')
    trajectory.pop('reward')
    trajectory.pop('discount')
    # trajectory.pop('traj_metadata')
    trajectory['observation'].pop('state_velocity')
    # trajectory['observation'].pop('state')
    # trajectory['traj_metadata'] = trajectory['traj_metadata'][:-stride:stride]
    # dict_keys(['image_left', 'image_wrist', 'state', 'image_top', 'image_right', 'state_velocity'])
    # trajectory.pop('language_instruction_2')
    trajectory['observation']['image_wrist'] = trajectory['observation']['image_wrist']
    # trajectory['observation']['image_top'] = trajectory['observation']['image_top'][:-stride:stride]
    

# if not item['traj_metadata']['episode_metadata']['file_path'][0].decode().split('/')[-2].__contains__('scene') \
#                         and not item['traj_metadata']['episode_metadata']['file_path'][0].decode().split('/')[-1].__contains__('scene'):
#                         tag = item['traj_metadata']['episode_metadata']['file_path'][0].decode().split('/')[-2] + item['traj_metadata']['episode_metadata']['file_path'][0].decode().split('/')[-1]
#                         Image.fromarray(tf.image.decode_jpeg(item['observation']['image_right'][0], channels=3).numpy()).save('./temp/img_{}_{}_{}_{}.png'.format(i, item['language_instruction'][0].decode(), item['traj_metadata']['episode_metadata']['trial_id'][0].decode(),tag))
#                         Image.fromarray(tf.image.decode_jpeg(item['observation']['image_left'][0], channels=3).numpy()).save('./temp/imgleft_{}_{}_{}_{}.png'.format(i, item['language_instruction'][0].decode(), item['traj_metadata']['episode_metadata']['trial_id'][0].decode(),tag))
#                         Image.fromarray(tf.image.decode_jpeg(item['observation']['image_top'][0], channels=3).numpy()).save('./temp/imgtop_{}_{}_{}_{}.png'.format(i, item['language_instruction'][0].decode(), item['traj_metadata']['episode_metadata']['trial_id'][0].decode(),tag))
        # 检查 description 字符串是否包含 "scene"
    # is_scene = tf.strings.regex_full_match(description, ".*scene.*")
    
    # # 定义旋转函数
    # def rotate_image():
    #     return tf.image.rot90(sample['image'], k=1)  # 旋转90度
    
    # # 如果包含 "scene" 就旋转图像，否则不做处理
    # image = tf.cond(is_scene, lambda: tf.image.rot90(sample['image'], k=1), lambda: sample['image'])
    # is_scene = tf.strings.regex_full_match(trajectory['traj_metadata']['episode_metadata']['file_path'][0], ".*scene.*")
    # # import ipdb;ipdb.set_trace()
    # # import ipdb;ipdb.set_trace()
    # func1 = lambda x: tf.io.decode_image(x, expand_animations=False, dtype=tf.uint8)
    # trajectory['observation']['image_right'] = tf.map_fn(func1, trajectory['observation']['image_right'][:-stride:stride], dtype=tf.uint8)
    # trajectory['observation']['image_left'] = tf.map_fn(func1, trajectory['observation']['image_left'][:-stride:stride], dtype=tf.uint8)
    # trajectory['observation']['image_right'] = tf.cond(is_scene, lambda: trajectory['observation']['image_right'], lambda: tf.image.rot90(trajectory['observation']['image_right'], k=3))
    # trajectory['observation']['image_left'] = tf.cond(is_scene, lambda: trajectory['observation']['image_left'], lambda: tf.image.rot90(trajectory['observation']['image_left'], k=1))

    trajectory['observation']['image_left'], trajectory['observation']['image_top'], trajectory['observation']['image_right'] = (
        rand_swap_exterior_images3(
            trajectory["observation"]["image_left"],
            trajectory['observation']['image_top'],
            trajectory["observation"]["image_right"],
        )
    )
    # item['observation'].keys()
    # dict_keys(['image_left', 'image_wrist', 'state', 'image_top', 'image_right', 'state_velocity'])
    # dict_keys(['is_first', 'language_instruction', 'discount', 'observation', 'is_terminal', 'is_last', 'reward', 'action', 'traj_metadata', '_len', '_traj_index', '_frame_index'])
    trajectory['language_instruction'] = trajectory['language_instruction']
#    tf.print(dR, [trajectory['language_instruction'], tf.shape(trajectory["action"]), tf.shape(trajectory["observation"]["exterior_image_1_left"]), 
#                   tf.shape(trajectory["observation"]["exterior_image_2_left"]), tf.shape(trajectory["observation"]["proprio"])])

    return trajectory


def roboset_dataset_transform_state1(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension

    trajectory["observation"]["proprio"] = joints_to_pose_batch_tf(trajectory["observation"]["state"][...,:7])

    # gripper action is in -1...1 --> clip to 0...1, flip
    gripper_action = trajectory["action"][:, -1:]
    gripper_action = invert_gripper_actions(tf.clip_by_value(gripper_action, 0, 1))

    import tensorflow_graphics as tfg
    from openvla.prismatic.vla.datasets.rlds import traj_transforms
    stride = 1

    rot_matrix = tfg.geometry.transformation.rotation_matrix_3d.from_euler(trajectory["observation"]["proprio"][:-stride:stride,3:6]) # roll pitch yaw
    
    rot_matrix_next = tfg.geometry.transformation.rotation_matrix_3d.from_euler(trajectory["observation"]["proprio"][stride::stride,3:6], ) # roll pitch yaw
    rot_delta_matrix = traj_transforms.calculate_delta_transform(rot_matrix, rot_matrix_next)
    # dR = traj_transforms.nx4x4_matrix_to_euler_xyz(rot_delta_matrix)
    dR = tfg.geometry.transformation.euler.from_rotation_matrix(rot_delta_matrix)

    # tf.print('3', trajectory["action_dict"]['cartesian_position'][:-stride:stride,3][0], trajectory["action_dict"]['cartesian_position'][stride::stride,3][0])
    # tf.print(trajectory["action_dict"]['cartesian_position'][:-stride:stride,4][0], trajectory["action_dict"]['cartesian_position'][stride::stride,4][0])
    # tf.print(trajectory["action_dict"]['cartesian_position'][:-stride:stride,5][0], trajectory["action_dict"]['cartesian_position'][stride::stride,5][0])
    dR = tf.cast(dR, dtype=tf.float32)
    # tf.print(dR[0])
    dt = trajectory["observation"]["proprio"][stride::stride, :3] - trajectory["observation"]["proprio"][:-stride:stride, :3]
    # trajectory['action'] = tf.concat([trans_delta, rot_delta_euler, trajectory['action'][5:6]])
    
    # dt1 = trajectory["action_dict"]["cartesian_velocity"][:, :3]
    # dR1 = trajectory["action_dict"]["cartesian_velocity"][:, 3:6]
    # tf.print(dR, [tf.shape(dR), tf.shape(dt), tf.shape(dt1), tf.shape(dR1)])
    # print(dR, dt)
    # tf.print(gripper_action)
    
    tmp_proprio = trajectory["observation"]["proprio"]
    trajectory["observation"]["state"] = trajectory["observation"]["proprio"]
    trajectory = tf.nest.map_structure(lambda x: x[:-1], trajectory)
    trajectory["observation"]["proprio"] = tmp_proprio[1:]
    trajectory["observation"]["proprio"] = tf.concat(
        (
            tmp_proprio[1:],
            gripper_action[stride::stride],
        ),
        axis=-1,
    )
    trajectory["observation"]["state"] = tf.concat(
        (
            trajectory["observation"]["state"],
            gripper_action[stride::stride],
        ),
        axis=-1,
    )
    trajectory["action"] = tf.concat(
        (
            dt,
            dR,
            gripper_action[stride::stride],
        ),
        axis=-1,
    )

    # trajectory["observation"]["proprio"] = proprio
    # print(trajectory.keys()) dict_keys(['language_instruction_2', 'action_dict', 
    # 'language_instruction', 'action', 'language_instruction_3', 'is_terminal', 'is_first', 'is_last', 
    # 'reward', 'observation', 'discount', 'traj_metadata', '_le
# n', '_traj_index', '_frame_index'])
    # trajectory.pop('action_dict')
    # trajectory.pop('language_instruction_3')
    trajectory.pop('_frame_index')
    trajectory.pop('_traj_index')
    trajectory.pop('_len')
    trajectory.pop('is_terminal')
    trajectory.pop('is_first')
    trajectory.pop('is_last')
    trajectory.pop('reward')
    trajectory.pop('discount')
    # trajectory.pop('traj_metadata')
    trajectory['observation'].pop('state_velocity')
    # trajectory['observation'].pop('state')
    # trajectory['traj_metadata'] = trajectory['traj_metadata'][:-stride:stride]
    # dict_keys(['image_left', 'image_wrist', 'state', 'image_top', 'image_right', 'state_velocity'])
    # trajectory.pop('language_instruction_2')
    trajectory["action"] = tf.cast(trajectory["action"], tf.float32)
    trajectory["observation"]["state"] = tf.cast(trajectory["observation"]["state"], tf.float32)
    trajectory["observation"]["proprio"] = tf.cast(trajectory["observation"]["proprio"], tf.float32)
        
    # trajectory['observation']['image_wrist'] = trajectory['observation']['image_wrist']
    # trajectory['observation']['image_top'] = trajectory['observation']['image_top'][:-stride:stride]

    trajectory['observation']['image_left'], trajectory['observation']['image_top'], trajectory['observation']['image_right'] = (
        rand_swap_exterior_images3(
            trajectory["observation"]["image_left"],
            trajectory['observation']['image_top'],
            trajectory["observation"]["image_right"],
        )
    )
    
    trajectory['language_instruction'] = trajectory['language_instruction']

    return trajectory
##########################################################################################
def roboset_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]

    # gripper action is in -1...1 --> clip to 0...1, flip
    gripper_action = trajectory["action"][:, -1:]
    gripper_action = invert_gripper_actions(tf.clip_by_value(gripper_action, 0, 1))

    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :7],
            gripper_action,
        ),
        axis=-1,
    )
    return trajectory


def rh20t_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["tcp_base"],
            tf.cast(trajectory["action"]["gripper"][:, None], tf.float32),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["tcp_base"],
            trajectory["observation"]["gripper_width"][..., None],
        ),
        axis=-1,
    )
    return trajectory


def tdroid_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            binarize_gripper_actions(trajectory["action"][:, -1])[:, None],
        ],
        axis=1,
    )
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["cartesian_position"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["gripper_position"][:, -1:]
    return trajectory


# === Registry ===
OXE_STANDARDIZATION_TRANSFORMS = {
    "bridge_oxe": bridge_oxe_dataset_transform,
    "bridge_orig": bridge_orig_dataset_transform,
    "bridge_orig_state": bridge_orig_dataset_transform_state,
    "bridge_dataset": bridge_orig_dataset_transform,
    "ppgm": ppgm_dataset_transform,
    "ppgm_static": ppgm_dataset_transform,
    "ppgm_wrist": ppgm_dataset_transform,
    "fractal20220817_data": rt1_dataset_transform,
    "kuka": kuka_dataset_transform,
    "kuka_state": kuka_dataset_transform_state,
    "taco_play": taco_play_dataset_transform,
    "jaco_play": jaco_play_dataset_transform,
    "berkeley_cable_routing": berkeley_cable_routing_dataset_transform,
    "roboturk": roboturk_dataset_transform,
    "nyu_door_opening_surprising_effectiveness": nyu_door_opening_dataset_transform,
    "viola": viola_dataset_transform,
    "berkeley_autolab_ur5": berkeley_autolab_ur5_dataset_transform,
    "toto": toto_dataset_transform,
    "language_table": language_table_dataset_transform,
    "columbia_cairlab_pusht_real": pusht_dataset_transform,
    "stanford_kuka_multimodal_dataset_converted_externally_to_rlds": stanford_kuka_multimodal_dataset_transform,
    "nyu_rot_dataset_converted_externally_to_rlds": nyu_rot_dataset_transform,
    "stanford_hydra_dataset_converted_externally_to_rlds": stanford_hydra_dataset_transform,
    "stanford_hydra_dataset_converted_externally_to_rlds_state": stanford_hydra_dataset_transform_state,
    "austin_buds_dataset_converted_externally_to_rlds": austin_buds_dataset_transform,
    "nyu_franka_play_dataset_converted_externally_to_rlds": nyu_franka_play_dataset_transform,
    "nyu_franka_play_dataset_converted_externally_to_rlds_state": nyu_franka_play_dataset_transform_state,
    "maniskill_dataset_converted_externally_to_rlds": maniskill_dataset_transform,
    "furniture_bench_dataset_converted_externally_to_rlds": furniture_bench_dataset_transform,
    "cmu_franka_exploration_dataset_converted_externally_to_rlds": cmu_franka_exploration_dataset_transform,
    "ucsd_kitchen_dataset_converted_externally_to_rlds": ucsd_kitchen_dataset_transform,
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds": ucsd_pick_place_dataset_transform,
    "austin_sailor_dataset_converted_externally_to_rlds": austin_sailor_dataset_transform,
    "austin_sirius_dataset_converted_externally_to_rlds": austin_sirius_dataset_transform,
    "bc_z": bc_z_dataset_transform,
    "utokyo_pr2_opening_fridge_converted_externally_to_rlds": tokyo_pr2_opening_fridge_dataset_transform,
    "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds": tokyo_pr2_tabletop_manipulation_dataset_transform,
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds": utokyo_xarm_pick_place_dataset_transform,
    "utokyo_xarm_bimanual_converted_externally_to_rlds": utokyo_xarm_bimanual_dataset_transform,
    "robo_net": robo_net_dataset_transform,
    "berkeley_mvp_converted_externally_to_rlds": berkeley_mvp_dataset_transform,
    "berkeley_rpt_converted_externally_to_rlds": berkeley_rpt_dataset_transform,
    "kaist_nonprehensile_converted_externally_to_rlds": kaist_nonprehensible_dataset_transform,
    "stanford_mask_vit_converted_externally_to_rlds": stanford_mask_vit_dataset_transform,
    "tokyo_u_lsmo_converted_externally_to_rlds": tokyo_lsmo_dataset_transform,
    "dlr_sara_pour_converted_externally_to_rlds": dlr_sara_pour_dataset_transform,
    "dlr_sara_grid_clamp_converted_externally_to_rlds": dlr_sara_grid_clamp_dataset_transform,
    "dlr_edan_shared_control_converted_externally_to_rlds": dlr_edan_shared_control_dataset_transform,
    "dlr_edan_shared_control_converted_externally_to_rlds_state": dlr_edan_shared_control_dataset_transform_state,
    "asu_table_top_converted_externally_to_rlds": asu_table_top_dataset_transform,
    "stanford_robocook_converted_externally_to_rlds": robocook_dataset_transform,
    "imperialcollege_sawyer_wrist_cam": imperial_wristcam_dataset_transform,
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds": iamlab_pick_insert_dataset_transform,
    "uiuc_d3field": uiuc_d3field_dataset_transform,
    "utaustin_mutex": utaustin_mutex_dataset_transform,
    "berkeley_fanuc_manipulation": berkeley_fanuc_dataset_transform,
    "berkeley_fanuc_manipulation_state": berkeley_fanuc_dataset_transform_state,
    "cmu_playing_with_food": cmu_playing_with_food_dataset_transform,
    "cmu_play_fusion": playfusion_dataset_transform,
    "cmu_stretch": cmu_stretch_dataset_transform,
    "berkeley_gnm_recon": gnm_dataset_transform,
    "berkeley_gnm_cory_hall": gnm_dataset_transform,
    "berkeley_gnm_sac_son": gnm_dataset_transform,
    "droid": droid_baseact_transform_delta_state,
    "droid_state": droid_baseact_transform_delta_state1,
    "fmb_dataset": fmb_dataset_transform,
    "fmb": fmb_dataset_transform,
    "dobbe": dobbe_dataset_transform,
    "roboset": roboset_dataset_transform,
    "robo_set": roboset_dataset_transform_state,
    "robo_set_state": roboset_dataset_transform_state1,
    "rh20t": rh20t_dataset_transform,
    ### T-DROID datasets
    "tdroid_carrot_in_bowl": tdroid_dataset_transform,
    "tdroid_pour_corn_in_pot": tdroid_dataset_transform,
    "tdroid_flip_pot_upright": tdroid_dataset_transform,
    "tdroid_move_object_onto_plate": tdroid_dataset_transform,
    "tdroid_knock_object_over": tdroid_dataset_transform,
    "tdroid_cover_object_with_towel": tdroid_dataset_transform,
    ### DROID Finetuning datasets
    "droid_wipe": droid_finetuning_transform,
    ### Calvin Finetuning datasets
    "Calvin_wipe": Calvin_finetuning_transform,
}
