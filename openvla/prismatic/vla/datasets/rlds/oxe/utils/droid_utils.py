"""Episode transforms for DROID dataset."""

from typing import Any, Dict

import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfg


def rmat_to_euler(rot_mat):
    return tfg.euler.from_rotation_matrix(rot_mat)


def euler_to_rmat(euler):
    return tfg.rotation_matrix_3d.from_euler(euler)


def invert_rmat(rot_mat):
    return tfg.rotation_matrix_3d.inverse(rot_mat)


def rotmat_to_rot6d(mat):
    """
    Converts rotation matrix to R6 rotation representation (first two rows in rotation matrix).
    Args:
        mat: rotation matrix

    Returns: 6d vector (first two rows of rotation matrix)

    """
    r6 = mat[..., :2, :]
    r6_0, r6_1 = r6[..., 0, :], r6[..., 1, :]
    r6_flat = tf.concat([r6_0, r6_1], axis=-1)
    return r6_flat


def velocity_act_to_wrist_frame(velocity, wrist_in_robot_frame):
    """
    Translates velocity actions (translation + rotation) from base frame of the robot to wrist frame.
    Args:
        velocity: 6d velocity action (3 x translation, 3 x rotation)
        wrist_in_robot_frame: 6d pose of the end-effector in robot base frame

    Returns: 9d velocity action in robot wrist frame (3 x translation, 6 x rotation as R6)

    """
    R_frame = euler_to_rmat(wrist_in_robot_frame[:, 3:6])
    R_frame_inv = invert_rmat(R_frame)

    # world to wrist: dT_pi = R^-1 dT_rbt
    vel_t = (R_frame_inv @ velocity[:, :3][..., None])[..., 0]

    # world to wrist: dR_pi = R^-1 dR_rbt R
    dR = euler_to_rmat(velocity[:, 3:6])
    dR = R_frame_inv @ (dR @ R_frame)
    dR_r6 = rotmat_to_rot6d(dR)
    return tf.concat([vel_t, dR_r6], axis=-1)


def rand_swap_exterior_images(img1, img2):
    """
    Randomly swaps the two exterior images (for training with single exterior input).
    """
    return tf.cond(tf.random.uniform(shape=[]) > 0.5, lambda: (img1, img2), lambda: (img2, img1))




def droid_baseact_transform_delta_state(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    import tensorflow_graphics as tfg
    from openvla.prismatic.vla.datasets.rlds import traj_transforms
    stride = 3
    start = 0

    # print(trajectory.keys()) dict_keys(['language_instruction_2', 'action_dict', 
    # 'language_instruction', 'action', 'language_instruction_3', 'is_terminal', 'is_first', 'is_last', 
    # 'reward', 'observation', 'discount', 'traj_metadata', '_le
# n', '_traj_index', '_frame_index'])
    

    action_list = []
    img1_list = []
    img2_list = []
    proprio_list= []
    language_instruction = []
    _traj_index_list = []
    _frame_index_list = []
    idx1_list = []
    idx2_list = []
    for start in range(0, 1):
        rot_matrix = tfg.geometry.transformation.rotation_matrix_3d.from_euler(trajectory["action_dict"]['cartesian_position'][start:-stride-start,3:6]) # roll pitch yaw
        
        rot_matrix_next = tfg.geometry.transformation.rotation_matrix_3d.from_euler(trajectory["action_dict"]['cartesian_position'][start+stride:,3:6], ) # roll pitch yaw
        rot_delta_matrix = traj_transforms.calculate_delta_transform(rot_matrix, rot_matrix_next)
        dR = tfg.geometry.transformation.euler.from_rotation_matrix(rot_delta_matrix)

        dR = tf.cast(dR, dtype=tf.float64)
        # tf.print(dR[0])
        dt = trajectory["action_dict"]['cartesian_position'][start+stride:, :3] - trajectory["action_dict"]['cartesian_position'][start:-stride-start, :3]
        
        # print(dR, dt)
        
        action_list.append(tf.concat(
            (
                dt,
                dR,
                1 - trajectory["action_dict"]["gripper_position"][start+stride:],
            ),
            axis=-1,
        ))
        item1, item2 = (
            rand_swap_exterior_images(
                (trajectory["observation"]["exterior_image_1_left"][start:-stride-start], tf.zeros_like(dt[:, :1])),
                (trajectory["observation"]["exterior_image_2_left"][start:-stride-start], tf.ones_like(dt[:, :1])),
            )
        )
        img1, idx1 = item1 
        img2, idx2 = item2
        img1_list.append(img1)
        img2_list.append(img2)
        proprio_list.append(tf.concat(
            (
                trajectory["observation"]["cartesian_position"][start:-stride-start+1],
                trajectory["observation"]["gripper_position"][start:-stride-start+1],
            ),
            axis=-1,
        ))

        
        language_instruction.append(trajectory['language_instruction'][start:-stride-start])
        _traj_index_list.append(trajectory['_traj_index'][start:-stride-start])
        _frame_index_list.append(trajectory['_frame_index'][start:-stride-start])
        idx1_list.append(idx1)
        idx2_list.append(idx2)

    trajectory['language_instruction'] = tf.concat(language_instruction, axis=0)
    trajectory["observation"]["proprio"] = tf.concat(proprio_list, axis=0)
    trajectory["observation"]["exterior_image_1_left"] = tf.concat(img1_list, axis=0)
    trajectory["observation"]["exterior_image_2_left"] = tf.concat(img2_list, axis=0)
    trajectory["action"] = tf.concat(action_list, axis=0)
    # trajectory['traj_index'] = tf.concat(_traj_index_list, axis=0)
    # trajectory['frame_index'] = tf.concat(_frame_index_list, axis=0)
    # trajectory['idx_1'] = tf.expand_dims(tf.concat(idx1_list, axis=0), axis=-1)

    # trajectory['idx_2'] = tf.expand_dims(tf.concat(idx2_list, axis=0), axis=-1)

    trajectory.pop('action_dict')
    trajectory.pop('language_instruction_3')
    trajectory.pop('_frame_index')
    trajectory.pop('_traj_index')
    trajectory.pop('_len')
    trajectory.pop('is_terminal')
    trajectory.pop('is_first')
    trajectory.pop('is_last')
    trajectory.pop('reward')
    trajectory.pop('discount')
    trajectory.pop('traj_metadata')
    trajectory.pop('language_instruction_2')

    return trajectory



def droid_baseact_transform_delta_state_with_trajid(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    import tensorflow_graphics as tfg
    from openvla.prismatic.vla.datasets.rlds import traj_transforms
    stride = 3
    start = 0

    # print(trajectory.keys()) dict_keys(['language_instruction_2', 'action_dict', 
    # 'language_instruction', 'action', 'language_instruction_3', 'is_terminal', 'is_first', 'is_last', 
    # 'reward', 'observation', 'discount', 'traj_metadata', '_le
# n', '_traj_index', '_frame_index'])
    

    action_list = []
    img1_list = []
    img2_list = []
    proprio_list= []
    language_instruction = []
    _traj_index_list = []
    _frame_index_list = []
    idx1_list = []
    idx2_list = []
    for start in range(0, 1):
        rot_matrix = tfg.geometry.transformation.rotation_matrix_3d.from_euler(trajectory["action_dict"]['cartesian_position'][start:-stride-start,3:6]) # roll pitch yaw
        
        rot_matrix_next = tfg.geometry.transformation.rotation_matrix_3d.from_euler(trajectory["action_dict"]['cartesian_position'][start+stride:,3:6], ) # roll pitch yaw
        rot_delta_matrix = traj_transforms.calculate_delta_transform(rot_matrix, rot_matrix_next)
        dR = tfg.geometry.transformation.euler.from_rotation_matrix(rot_delta_matrix)

        dR = tf.cast(dR, dtype=tf.float64)
        # tf.print(dR[0])
        dt = trajectory["action_dict"]['cartesian_position'][start+stride:, :3] - trajectory["action_dict"]['cartesian_position'][start:-stride-start, :3]
        
        # print(dR, dt)
        
        action_list.append(tf.concat(
            (
                dt,
                dR,
                1 - trajectory["action_dict"]["gripper_position"][start+stride:],
            ),
            axis=-1,
        ))
        item1, item2 = (
            rand_swap_exterior_images(
                (trajectory["observation"]["exterior_image_1_left"][start:-stride-start], tf.zeros_like(dt[:, :1])),
                (trajectory["observation"]["exterior_image_2_left"][start:-stride-start], tf.ones_like(dt[:, :1])),
            )
        )
        img1, idx1 = item1 
        img2, idx2 = item2
        img1_list.append(img1)
        img2_list.append(img2)
        proprio_list.append(tf.concat(
            (
                trajectory["observation"]["cartesian_position"][start:-stride-start+1],
                trajectory["observation"]["gripper_position"][start:-stride-start+1],
            ),
            axis=-1,
        ))

        
        language_instruction.append(trajectory['language_instruction'][start:-stride-start])
        _traj_index_list.append(trajectory['traj_metadata']['episode_metadata']['recording_folderpath'][start:-stride-start])

        _frame_index_list.append(trajectory['_frame_index'][start:-stride-start])
        idx1_list.append(idx1)
        idx2_list.append(idx2)

    trajectory['language_instruction'] = tf.concat(language_instruction, axis=0)
    trajectory["observation"]["proprio"] = tf.concat(proprio_list, axis=0)
    trajectory["observation"]["exterior_image_1_left"] = tf.concat(img1_list, axis=0)
    trajectory["observation"]["exterior_image_2_left"] = tf.concat(img2_list, axis=0)
    trajectory["action"] = tf.concat(action_list, axis=0)
    trajectory['traj_index'] = tf.concat(_traj_index_list, axis=0)
    # trajectory['frame_index'] = tf.concat(_frame_index_list, axis=0)
    trajectory['idx_1'] = tf.expand_dims(tf.concat(idx1_list, axis=0), axis=-1)

    # trajectory['idx_2'] = tf.expand_dims(tf.concat(idx2_list, axis=0), axis=-1)

    trajectory.pop('action_dict')
    trajectory.pop('language_instruction_3')
    trajectory.pop('_frame_index')
    trajectory.pop('_traj_index')
    trajectory.pop('_len')
    trajectory.pop('is_terminal')
    trajectory.pop('is_first')
    trajectory.pop('is_last')
    trajectory.pop('reward')
    trajectory.pop('discount')
    trajectory.pop('traj_metadata')
    trajectory.pop('language_instruction_2')

    return trajectory


def droid_baseact_transform_delta_state1(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    import tensorflow_graphics as tfg
    from openvla.prismatic.vla.datasets.rlds import traj_transforms
    stride = 3
    start = 0

    # print(trajectory.keys()) dict_keys(['language_instruction_2', 'action_dict', 
    # 'language_instruction', 'action', 'language_instruction_3', 'is_terminal', 'is_first', 'is_last', 
    # 'reward', 'observation', 'discount', 'traj_metadata', '_le
# n', '_traj_index', '_frame_index'])
    

    action_list = []
    img1_list = []
    img2_list = []
    proprio_list= []
    state_list = []
    language_instruction = []

    for start in range(0, 1):

        rot_matrix = tfg.geometry.transformation.rotation_matrix_3d.from_euler(trajectory["action_dict"]['cartesian_position'][start:-stride-start,3:6]) # roll pitch yaw
        
        rot_matrix_next = tfg.geometry.transformation.rotation_matrix_3d.from_euler(trajectory["action_dict"]['cartesian_position'][start+stride:,3:6], ) # roll pitch yaw
        rot_delta_matrix = traj_transforms.calculate_delta_transform(rot_matrix, rot_matrix_next)
        dR = tfg.geometry.transformation.euler.from_rotation_matrix(rot_delta_matrix)

        dR = tf.cast(dR, dtype=tf.float64)
        # tf.print(dR[0])
        dt = trajectory["action_dict"]['cartesian_position'][start+stride:, :3] - trajectory["action_dict"]['cartesian_position'][start:-stride-start, :3]
        
        # print(dR, dt)
        
        action_list.append(tf.concat(
            (
                dt,
                dR,
                1 - trajectory["action_dict"]["gripper_position"][start+stride:],
            ),
            axis=-1,
        ))
        img1, img2 = (
            rand_swap_exterior_images(
                trajectory["observation"]["exterior_image_1_left"][start:-stride-start],
                trajectory["observation"]["exterior_image_2_left"][start:-stride-start],
            )
        )
        img1_list.append(img1)
        img2_list.append(img2)
        proprio_list.append(tf.concat(
            (
                trajectory["observation"]["cartesian_position"][start+stride:],
                trajectory["observation"]["gripper_position"][start+stride:],
            ),
            axis=-1,
        ))
        state_list.append(tf.concat(
            (
                trajectory["observation"]["cartesian_position"][start:-stride-start],
                trajectory["observation"]["gripper_position"][start:-stride-start],
            ),
            axis=-1,
        ))

        
        language_instruction.append(trajectory['language_instruction'][start:-stride-start])
    trajectory['language_instruction'] = tf.concat(language_instruction, axis=0)
    trajectory["observation"]["proprio"] = tf.concat(proprio_list, axis=0)
    trajectory["observation"]["state"] = tf.concat(state_list, axis=0)
    
    trajectory["observation"]["exterior_image_1_left"] = tf.concat(img1_list, axis=0)
    trajectory["observation"]["exterior_image_2_left"] = tf.concat(img2_list, axis=0)
    trajectory["action"] = tf.concat(action_list, axis=0)

    
    trajectory["action"] = tf.cast(trajectory["action"], tf.float32)
    trajectory["observation"]["state"] = tf.cast(trajectory["observation"]["state"], tf.float32)
    trajectory["observation"]["proprio"] = tf.cast(trajectory["observation"]["proprio"], tf.float32)
    
    trajectory.pop('action_dict')
    trajectory.pop('language_instruction_3')
    trajectory.pop('_frame_index')
    trajectory.pop('_traj_index')
    trajectory.pop('_len')
    trajectory.pop('is_terminal')
    trajectory.pop('is_first')
    trajectory.pop('is_last')
    trajectory.pop('reward')
    trajectory.pop('discount')
    trajectory.pop('traj_metadata')
    trajectory.pop('language_instruction_2')
    # tf.print('trans:', trajectory['action'].dtype, trajectory['observation']['proprio'].dtype,trajectory['observation']['state'].dtype,)
    return trajectory

def droid_baseact_transform_delta(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    DROID dataset transformation for actions expressed in *base* frame of the robot.
    """
    stride = 3
    from openvla.prismatic.vla.datasets.rlds import traj_transforms
    rot_matrix = traj_transforms.euler_to_matrix_xyz(trajectory["action_dict"]['cartesian_position'][:-stride:stride,3], 
                                                     trajectory["action_dict"]['cartesian_position'][:-stride:stride,4], 
                                                     trajectory["action_dict"]['cartesian_position'][:-stride:stride,5]) # roll pitch yaw
    
    rot_matrix_next = traj_transforms.euler_to_matrix_xyz(trajectory["action_dict"]['cartesian_position'][stride::stride,3], 
                                                          trajectory["action_dict"]['cartesian_position'][stride::stride,4], 
                                                          trajectory["action_dict"]['cartesian_position'][stride::stride,5]) # roll pitch yaw
    # tf.print('3', trajectory["action_dict"]['cartesian_position'][:-stride:stride,3][0], trajectory["action_dict"]['cartesian_position'][stride::stride,3][0])
    # tf.print(trajectory["action_dict"]['cartesian_position'][:-stride:stride,4][0], trajectory["action_dict"]['cartesian_position'][stride::stride,4][0])
    # tf.print(trajectory["action_dict"]['cartesian_position'][:-stride:stride,5][0], trajectory["action_dict"]['cartesian_position'][stride::stride,5][0])
    rot_delta_matrix = traj_transforms.calculate_delta_transform(rot_matrix, rot_matrix_next)
    dR = traj_transforms.nx4x4_matrix_to_euler_xyz(rot_delta_matrix)
    dR = tf.cast(dR, dtype=tf.float64)
    # tf.print(dR[0])
    dt = trajectory["action_dict"]['cartesian_position'][stride::stride, :3] - trajectory["action_dict"]['cartesian_position'][:-stride:stride, :3]
    # trajectory['action'] = tf.concat([trans_delta, rot_delta_euler, trajectory['action'][5:6]])
    
    # dt1 = trajectory["action_dict"]["cartesian_velocity"][:, :3]
    # dR1 = trajectory["action_dict"]["cartesian_velocity"][:, 3:6]
    # tf.print(dR, [tf.shape(dR), tf.shape(dt), tf.shape(dt1), tf.shape(dR1)])
    # print(dR, dt)
    trajectory["action"] = tf.concat(
        (
            dt,
            dR,
            1 - trajectory["action_dict"]["gripper_position"][stride::stride],
        ),
        axis=-1,
    )
    trajectory["observation"]["exterior_image_1_left"], trajectory["observation"]["exterior_image_2_left"] = (
        rand_swap_exterior_images(
            trajectory["observation"]["exterior_image_1_left"][:-stride:stride],
            trajectory["observation"]["exterior_image_2_left"][:-stride:stride],
        )
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["cartesian_position"][:-stride:stride],
            trajectory["observation"]["gripper_position"][:-stride:stride],
        ),
        axis=-1,
    )
    # print(trajectory.keys()) dict_keys(['language_instruction_2', 'action_dict', 
    # 'language_instruction', 'action', 'language_instruction_3', 'is_terminal', 'is_first', 'is_last', 
    # 'reward', 'observation', 'discount', 'traj_metadata', '_le
# n', '_traj_index', '_frame_index'])
    trajectory.pop('action_dict')
    trajectory.pop('language_instruction_3')
    trajectory.pop('_frame_index')
    trajectory.pop('_traj_index')
    trajectory.pop('_len')
    trajectory.pop('is_terminal')
    trajectory.pop('is_first')
    trajectory.pop('is_last')
    trajectory.pop('reward')
    trajectory.pop('discount')
    trajectory.pop('traj_metadata')
    trajectory.pop('language_instruction_2')

    trajectory['language_instruction'] = trajectory['language_instruction'][:-stride:stride]
#    tf.print(dR, [trajectory['language_instruction'], tf.shape(trajectory["action"]), tf.shape(trajectory["observation"]["exterior_image_1_left"]), 
#                   tf.shape(trajectory["observation"]["exterior_image_2_left"]), tf.shape(trajectory["observation"]["proprio"])])
    return trajectory

def droid_baseact_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    DROID dataset transformation for actions expressed in *base* frame of the robot.
    """
    dt = trajectory["action_dict"]["cartesian_velocity"][:, :3]
    dR = trajectory["action_dict"]["cartesian_velocity"][:, 3:6]
    trajectory["action"] = tf.concat(
        (
            dt,
            dR,
            1 - trajectory["action_dict"]["gripper_position"],
        ),
        axis=-1,
    )
    trajectory["observation"]["exterior_image_1_left"], trajectory["observation"]["exterior_image_2_left"] = (
        rand_swap_exterior_images(
            trajectory["observation"]["exterior_image_1_left"],
            trajectory["observation"]["exterior_image_2_left"],
        )
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["cartesian_position"],
            trajectory["observation"]["gripper_position"],
        ),
        axis=-1,
    )
    return trajectory


def droid_wristact_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    DROID dataset transformation for actions expressed in *wrist* frame of the robot.
    """
    wrist_act = velocity_act_to_wrist_frame(
        trajectory["action_dict"]["cartesian_velocity"], trajectory["observation"]["cartesian_position"]
    )
    trajectory["action"] = tf.concat(
        (
            wrist_act,
            trajectory["action_dict"]["gripper_position"],
        ),
        axis=-1,
    )
    trajectory["observation"]["exterior_image_1_left"], trajectory["observation"]["exterior_image_2_left"] = (
        rand_swap_exterior_images(
            trajectory["observation"]["exterior_image_1_left"],
            trajectory["observation"]["exterior_image_2_left"],
        )
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["cartesian_position"],
            trajectory["observation"]["gripper_position"],
        ),
        axis=-1,
    )
    return trajectory


def droid_finetuning_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    DROID dataset transformation for actions expressed in *base* frame of the robot.
    """
    dt = trajectory["action_dict"]["cartesian_velocity"][:, :3]
    dR = trajectory["action_dict"]["cartesian_velocity"][:, 3:6]
    trajectory["action"] = tf.concat(
        (
            dt,
            dR,
            1 - trajectory["action_dict"]["gripper_position"],
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["cartesian_position"],
            trajectory["observation"]["gripper_position"],
        ),
        axis=-1,
    )
    return trajectory

def Calvin_finetuning_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    DROID dataset transformation for actions expressed in *base* frame of the robot.
    """
    dt = trajectory["action_dict"]["cartesian_velocity"][:, :3]
    dR = trajectory["action_dict"]["cartesian_velocity"][:, 3:6]
    trajectory["action"] = tf.concat(
        (
            dt,
            dR,
            1 - trajectory["action_dict"]["gripper_position"],
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["cartesian_position"],
            trajectory["observation"]["gripper_position"],
        ),
        axis=-1,
    )
    return trajectory


def zero_action_filter(traj: Dict) -> bool:
    """
    Filters transitions whose actions are all-0 (only relative actions, no gripper action).
    Note: this filter is applied *after* action normalization, so need to compare to "normalized 0".
    """
    DROID_Q01 = tf.convert_to_tensor(
        [
            -0.7776297926902771,
            -0.5803514122962952,
            -0.5795090794563293,
            -0.6464047729969025,
            -0.7041108310222626,
            -0.8895104378461838,
        ]
    )
    DROID_Q99 = tf.convert_to_tensor(
        [
            0.7597932070493698,
            0.5726242214441299,
            0.7351000607013702,
            0.6705610305070877,
            0.6464948207139969,
            0.8897542208433151,
        ]
    )
    DROID_NORM_0_ACT = 2 * (tf.zeros_like(traj["action"][:, :6]) - DROID_Q01) / (DROID_Q99 - DROID_Q01 + 1e-8) - 1

    return tf.reduce_any(tf.math.abs(traj["action"][:, :6] - DROID_NORM_0_ACT) > 1e-5)
