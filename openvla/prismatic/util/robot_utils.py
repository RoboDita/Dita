import numpy as np


from rt1_pytorch.openx_utils.geometry import compact_axis_angle_from_quaternion, inv_scale_action
from transforms3d.euler import euler2quat, euler2mat, quat2euler, mat2euler
from transforms3d.quaternions import mat2quat, quat2mat

import mani_skill2.envs
from mani_skill2.agents.base_controller import CombinedController
from mani_skill2.agents.controllers import *

TARGET_CONTROL_MODE = 'pd_ee_delta_pose' # param can be one of ['pd_ee_delta_pose', 'pd_ee_target_delta_pose']

def transform_position_cv(vector, r_t_matrix):
    return (r_t_matrix @ vector.T).T

# equal to camera_quaternion.inv() * base_quaternion
def transform_rotation_cv(mat, r_t_matrix):
    r_matrix = r_t_matrix[:3, :3]
    return r_matrix @ mat

def transform_pose_cv(pose, base_world, r_t_matrix, inv = False):
    import sapien.core as sapien
    if not inv:
        pose = base_world * pose
    else:
        r_t_matrix = np.linalg.inv(r_t_matrix)
    v1 = np.ones((1, 4))
    v1[:, :3] = pose.p
    r1 = quat2mat(pose.q)
    v2 = transform_position_cv(v1, r_t_matrix)
    r2 = transform_rotation_cv(r1, r_t_matrix)
    transformed_pose = sapien.Pose(p=v2[0, :3], q=mat2quat(r2))
    if inv:
        transformed_pose = base_world.inv() * transformed_pose
    return transformed_pose

def transform_pose_to_uv(pose_camera, camera_intrinsic_cv):
    distance = 0.1
    rot_matrix = quat2mat(pose_camera.q)
    camera_vector = np.ones((1, 4))
    camera_vector[:, :3] = pose_camera.p
    x_vector = camera_vector.copy()
    x_vector[:, :3] = pose_camera.p + distance * rot_matrix[:, 0]
    y_vector = camera_vector.copy()
    y_vector[:, :3] = pose_camera.p + distance * rot_matrix[:, 1]
    z_vector = camera_vector.copy()
    z_vector[:, :3] = pose_camera.p + distance * rot_matrix[:, 2]

    for i in range(3):
        camera_vector[:, i] /= camera_vector[:, 3]
        x_vector[:, i] /= x_vector[:, 3]
        y_vector[:, i] /= y_vector[:, 3]
        z_vector[:, i] /= z_vector[:, 3]
    camera_vector = camera_vector[:, :3]
    x_vector = x_vector[:, :3]
    y_vector = y_vector[:, :3]
    z_vector = z_vector[:, :3]

    uv_vector = (camera_intrinsic_cv @ camera_vector.T).T
    uv_vector_dx = (camera_intrinsic_cv @ x_vector.T).T
    uv_vector_dy = (camera_intrinsic_cv @ y_vector.T).T
    uv_vector_dz = (camera_intrinsic_cv @ z_vector.T).T
    for i in range(2):
        uv_vector[:, i] /= uv_vector[:, 2]
        uv_vector_dx[:, i] /= uv_vector_dx[:, 2]
        uv_vector_dy[:, i] /= uv_vector_dy[:, 2]
        uv_vector_dz[:, i] /= uv_vector_dz[:, 2]

    return uv_vector[0], uv_vector_dx[0], uv_vector_dy[0], uv_vector_dz[0]

def cal_delta(prev, target, gripper, method):
    delta_pos = target.p - prev.p
    if method == 0:
        delta_euler = np.array(quat2euler(target.q)) - np.array(quat2euler(prev.q)) # xyz
    elif method == 1:
        r_target = quat2mat(target.q)
        r_prev = quat2mat(prev.q)
        r_diff = r_target @ r_prev.T
        delta_euler = np.array(mat2euler(r_diff))
    else:
        print("cal_delta: invaild method!")

    delta = np.hstack([delta_pos, delta_euler, gripper])
    return delta

def cal_action(env, delta, extrinsic_cv, method):
    import sapien.core as sapien
    assert (len(delta) == 7 and (method == 0 or method == 1))\
            or (len(delta) == 8 and method == 2)
    target_mode = "target" in TARGET_CONTROL_MODE
    controller: CombinedController = env.agent.controller
    arm_controller: PDEEPoseController = controller.controllers["arm"]
    assert arm_controller.config.frame == "ee"
    ee_link: sapien.Link = arm_controller.ee_link
    base_pose = env.agent.robot.pose
    if target_mode:
        prev_ee_pose_at_base = arm_controller._target_pose
    else:
        prev_ee_pose_at_base = base_pose.inv() * ee_link.pose
    prev_ee_pose_at_camera = transform_pose_cv(prev_ee_pose_at_base, base_pose, extrinsic_cv)
    target_ee_pose_at_camera = sapien.Pose(p=prev_ee_pose_at_camera.p + delta[:3])

    if method == 0:
        e_prev = quat2euler(prev_ee_pose_at_camera.q)
        e_target = np.array(e_prev) + delta[3:6]
        target_ee_pose_at_camera.set_q(euler2quat(e_target[0], e_target[1], e_target[2]))
    elif method == 1:
        r_prev = quat2mat(prev_ee_pose_at_camera.q)
        r_diff = euler2mat(delta[3], delta[4], delta[5])
        r_target = r_diff @ r_prev
        target_ee_pose_at_camera.set_q(mat2quat(r_target))
    elif method == 2:
        r_prev = quat2mat(prev_ee_pose_at_camera.q)
        r_diff = quat2mat(delta[3:7])
        r_target = r_diff @ r_prev
        target_ee_pose_at_camera.set_q(mat2quat(r_target))
    else:
        print("cal_action: invaild method!")

    target_ee_pose = transform_pose_cv(target_ee_pose_at_camera, base_pose, extrinsic_cv, True)
    ee_pose_at_ee = prev_ee_pose_at_base.inv() * target_ee_pose
    ee_pose_at_ee = np.r_[
        ee_pose_at_ee.p,
        compact_axis_angle_from_quaternion(ee_pose_at_ee.q),
    ]
    arm_action = inv_scale_action(ee_pose_at_ee, -0.1, 0.1)
    action = np.hstack([arm_action, delta[-1]])
    return action

def eef_pose(env, extrinsic_cv=np.eye(4), camera_coord=False):
    import sapien.core as sapien
    target_mode = "target" in TARGET_CONTROL_MODE
    controller: CombinedController = env.agent.controller
    arm_controller: PDEEPoseController = controller.controllers["arm"]
    assert arm_controller.config.frame == "ee"
    ee_link: sapien.Link = arm_controller.ee_link
    base_pose = env.agent.robot.pose
    if target_mode:
        prev_ee_pose_at_base = arm_controller._target_pose
    else:
        prev_ee_pose_at_base = base_pose.inv() * ee_link.pose
    if not camera_coord:
        return prev_ee_pose_at_base

    prev_ee_pose_at_camera = transform_pose_cv(prev_ee_pose_at_base, base_pose, extrinsic_cv)
    return prev_ee_pose_at_camera

def cal_action_from_pose(env, pose, extrinsic_cv, camera_coord=True):
    import sapien.core as sapien
    assert len(pose) == 8
    base_pose = env.agent.robot.pose
    prev_ee_pose_at_base = eef_pose(env)
    target_ee_pose_at_camera = sapien.Pose(p=pose[:3], q=pose[3:7])

    if camera_coord:
        target_ee_pose = transform_pose_cv(target_ee_pose_at_camera, base_pose, extrinsic_cv, True)
    else:
        target_ee_pose = target_ee_pose_at_camera
    ee_pose_at_ee = prev_ee_pose_at_base.inv() * target_ee_pose
    ee_pose_at_ee = np.r_[
        ee_pose_at_ee.p,
        compact_axis_angle_from_quaternion(ee_pose_at_ee.q),
    ]
    arm_action = inv_scale_action(ee_pose_at_ee, -0.1, 0.1)
    action = np.hstack([arm_action, pose[-1]])
    return action

def interpolate_trajectory(waypoints, steps):
    """
    Interpolates a trajectory between given waypoints.

    Args:
    waypoints (list): List of lists representing the waypoints. Each inner list should contain
                      the position (x, y, z) and orientation (w, x, y, z) in quaternion format.
                      Format: [x, y, z, w, i, j, k]
                      At least two waypoints are required. The first and last points of the
                      output trajectory will be the same as the first and last points of the
                      input trajectory, respectively.
    steps (int): Total number of points in the output trajectory, including the first and last
                 waypoints.

    Returns:
    list: A list of interpolated positions and orientations along the trajectory.
          Each element in the list is a tuple representing a point on the trajectory,
          with the format (x, y, z, w, i, j, k).

    Note:
    This function uses cubic spline interpolation for positions and spherical linear
    interpolation (Slerp) for orientations between waypoints.
    """
    assert len(waypoints) >= 2 and steps >= 2, "waypoints size and output steps must be at least 2"

    from scipy.interpolate import CubicSpline
    from scipy.spatial.transform import Rotation as R
    from scipy.spatial.transform import Slerp

    positions = np.array([wp[:3] for wp in waypoints])
    key_times = np.zeros(len(positions))
    for i in range(1, len(positions)):
        key_times[i] = key_times[i-1] + np.linalg.norm(positions[i]-positions[i-1])

    orientations = np.array([[wp[i] for i in [4,5,6,3]] for wp in waypoints])
    key_rots = R.from_quat(orientations)

    spline = CubicSpline(key_times, positions, axis=0)
    slerp = Slerp(key_times, key_rots)

    new_positions = spline(np.linspace(0, key_times[-1], steps))

    new_positions[0] = positions[0]
    new_positions[-1] = positions[-1]

    times = np.zeros(len(new_positions))
    for i in range(1, len(new_positions)):
        times[i] = times[i-1] + np.linalg.norm(new_positions[i]-new_positions[i-1])
    times *= key_times[-1] / times[-1]
    times[-1] = key_times[-1]
    interp_rots = slerp(times)
    new_orientations = interp_rots.as_quat()

    return [(new_positions[i][0], new_positions[i][1], new_positions[i][2],
             new_orientations[i][3], new_orientations[i][0], new_orientations[i][1],
             new_orientations[i][2]) for i in range(len(new_positions))]

def joints_to_pose(joints):
    """
    Apply forward kinematics method for franka panda robot.

    Args:
    joints (list): List of target joints value, must orderd from 0 to 6

    Returns:
    list: List of target eef pose / panda_link8 with the format (x, y, z, w, i, j, k).
    """
    assert len(joints) == 7, "joints length must be 7"

    def dh_params(joints):
        M_PI = np.pi

        # Create DH parameters (data given by maker franka-emika)
        dh = np.array(
            [
                [0, 0, 0.333, joints[0]],
                [-M_PI / 2, 0, 0, joints[1]],
                [M_PI / 2, 0, 0.316, joints[2]],
                [M_PI / 2, 0.0825, 0, joints[3]],
                [-M_PI / 2, -0.0825, 0.384, joints[4]],
                [M_PI / 2, 0, 0, joints[5]],
                [M_PI / 2, 0.088, 0.107, joints[6]],
            ]
        )
        return dh

    def TF_matrix(i, dh):
        # Define Transformation matrix based on DH params
        alpha = dh[i][0]
        a = dh[i][1]
        d = dh[i][2]
        q = dh[i][3]

        TF = np.array(
            [
                [np.cos(q), -np.sin(q), 0, a],
                [
                    np.sin(q) * np.cos(alpha),
                    np.cos(q) * np.cos(alpha),
                    -np.sin(alpha),
                    -np.sin(alpha) * d,
                ],
                [
                    np.sin(q) * np.sin(alpha),
                    np.cos(q) * np.sin(alpha),
                    np.cos(alpha),
                    np.cos(alpha) * d,
                ],
                [0, 0, 0, 1],
            ]
        )
        return TF

    dh_parameters = dh_params(joints)

    T_01 = TF_matrix(0,dh_parameters)
    T_12 = TF_matrix(1,dh_parameters)
    T_23 = TF_matrix(2,dh_parameters)
    T_34 = TF_matrix(3,dh_parameters)
    T_45 = TF_matrix(4,dh_parameters)
    T_56 = TF_matrix(5,dh_parameters)
    T_67 = TF_matrix(6,dh_parameters)
    T_07 = T_01@T_12@T_23@T_34@T_45@T_56@T_67 


    translation = T_07[:3,3]
    quaternion = mat2euler(T_07[:3,:3])

    return np.concatenate((translation, quaternion))


