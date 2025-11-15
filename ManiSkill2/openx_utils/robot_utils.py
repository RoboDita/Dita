import numpy as np
import sapien.core as sapien

from geometry import compact_axis_angle_from_quaternion, inv_scale_action
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