from pytorch3d.transforms import (
    Transform3d,
    euler_angles_to_matrix,
    matrix_to_euler_angles,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_to_matrix,quaternion_to_axis_angle
)
import numpy as np
import torch
import scipy.spatial.transform as st

def normalize(data, data_min, data_max, norm_max, norm_min):

    rescale_data = (data - data_min) / (data_max - data_min) * (norm_max - norm_min) + norm_min
    rescale_data = torch.clip(rescale_data, min=norm_min, max=norm_max)

    return rescale_data

def is_diff_small(pose1, pose2, threshold_sum=2e-2, threshold_max=5e-3):
    diff_sum = (pose2 - pose1).abs().sum()
    diff_max = (pose2 - pose1).abs().max()
    if diff_sum <= threshold_sum and diff_max <= threshold_max:
        return True
    else:
        return False

def dict_to_gpu(dict, DEVICE):

    gpu_dict = {}
    for k in dict:
        if k == "camera_extrinsic_cv":
            continue
        b, sample_per_episode = dict[k].shape[:2]
        gpu_dict[k] = dict[k].reshape(b * sample_per_episode, *dict[k].shape[2:]).to(DEVICE, non_blocking=True)
        # if k == 'image':
        #     gpu_dict[k] = gpu_dict[k].permute(0,1,3,4,2).contiguous()
    return gpu_dict
# def is_diff_small(pose1, pose2, threshold_sum=5e-3, threshold_max=5e-3):
#     diff_sum = (pose2 - pose1).abs().sum()
#     if diff_sum <= threshold_sum:
#         return True
#     else:
#         return False


def quaternion_to_euler_radians(w, x, y, z):
    roll = np.arctan2(2 * (w * x + y * z), w**2 + z**2 - (x**2 + y**2))

    sinpitch = 2 * (w * y - z * x)
    pitch = np.arcsin(sinpitch)

    yaw = np.arctan2(2 * (w * z + x * y), w**2 + x**2 - (y**2 + z**2))

    return torch.tensor([roll, pitch, yaw], dtype=torch.float32)


@torch.no_grad()
def euler_to_quaternion(eulers):
    quaternion = st.Rotation.from_euler('xyz', eulers).as_quat()
    return torch.tensor([quaternion[-1], quaternion[0], quaternion[1], quaternion[2]])

def add_noise_to_pose(gripper_change_pose, noise_stddev_rot, noise_stddev_trans):
    euler = quaternion_to_euler_radians(gripper_change_pose[3].item(),gripper_change_pose[4].item(), gripper_change_pose[5].item(), gripper_change_pose[6].item())
    noisy_euler = add_noise_to_euler(euler[None, :], noise_stddev_rot)[0]
    noisy_quaternion = euler_to_quaternion(noisy_euler.cpu().numpy())
    noisy_translation = add_noise_to_translation(gripper_change_pose[None, :3], noise_stddev_trans)[0]
    gripper_change_pose = torch.cat([noisy_translation, noisy_quaternion], dim=-1).to(torch.float32)
    return gripper_change_pose

@torch.no_grad()
def process_traj_v3(world2cam, pose1, pose2, use_euler=0):

    rot_mat1 = quaternion_to_matrix(pose1[3:])
    rot_mat2 = quaternion_to_matrix(pose2[3:])
    pose1_mat, pose2_mat = torch.eye(4), torch.eye(4)

    pose1_mat[:3, :3] = rot_mat1
    pose2_mat[:3, :3] = rot_mat2
    pose1_mat[:3, 3] = pose1[:3]
    pose2_mat[:3, 3] = pose2[:3]

    pose1_transform = Transform3d(matrix=pose1_mat.T)
    pose2_transform = Transform3d(matrix=pose2_mat.T)
    world2cam_transform = Transform3d(matrix=world2cam.T)
    pose1_cam = pose1_transform.compose(world2cam_transform)
    pose2_cam = pose2_transform.compose(world2cam_transform)

    pose1_to_pose2 = pose1_cam.inverse().compose(pose2_cam)

    # translation_delta = pose1_to_pose2.get_matrix()[0, -1, :3]
    translation_delta = pose2_cam.get_matrix()[0, -1, :3] - pose1_cam.get_matrix()[0, -1, :3]

    if use_euler:
        rotation_delta = matrix_to_euler_angles(pose1_to_pose2.get_matrix()[0, :3, :3].T, convention="XYZ")
    else:
        rotation_delta = matrix_to_quaternion(pose1_to_pose2.get_matrix()[0, :3, :3].T)

    return translation_delta.to(torch.float32), rotation_delta.to(torch.float32)



@torch.no_grad()
def get_pose_cam(world2cam, pose1, use_euler=0):

    rot_mat1 = quaternion_to_matrix(pose1[3:])
    pose1_mat = torch.eye(4)

    pose1_mat[:3, :3] = rot_mat1
    pose1_mat[:3, 3] = pose1[:3]


    pose1_transform = Transform3d(matrix=pose1_mat.T)

    world2cam_transform = Transform3d(matrix=world2cam.T)
    pose1_cam = pose1_transform.compose(world2cam_transform)
    vector = pose1_cam.get_matrix()[0, -1, :3]
    if use_euler:
        rotation = matrix_to_euler_angles(pose1_cam.get_matrix()[0, :3, :3].T, convention="XYZ")
    else:
        rotation = matrix_to_quaternion(pose1_cam.get_matrix()[0, :3, :3].T)

    return torch.cat([vector, rotation])


@torch.no_grad()
def process_traj(world2cam, pose1, pose2):
    return process_traj_v3(world2cam, pose1, pose2)


def add_noise_to_euler(euler_angles, noise_stddev):  
    """  
    Add Gaussian noise to Euler angles (roll, pitch, yaw) for a batch of data.  
  
    Parameters:  
    - euler_angles: torch.Tensor of shape (batch_size, 3) representing roll, pitch, yaw in radians.  
    - noise_stddev: float or torch.Tensor of shape (batch_size, 3) or (batch_size,) for the standard deviation of the Gaussian noise to add.  
  
    Returns:  
    - noisy_euler_angles: torch.Tensor of shape (batch_size, 3) with added noise.  
    """  
    if isinstance(noise_stddev, float):  
        noise_stddev = noise_stddev * torch.ones_like(euler_angles)  
    elif noise_stddev.dim() == 1:  
        noise_stddev = noise_stddev.unsqueeze(-1).repeat(1, 3)  
  
    noise = torch.randn_like(euler_angles) * noise_stddev  
    noisy_euler_angles = euler_angles + noise  
    # Ensure angles are within -pi to pi range  
    noisy_euler_angles = torch.remainder(noisy_euler_angles + torch.pi, 2 * torch.pi) - torch.pi  
    return noisy_euler_angles  
  
def add_noise_to_translation(translation, noise_stddev):  
    """  
    Add Gaussian noise to a translation vector for a batch of data.  
  
    Parameters:  
    - translation: torch.Tensor of shape (batch_size, 3) representing x, y, z translations.  
    - noise_stddev: float or torch.Tensor of shape (batch_size, 3) or (batch_size,) for the standard deviation of the Gaussian noise to add.  
  
    Returns:  
    - noisy_translation: torch.Tensor of shape (batch_size, 3) with added noise.  
    """  
    if isinstance(noise_stddev, float):  
        noise_stddev = noise_stddev * torch.ones_like(translation)  
    elif noise_stddev.dim() == 1:  
        noise_stddev = noise_stddev.unsqueeze(-1).repeat(1, 3)  
  
    noise = torch.randn_like(translation) * noise_stddev  
    noisy_translation = translation + noise  
    return noisy_translation  

def add_noise_to_quaternion(quaternions, noise_scale=0.01):  
    """  
    Add noise to a batch of quaternions.  
  
    Parameters:  
    - quaternions: torch.Tensor of shape (N, 4) where N is the batch size and 4 represents the quaternion (w, x, y, z).  
    - noise_scale: float, the scale of the noise to add. The noise will be sampled from a normal distribution with mean 0 and standard deviation proportional to this scale.  
  
    Returns:  
    - noisy_quaternions: torch.Tensor of shape (N, 4) representing the quaternions with added noise and renormalized.  
    """  
    # Extract the vector part (x, y, z) of the quaternions  
    vector_part = quaternions[:, 1:]  
      
    # Add noise to the vector part. We use a normal distribution for the noise.  
    # Note: The noise scale is multiplied by the norm of the vector part to make the noise relative to the size of the quaternion.  
    # This step is optional and depends on your specific use case.  
    # Here, we apply the noise scale directly without considering the norm of the vector part for simplicity.  
    noise = noise_scale * torch.randn_like(vector_part)  
      
    # Add the noise to the vector part  
    noisy_vector_part = vector_part + noise  
      
    # The scalar part (w) remains unchanged  
    scalar_part = quaternions[:, 0:1]  
      
    # Combine the scalar part and the noisy vector part  
    noisy_quaternions = torch.cat([scalar_part, noisy_vector_part], dim=1)  
      
    # Renormalize the quaternions to ensure they still represent valid rotations  
    norms = torch.linalg.norm(noisy_quaternions, dim=1, keepdim=True)  
    noisy_quaternions_renormalized = noisy_quaternions / norms  
      
    return noisy_quaternions_renormalized  