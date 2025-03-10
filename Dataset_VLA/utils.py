import blosc
import pickle

import einops
from pickle import UnpicklingError
import numpy as np
from scipy.interpolate import CubicSpline, interp1d
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_f
import math

def alpha2rotm(a):
    """Alpha euler angle to rotation matrix."""
    rotm = np.array([
        [1, 0, 0],
        [0, np.cos(a), -np.sin(a)],
        [0, np.sin(a),  np.cos(a)]
    ])
    return rotm

def beta2rotm(b):
    """Beta euler angle to rotation matrix."""
    rotm = np.array([
        [np.cos(b), 0, np.sin(b)],
        [0, 1, 0],
        [-np.sin(b), 0, np.cos(b)]
    ])
    return rotm

def gamma2rotm(c):
    """Gamma euler angle to rotation matrix."""
    rotm = np.array([
        [np.cos(c), -np.sin(c), 0],
        [np.sin(c),  np.cos(c), 0],
        [0, 0, 1]
    ])
    return rotm

def euler2rotm(euler_angles):
    """Euler angle (ZYX) to rotation matrix."""
    alpha = euler_angles[0]
    beta = euler_angles[1]
    gamma = euler_angles[2]

    rotm_a = alpha2rotm(alpha)
    rotm_b = beta2rotm(beta)
    rotm_c = gamma2rotm(gamma)

    rotm = rotm_c @ rotm_b @ rotm_a

    return rotm

def isRotm(R):
    # Checks if a matrix is a valid rotation matrix.
    # Forked from Andy Zeng
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotm2euler(R):
    # Forked from: https://learnopencv.com/rotation-matrix-to-euler-angles/
    # R = Rz * Ry * Rx
    assert(isRotm(R))
    sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6
 
    if not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    
    # (-pi , pi]
    while x > np.pi:
        x -= (2 * np.pi)
    while x <= -np.pi:
        x += (2 * np.pi)
    while y > np.pi:
        y -= (2 * np.pi)
    while y <= -np.pi:
        y += (2 * np.pi)
    while z > np.pi:
        z -= (2 * np.pi)
    while z <= -np.pi:
        z += (2 * np.pi)
    return np.array([x, y, z])




def loader(file, client = None):
    if str(file).__contains__("s3://"):
        if str(file).endswith(".npy"):
            try:
                content = np.load(client.get(file), allow_pickle=True)
                return content
            except UnpicklingError as e:
                print(f"Can't load {file}: {e}")
        elif str(file).endswith(".dat"):
            try:
                with open(file, "rb") as f:
                    content = pickle.loads(blosc.decompress(client.get(file)))
                return content
            except UnpicklingError as e:
                print(f"Can't load {file}: {e}")
        elif str(file).endswith(".pkl"):
            try:
                with open(file, 'rb') as f:
                    content = pickle.load(client.get(file))
                return content
            except UnpicklingError as e:
                print(f"Can't load {file}: {e}")
        pass
    else:
        if str(file).endswith(".npy"):
            try:
                content = np.load(file, allow_pickle=True)
                return content
            except UnpicklingError as e:
                print(f"Can't load {file}: {e}")
        elif str(file).endswith(".dat"):
            try:
                with open(file, "rb") as f:
                    content = pickle.loads(blosc.decompress(f.read()))
                return content
            except UnpicklingError as e:
                print(f"Can't load {file}: {e}")
        elif str(file).endswith(".pkl"):
            try:
                with open(file, 'rb') as f:
                    content = pickle.load(f)
                return content
            except UnpicklingError as e:
                print(f"Can't load {file}: {e}")
    return None


class Resize:
    """Resize and pad/crop the image and aligned point cloud."""

    def __init__(self, scales):
        self.scales = scales

    def __call__(self, **kwargs):
        """Accept tensors as T, N, C, H, W."""
        keys = list(kwargs.keys())

        if len(keys) == 0:
            raise RuntimeError("No args")

        # Sample resize scale from continuous range
        sc = np.random.uniform(*self.scales)

        t, n, c, raw_h, raw_w = kwargs[keys[0]].shape
        kwargs = {n: arg.flatten(0, 1) for n, arg in kwargs.items()}
        resized_size = [int(raw_h * sc), int(raw_w * sc)]

        # Resize
        kwargs = {
            n: transforms_f.resize(
                arg,
                resized_size,
                transforms.InterpolationMode.NEAREST
            )
            for n, arg in kwargs.items()
        }

        # If resized image is smaller than original, pad it with a reflection
        if raw_h > resized_size[0] or raw_w > resized_size[1]:
            right_pad, bottom_pad = max(raw_w - resized_size[1], 0), max(
                raw_h - resized_size[0], 0
            )
            kwargs = {
                n: transforms_f.pad(
                    arg,
                    padding=[0, 0, right_pad, bottom_pad],
                    padding_mode="reflect",
                )
                for n, arg in kwargs.items()
            }

        # If resized image is larger than original, crop it
        i, j, h, w = transforms.RandomCrop.get_params(
            kwargs[keys[0]], output_size=(raw_h, raw_w)
        )
        kwargs = {
            n: transforms_f.crop(arg, i, j, h, w) for n, arg in kwargs.items()
        }

        kwargs = {
            n: einops.rearrange(arg, "(t n) c h w -> t n c h w", t=t)
            for n, arg in kwargs.items()
        }

        return kwargs


class TrajectoryInterpolator:
    """Interpolate a trajectory to have fixed length."""

    def __init__(self, use=False, interpolation_length=50):
        self._use = use
        self._interpolation_length = interpolation_length

    def __call__(self, trajectory):
        if not self._use:
            return trajectory
        trajectory = trajectory.numpy()
        # Calculate the current number of steps
        old_num_steps = len(trajectory)

        # Create a 1D array for the old and new steps
        old_steps = np.linspace(0, 1, old_num_steps)
        new_steps = np.linspace(0, 1, self._interpolation_length)

        # Interpolate each dimension separately
        resampled = np.empty((self._interpolation_length, trajectory.shape[1]))
        for i in range(trajectory.shape[1]):
            if i == (trajectory.shape[1] - 1):  # gripper opening
                interpolator = interp1d(old_steps, trajectory[:, i])
            else:
                interpolator = CubicSpline(old_steps, trajectory[:, i])

            resampled[:, i] = interpolator(new_steps)

        resampled = torch.tensor(resampled)
        if trajectory.shape[1] == 8:
            from diffuser_actor.utils.utils import normalise_quat
            resampled[:, 3:7] = normalise_quat(resampled[:, 3:7])
        return resampled
