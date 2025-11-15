import os
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import json

SEED = 42

NUM_CAMERAS = 300000

CAMERA_POS_RANGE = {
    "theta": (10, 80),
    "phi": (0, 180),
    "radius": (0.4, 0.8),
}

CAMERA_LOOKAT_RANGE = {
    "theta": (0, 180),
    "phi": (0, 360),
    "radius": (0.1, 0.3),
}

CAMERA_UP_RANGE = {
    "delta": (-10, 10),
}

def generate_camera_pool(num_cameras, camera_pos_range, camera_lookat_range, camera_up_range):
    cameras = []
    for i_cam in range(num_cameras):
        print(f"Generating camera {i_cam+1}/{num_cameras}...")
        # Generate camera position
        pos_theta = np.random.uniform(camera_pos_range["theta"][0], camera_pos_range["theta"][1])
        pos_phi = np.random.uniform(camera_pos_range["phi"][0], camera_pos_range["phi"][1])
        pos_radius = np.random.uniform(camera_pos_range["radius"][0], camera_pos_range["radius"][1])
        pos_z = pos_radius * math.cos(math.radians(pos_theta))
        pos_x = pos_radius * math.sin(math.radians(pos_theta)) * math.sin(math.radians(pos_phi))
        pos_y = pos_radius * math.sin(math.radians(pos_theta)) * math.cos(math.radians(pos_phi))
        cam_pos = [pos_x, pos_y, pos_z]
        
        # Generate camera lookat
        lookat_theta = np.random.uniform(camera_lookat_range["theta"][0], camera_lookat_range["theta"][1])
        lookat_phi = np.random.uniform(camera_lookat_range["phi"][0], camera_lookat_range["phi"][1])
        lookat_radius = np.random.uniform(camera_lookat_range["radius"][0], camera_lookat_range["radius"][1])
        lookat_z = lookat_radius * math.cos(math.radians(lookat_theta))
        lookat_x = lookat_radius * math.sin(math.radians(lookat_theta)) * math.sin(math.radians(lookat_phi))
        lookat_y = lookat_radius * math.sin(math.radians(lookat_theta)) * math.cos(math.radians(lookat_phi))
        cam_lookat = [lookat_x, lookat_y, lookat_z]
        
        # Generate camera up
        up_delta = np.random.uniform(camera_up_range["delta"][0], camera_up_range["delta"][1])
        forward = np.array(cam_lookat) - np.array(cam_pos)
        forward /= np.linalg.norm(forward)
        left = np.cross(np.array([0, 0, 1]), forward)
        left /= np.linalg.norm(left)
        up = np.cross(forward, left)
        up /= np.linalg.norm(up)
        # rotate up in the plane of up and right by delta degrees
        r = R.from_rotvec(np.radians(up_delta) * forward)
        cam_up = r.apply(up).tolist()
        
        cameras.append({
            "pos": cam_pos,
            "lookat": cam_lookat,
            "up": cam_up,
            "pos_theta": pos_theta,
            "pos_phi": pos_phi,
            "pos_radius": pos_radius,
            "lookat_theta": lookat_theta,
            "lookat_phi": lookat_phi,
            "lookat_radius": lookat_radius,
            "up_delta": up_delta,
        })
        
    return cameras


def format_camera(cameras):
    camera_array = np.zeros((len(cameras), 16))
    for i_cam, cam in enumerate(cameras):
        camera_array[i_cam] = np.array([
            cam["pos"][0], cam["pos"][1], cam["pos"][2],
            cam["lookat"][0], cam["lookat"][1], cam["lookat"][2],
            cam["up"][0], cam["up"][1], cam["up"][2],
            cam["pos_theta"], cam["pos_phi"], cam["pos_radius"],
            cam["lookat_theta"], cam["lookat_phi"], cam["lookat_radius"],
            cam["up_delta"]
        ])
    return camera_array


if __name__ == "__main__":
    np.random.seed(SEED)
    cameras = generate_camera_pool(NUM_CAMERAS, CAMERA_POS_RANGE, CAMERA_LOOKAT_RANGE, CAMERA_UP_RANGE)
    camera_array = format_camera(cameras)
    save_name = f"configs/camera_pool_{NUM_CAMERAS//1000}k.npz"
    np.savez_compressed(save_name, cameras=camera_array)
    print(f"Camera pool saved to {save_name}")

