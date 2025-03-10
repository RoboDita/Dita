
from pytorch3d.transforms import (
    Transform3d,
    matrix_to_euler_angles,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_to_matrix,
)
import torch
import torch.nn.functional as F

def calc_l2_loss(act, output, camera_extrinsic_cv):

    gt_list = []
    out_list = []
    cam2world = Transform3d(matrix=camera_extrinsic_cv.mT).inverse()

    def get_world_translation_rotation(output):

        translation_delta = output["world_vector"].flatten(0, 1)
        rotation_delta = output["rotation_delta"].flatten(0, 1)
        rotation_delta[..., 0] += 1.0

        pose_1_cam = Transform3d(device=translation_delta.device)
        pose_2_cam = pose_1_cam.rotate(quaternion_to_matrix(rotation_delta)).translate(translation_delta)

        pose1_world = pose_1_cam.compose(cam2world)
        pose2_world = pose_2_cam.compose(cam2world)
        translation_delta_world = pose2_world.get_matrix()[:, -1, :3] - pose1_world.get_matrix()[:, -1, :3]
        rotation_delta_world = matrix_to_quaternion(pose1_world.inverse().compose(pose2_world).get_matrix()[:, :3, :3])

        return translation_delta_world, rotation_delta_world

    translation_pred, rotation_pred = get_world_translation_rotation(output)
    translation_gt, rotation_gt = get_world_translation_rotation(act)

    for k in ["world_vector", "rotation_delta", "gripper_closedness_action", "terminate_episode"]:
        if k == "world_vector":
            gt_list.append(translation_gt)
            out_list.append(translation_pred)
        elif k == "rotation_delta":
            gt_list.append(rotation_gt)
            out_list.append(rotation_pred)
        else:
            gt_list.append(act[k].flatten(0, 1))
            out_list.append(output[k].flatten(0, 1))

    gt = torch.cat(gt_list, dim=-1)
    out = torch.cat(out_list, dim=-1)

    criterion = F.mse_loss

    loss = criterion(gt, out).detach()
    loss_wv = criterion(gt[..., :3], out[..., :3]).detach()
    loss_rota_delta = criterion(gt[..., 3:7], out[..., 3:7]).detach()
    loss_grip_close = criterion(gt[..., 7:8], out[..., 7:8]).detach()
    loss_term = criterion(gt[..., 8:].to(torch.float32), out[..., 8:].to(torch.float32)).detach()

    return loss, loss_wv, loss_rota_delta, loss_grip_close, loss_term


def calc_terminate_recall(labels, outputs):

    labels = ~(labels[:, :, -1].to(torch.bool))
    outputs = ~(outputs[:, :, -1].to(torch.bool))

    TP = ((labels == outputs) & (outputs)).sum()
    TP_and_FN = labels.sum()
    return TP, TP_and_FN


def Check_Wrong_tokens(labels, pred):

    wrong_map = ~(labels == pred)
    wrong_indices = torch.nonzero(wrong_map)
    print(wrong_indices, flush=True)
    return



