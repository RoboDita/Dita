import os
import os.path as osp
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import cv2

from sapien.core import Actor


def get_2d_segm_for_actor(actor: Actor, observed_segm_map):
    actor_mask = observed_segm_map == actor.id
    # ys, xs = np.where(actor_mask)
    # bbox = np.array([np.min(xs), np.min(ys), np.max(xs), np.max(ys)])
    # return bbox, actor_mask
    return actor_mask



def get_2d_bbox_from_segm(segm_map):
    ys, xs = np.where(segm_map)
    bbox = np.array([np.min(xs), np.min(ys), np.max(xs), np.max(ys)])
    return bbox


def rle_encode_mask(mask):
    shape = mask.shape
    flat_mask = mask.flatten().astype(np.uint8)
    rle = []
    n = len(flat_mask)
    current_value = flat_mask[0]
    start = 0
    
    for i in range(1, n):
        if flat_mask[i] != current_value:
            rle.append([start, i - start])
            start = i
            current_value = flat_mask[i]
    
    rle.append([start, n - start])
    
    return [rle, flat_mask[0], list(shape)]


def rle_decode_mask(rle):
    shape = tuple(rle[2])
    mask = np.zeros(shape, dtype=np.uint8).reshape(-1)
    current_value = rle[1]
    
    for start, length in rle[0]:
        mask[start:start + length] = current_value
        current_value = 1 - current_value
    
    mask = mask.reshape(shape)
    
    return mask > 0
    

