"""
goal_relabeling.py

Contains simple goal relabeling logic for BC use-cases where rewards and next_observations are not required.
Each function should add entries to the "task" dict.
"""

from typing import Dict

import tensorflow as tf

from prismatic.vla.datasets.rlds.utils.data_utils import tree_merge


def uniform(traj: Dict) -> Dict:
    """Relabels with a true uniform distribution over future states."""
    traj_len = tf.shape(tf.nest.flatten(traj["observation"])[0])[0]

    # Select a random future index for each transition i in the range [i + 1, traj_len)
    rand = tf.random.uniform([traj_len])
    low = tf.cast(tf.range(traj_len) + 1, tf.float32)
    high = tf.cast(traj_len, tf.float32)
    goal_idxs = tf.cast(rand * (high - low) + low, tf.int32)

    # Sometimes there are floating-point errors that cause an out-of-bounds
    goal_idxs = tf.minimum(goal_idxs, traj_len - 1)

    # Adds keys to "task" mirroring "observation" keys (`tree_merge` to combine "pad_mask_dict" properly)
    goal = tf.nest.map_structure(lambda x: tf.gather(x, goal_idxs), traj["observation"])
    # this will generate a timestep key, why?
    traj["task"] = tree_merge(traj["task"], goal)
    
    return traj

def gripper_change_goal(traj: Dict) -> Dict:
    """Relabels with a true uniform distribution over future states."""
    gripper_change_status = traj['action'][..., 1:, -1] - traj['action'][..., :-1,  -1]
    gripper_change_idx = tf.pad(gripper_change_status != 0, [(0, 0), (1, 0)])
    goal_idx = tf.where(gripper_change_status)
    goal = tf.zeros_like(traj['action'][..., :, -1])
    goal[goal_idx] = 1.
    # tf.pad(goal, [(0, 0), (1, 0)])

    # Adds keys to "task" mirroring "observation" keys (`tree_merge` to combine "pad_mask_dict" properly)
    # goal = tf.nest.map_structure(lambda x: tf.gather(x, goal_idxs), traj["observation"])
    # this will generate a timestep key, why?
    traj["task"]['timesteps'] = goal
    
    return traj
