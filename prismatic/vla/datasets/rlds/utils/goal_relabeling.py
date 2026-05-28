"""
goal_relabeling.py

Contains simple goal relabeling logic for BC use-cases where rewards and next_observations are not required.
Each function should add entries to the "task" dict.
"""

from typing import Dict

import tensorflow as tf

from prismatic.vla.datasets.rlds.utils.data_utils import tree_merge


def uniform(traj: Dict) -> Dict:
    # 对一条 trajectory 里的每个时间步 i，随机选择一个未来时间步 j > i 作为 goal，然后把那个未来时间步的 observation 复制到当前时间步的 task 里面。
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
    traj["task"] = tree_merge(traj["task"], goal)

    return traj
