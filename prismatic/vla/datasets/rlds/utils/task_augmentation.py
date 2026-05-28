"""
task_augmentation.py

Contains basic logic for randomly zeroing out keys in the task specification.
"""

from typing import Dict

import tensorflow as tf

from prismatic.vla.datasets.rlds.utils.data_utils import to_padding


def delete_task_conditioning(traj: Dict, keep_image_prob: float) -> Dict:
    """
    Randomly drops out either the goal images or the language instruction. Only does something if both of
    these are present.

    如果一条 trajectory 的 task 里同时有语言指令和 goal image/depth，那么训练时随机只保留其中一种：要么保留 goal image/depth，要么保留 language instruction。

    Args:
        traj: A dictionary containing trajectory data. Should have a "task" key.
        keep_image_prob: The probability of keeping the goal images. The probability of keeping the language
            instruction is 1 - keep_image_prob.
    """
    if "language_instruction" not in traj["task"]:
        return traj

    image_keys = {key for key in traj["task"].keys() if key.startswith("image_") or key.startswith("depth_")}
    if not image_keys:
        return traj 

    traj_len = tf.shape(traj["action"])[0]
    should_keep_images = tf.random.uniform([traj_len]) < keep_image_prob
    should_keep_images |= ~traj["task"]["pad_mask_dict"]["language_instruction"]

    for key in image_keys | {"language_instruction"}:
        # 删除 image 或者 删除 language
        should_keep = should_keep_images if key in image_keys else ~should_keep_images
        # pad out the key
        traj["task"][key] = tf.where(
            should_keep,
            traj["task"][key],
            to_padding(traj["task"][key]),
        )
        # zero out the pad mask dict for the key
        traj["task"]["pad_mask_dict"][key] = tf.where(
            should_keep,
            traj["task"]["pad_mask_dict"][key],
            tf.zeros_like(traj["task"]["pad_mask_dict"][key]),
        )

    # when no goal images are present, the goal timestep becomes the final timestep
    traj["task"]["timestep"] = tf.where(
        should_keep_images,
        traj["task"]["timestep"],
        traj_len - 1,
    )

    return traj
