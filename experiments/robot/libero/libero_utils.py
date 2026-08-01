"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os
import re
from pathlib import Path

import imageio
import numpy as np
import tensorflow as tf
import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
)


SCENE_PREFIX_RE = re.compile(r"^[A-Z_]+_SCENE\d+_")
LIBERO_PLUS_SUFFIX_RE = re.compile(
    r"(_view_.*|_moved_level\d+_sample\d+|_level\d+_sample\d+|"
    r"_initstate_\d+|_(?:add|light|noise|table|tb)_\d+)$"
)


def canonical_task_name(task_name: str) -> str:
    task_stem = Path(task_name).stem
    if "_language_" in task_stem:
        task_stem = task_stem.split("_language_", 1)[0]
    return LIBERO_PLUS_SUFFIX_RE.sub("", task_stem)


def language_from_task_name(task_name: str) -> str:
    base_task = SCENE_PREFIX_RE.sub("", canonical_task_name(task_name))
    return " ".join(base_task.split("_"))


def bddl_language_instruction(bddl_file: str) -> str:
    return BDDLUtils.get_problem_info(bddl_file)["language_instruction"]


def language_bddl_file(task, bddl_file: str) -> Path:
    bddl_path = Path(bddl_file)
    if bddl_path.exists():
        return bddl_path
    if "_language_" in task.name and "_view_" in task.name:
        candidate = bddl_path.with_name(f"{task.name.split('_view_', 1)[0]}.bddl")
        if candidate.exists():
            return candidate
    return bddl_path


def get_official_task_language(task, bddl_file: str) -> str:
    if "_language_" in task.name:
        return bddl_language_instruction(str(language_bddl_file(task, bddl_file)))
    return language_from_task_name(task.name)


def get_task_language(task, bddl_file: str, language_instruction_mode: str = "official") -> str:
    if language_instruction_mode == "raw":
        return task.language
    if language_instruction_mode == "official":
        return get_official_task_language(task, bddl_file)
    raise ValueError("language_instruction_mode must be one of: official, raw.")


def get_initial_states_task_key(all_initial_states, task_description: str, task_name: str) -> str:
    candidates = [task_description.replace(" ", "_"), task_name]
    for key in candidates:
        if key in all_initial_states:
            return key
    raise KeyError(
        f"Could not find initial states for task. Tried keys: {candidates}. "
        f"Available keys include: {list(all_initial_states)[:5]}"
    )


def get_libero_env(task, model_family, resolution=256, language_instruction_mode="official", seed=0):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    task_description = get_task_language(task, task_bddl_file, language_instruction_mode)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def get_libero_image(obs):
    """Extracts third-person image from observations and preprocesses it."""
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def get_libero_wrist_image(obs):
    """Extracts wrist camera image from observations and preprocesses it."""
    img = obs["robot0_eye_in_hand_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--openvla_oft--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den
