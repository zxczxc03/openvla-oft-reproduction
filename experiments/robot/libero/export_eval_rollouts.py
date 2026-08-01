"""
run_libero_eval.py

Evaluates a trained policy in a LIBERO simulation benchmark task suite.
"""

import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

import h5py
import time

import wandb

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_initial_states_task_key,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 250,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)

    unnorm_key: Union[str, Path] = ""                # Action/proprio normalization statistics key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL  # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "RESET"               # "RESET", "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)
    language_instruction_mode: str = "official"      # "official" cleans LIBERO-plus suffixes; "raw" uses task.language

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    save_failed_episodes: bool = True
    failed_episode_dir: str = "./failed_episodes"
    save_success_episodes: bool = True
    success_episode_dir: str = "./success_episodes"
    
    seed: int = 0                                    # Random Seed (for reproducibility)

    # fmt: on


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"
    assert cfg.language_instruction_mode in {"official", "raw"}, (
        "language_instruction_mode must be one of: official, raw"
    )


def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)

    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,  # 8-dimensional proprio for LIBERO
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Select the action/proprio normalization statistics used during evaluation."""
    # Prefer the explicit shared statistics emitted by mixed-dataset training.
    if cfg.unnorm_key:
        unnorm_key = str(cfg.unnorm_key)
    elif "shared_bounds" in model.norm_stats:
        unnorm_key = "shared_bounds"
    else:
        unnorm_key = cfg.task_suite_name

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, (
        f"Normalization statistics key {unnorm_key} not found in VLA `norm_stats`! "
        f"Available keys: {list(model.norm_stats.keys())}"
    )

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key


def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    run_id += f"--lang-{cfg.language_instruction_mode}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging if enabled
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    if cfg.initial_states_path == "RESET":
        log_message("Using initial states produced by env.reset()", log_file)
        return None, None

    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None


def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation, img  # Return both processed observation and original image for replay


def process_action(action, model_family):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action



def get_flattened_sim_state(env):
    """
    保存 MuJoCo 内部状态。
    如果你的 env 外面包了 wrapper，可能需要改成 env.env.sim。
    """
    return np.array(env.sim.get_state().flatten(), dtype=np.float64)


def make_safe_name(text: str, max_length: int = 80) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text.lower())[:max_length]


def get_episode_hdf5_path(cfg, task_description, task_id, episode_idx, success):
    output_dir = cfg.success_episode_dir if success else cfg.failed_episode_dir
    safe_task_name = make_safe_name(task_description)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    unique_suffix = time.time_ns()
    episode_name = f"{timestamp}-{unique_suffix}--task={task_id}--ep={episode_idx}--{safe_task_name}"

    episode_dir = os.path.join(output_dir, episode_name)
    os.makedirs(episode_dir, exist_ok=True)
    return os.path.join(episode_dir, "demo.hdf5")


def write_regeneration_demo_group(hdf5_file, bddl_file, task_description, rollout, success):
    data_group = hdf5_file.create_group("data")
    data_group.attrs["bddl_file_name"] = bddl_file
    data_group.attrs["language_instruction"] = task_description
    data_group.attrs["num_demos"] = 1
    data_group.attrs["total"] = len(rollout["actions"])
    data_group.attrs["success"] = bool(success)

    try:
        with open(bddl_file, "r", encoding="utf-8") as bddl_file_obj:
            data_group.attrs["bddl_file_content"] = bddl_file_obj.read()
    except OSError:
        pass

    demo_group = data_group.create_group("demo_0")
    demo_group.create_dataset("states", data=np.asarray(rollout["sim_states"], dtype=np.float64))
    demo_group.create_dataset("actions", data=np.asarray(rollout["actions"], dtype=np.float32))
    demo_group.create_dataset("robot_states", data=np.asarray(rollout["robot_states"], dtype=np.float32))
    demo_group.create_dataset("rewards", data=np.asarray(rollout["rewards"], dtype=np.float32))
    demo_group.create_dataset("dones", data=np.asarray(rollout["dones"], dtype=np.bool_))
    demo_group.attrs["num_samples"] = len(rollout["actions"])
    demo_group.attrs["success"] = bool(success)


def get_robot_state_from_obs(obs):
    """
    和 prepare_observation 里保持一致：
    eef_pos + eef_quat(axis-angle) + gripper_qpos
    """
    return np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)


def save_episode_hdf5(
    cfg,
    env,
    rollout,
    task_description,
    task_id,
    episode_idx,
    initial_state,
    policy_start_state,
    initial_state_source,
    bddl_file,
    success
):
    trajectory_lengths = {
        key: len(rollout[key]) for key in ("sim_states", "actions", "robot_states", "rewards", "dones")
    }
    if len(set(trajectory_lengths.values())) != 1:
        raise ValueError(f"Cannot save inconsistent episode lengths: {trajectory_lengths}")

    hdf5_path = get_episode_hdf5_path(cfg, task_description, task_id, episode_idx, success)

    with h5py.File(hdf5_path, "w") as f:
        f.attrs["task_suite_name"] = cfg.task_suite_name
        f.attrs["task_description"] = task_description
        f.attrs["task_id"] = task_id
        f.attrs["episode_idx"] = episode_idx
        f.attrs["bddl_file"] = bddl_file
        f.attrs["model_family"] = cfg.model_family
        f.attrs["pretrained_checkpoint"] = str(cfg.pretrained_checkpoint)
        f.attrs["seed"] = cfg.seed
        f.attrs["success"] = success
        f.attrs["initial_state_source"] = initial_state_source
        f.attrs["initial_state_timing"] = "before_stabilization_wait"
        f.attrs["policy_start_state_timing"] = "after_stabilization_wait_before_first_action"
        f.attrs["sim_state_timing"] = "pre_action"
        f.attrs["num_steps_wait"] = cfg.num_steps_wait
        f.attrs["num_steps"] = len(rollout["actions"])

        if initial_state is not None:
            f.create_dataset("initial_state", data=np.asarray(initial_state, dtype=np.float64))
        if policy_start_state is not None:
            f.create_dataset("policy_start_state", data=np.asarray(policy_start_state, dtype=np.float64))

        # 如果能拿到 model xml，也存下来，后续精确 replay 更稳
        try:
            f.attrs["model_xml"] = env.model.get_xml()
        except Exception:
            pass

        f.create_dataset("sim_states", data=np.asarray(rollout["sim_states"], dtype=np.float64))
        f.create_dataset("actions", data=np.asarray(rollout["actions"], dtype=np.float32))
        f.create_dataset("robot_states", data=np.asarray(rollout["robot_states"], dtype=np.float32))
        f.create_dataset("rewards", data=np.asarray(rollout["rewards"], dtype=np.float32))
        f.create_dataset("dones", data=np.asarray(rollout["dones"], dtype=np.bool_))
        write_regeneration_demo_group(f, bddl_file, task_description, rollout, success)

    return hdf5_path


def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    log_file=None,
):
    """Run a single episode in the environment."""
    # Initialize action queue
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
              f"({NUM_ACTIONS_CHUNK}) constant defined in prismatic.vla.constants! For best performance (in terms of "
               "both speed and success rate), we recommend executing the full action chunk.")
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    t = 0
    replay_images = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    rollout = {
        "sim_states": [],
        "actions": [],
        "robot_states": [],
        "rewards": [],
        "dones": [],
    }

    # Run episode
    success = False
    episode_error = None
    episode_initial_state = None
    policy_start_state = None
    try:
        # RESET mode leaves the environment at its own sampled reset state.
        obs = env.reset()
        if initial_state is not None:
            obs = env.set_init_state(initial_state)
        episode_initial_state = get_flattened_sim_state(env)

        while t < max_steps + cfg.num_steps_wait:
            # Do nothing for the first few timesteps to let objects stabilize
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            if policy_start_state is None:
                policy_start_state = get_flattened_sim_state(env)

            # Prepare observation
            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)

            # If action queue is empty, requery model
            if len(action_queue) == 0:
                # Query model to get action
                actions = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )
                action_queue.extend(actions)

            # Get action from queue
            action = action_queue.popleft()

            # Process action
            action = process_action(action, cfg.model_family)
            pre_action_sim_state = get_flattened_sim_state(env)
            pre_action_robot_state = get_robot_state_from_obs(obs)

            # Execute action in environment
            obs, reward, done, info = env.step(action.tolist())

            if cfg.save_failed_episodes or cfg.save_success_episodes:
                rollout["sim_states"].append(pre_action_sim_state)
                rollout["actions"].append(action.copy())
                rollout["robot_states"].append(pre_action_robot_state)
                rollout["rewards"].append(reward)
                rollout["dones"].append(done)

            if done:
                success = True
                break
            t += 1

    except Exception as e:
        episode_error = f"{type(e).__name__}: {e}"
        log_message(f"Episode error: {episode_error}", log_file)

    return success, replay_images, rollout, episode_initial_state, policy_start_state, episode_error


def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    total_episodes=0,
    total_successes=0,
    total_invalid_episodes=0,
    log_file=None,
):
    """Run evaluation for a single task."""
    # Get task
    task = task_suite.get_task(task_id)

    # Get initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)
    if cfg.initial_states_path == "DEFAULT" and cfg.num_trials_per_task > len(initial_states):
        raise ValueError(
            f"num_trials_per_task={cfg.num_trials_per_task} exceeds the {len(initial_states)} available "
            "initial states for this task. LIBERO-Plus should be evaluated with --num_trials_per_task 1."
        )

    # Initialize environment and get task description
    env, task_description = get_libero_env(
        task,
        cfg.model_family,
        resolution=cfg.env_img_res,
        language_instruction_mode=cfg.language_instruction_mode,
        seed=cfg.seed,
    )

    # Start episodes
    task_episodes, task_successes, task_invalid_episodes = 0, 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)

        # Handle initial state
        if cfg.initial_states_path == "RESET":
            initial_state = None
            initial_state_source = "env_reset"
        elif cfg.initial_states_path == "DEFAULT":
            # Use default initial state
            initial_state = initial_states[episode_idx]
            initial_state_source = "benchmark_initial_state"
        else:
            # Get keys for fetching initial episode state from JSON
            initial_states_task_key = get_initial_states_task_key(
                all_initial_states,
                task_description,
                task.name,
            )
            episode_key = f"demo_{episode_idx}"

            # Skip episode if expert demonstration failed to complete the task
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                continue

            # Get initial state
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])
            initial_state_source = f"custom_json:{cfg.initial_states_path}"

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        # Run episode
        success, replay_images, rollout, episode_initial_state, policy_start_state, episode_error = run_episode(
            cfg,
            env,
            task_description,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            initial_state,
            log_file,
        )

        if episode_error is not None:
            task_invalid_episodes += 1
            total_invalid_episodes += 1
            log_message(
                f"Excluding invalid episode task={task_id} episode={episode_idx} from evaluation and failure bank.",
                log_file,
            )
            continue

        # Update counters
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        # Save replay video
        save_rollout_video(
            replay_images, total_episodes, success=success, task_description=task_description, log_file=log_file
        )

        # Save only the episode types requested by config.
        should_save = (success and cfg.save_success_episodes) or ((not success) and cfg.save_failed_episodes)
        if should_save:

            from libero.libero import get_libero_path

            bddl_file = os.path.join(
                get_libero_path("bddl_files"),
                task.problem_folder,
                task.bddl_file,
            )

            episode_path = save_episode_hdf5(
                cfg=cfg,
                env=env,
                rollout=rollout,
                task_description=task_description,
                task_id=task_id,
                episode_idx=episode_idx,
                initial_state=episode_initial_state,
                policy_start_state=policy_start_state,
                initial_state_source=initial_state_source,
                bddl_file=bddl_file,
                success=success
            )
            episode_kind = "success" if success else "failed"
            log_message(f"Saved {episode_kind} episode to: {episode_path}", log_file)


        # Log results
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    try:
        env.close()
    except Exception as e:
        log_message(f"Environment close error: {type(e).__name__}: {e}", log_file)

    # Log task results
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)
    log_message(f"Invalid episodes for current task: {task_invalid_episodes}", log_file)
    log_message(f"Invalid episodes in total: {total_invalid_episodes}", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{task_description}": task_success_rate,
                f"num_episodes/{task_description}": task_episodes,
                f"invalid_episodes/{task_description}": task_invalid_episodes,
            }
        )

    return total_episodes, total_successes, total_invalid_episodes


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""
    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Initialize model and components
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)
    log_message(f"Language instruction mode: {cfg.language_instruction_mode}", log_file)
    log_message(f"Seed: {cfg.seed}", log_file)

    # Start evaluation
    total_episodes, total_successes, total_invalid_episodes = 0, 0, 0
    for task_id in tqdm.tqdm(range(num_tasks)):
        total_episodes, total_successes, total_invalid_episodes = run_task(
            cfg,
            task_suite,
            task_id,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            total_episodes,
            total_successes,
            total_invalid_episodes,
            log_file,
        )

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Total invalid episodes: {total_invalid_episodes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": final_success_rate,
                "num_episodes/total": total_episodes,
                "invalid_episodes/total": total_invalid_episodes,
            }
        )
        wandb.save(local_log_filepath)

    # Close log file
    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_libero()
