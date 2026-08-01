"""
Generate OpenVLA-compatible LIBERO datasets from recollected raw demos.

The input is a directory such as ``recovery_demonstration_data`` containing
one or more nested ``demo.hdf5`` files. Each input file must contain:

    /data                                  (group)
    /data.attrs["bddl_file_name"]           (path or filename of the task BDDL)
    /data/<demo_key>/states                (MuJoCo simulator states)
    /data/<demo_key>/actions               (LIBERO EEF actions)

The script replays every raw demo with the same environment setup used by
OpenVLA regeneration and writes the final regenerated HDF5 files in one pass.

Example:
    python experiments/robot/libero/regenerate_recovery_dataset_for_openvla.py \
        --libero_task_suite libero_spatial \
        --recovery_demo_dir recovery_demonstration_data \
        --libero_target_dir /tmp/libero_spatial_recovery_no_noops \
        --overwrite
"""

"""
python experiments/robot/libero/regenerate_recovery_dataset_for_openvla.py \
  --libero_task_suite libero_spatial \
  --recovery_demo_dir recovery_demonstration_data \
  --libero_target_dir recovery_openvla_dataset/libero_spatial_no_noops \
  --dry_run

python experiments/robot/libero/regenerate_recovery_dataset_for_openvla.py \
  --libero_task_suite libero_spatial \
  --recovery_demo_dir recovery_demonstration_data \
  --libero_target_dir recovery_openvla_dataset/libero_spatial_no_noops \
  --overwrite

"""

import argparse
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import robosuite.utils.transform_utils as T
import tqdm

import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


IMAGE_RESOLUTION = 256
NOOP_SKIP_WINDOW = 5
DUMMY_GRIPPER_COMMAND = -1.0
SCENE_PREFIX_RE = re.compile(r"^[A-Z_]+_SCENE\d+_")
LIBERO_PLUS_SUFFIX_RE = re.compile(
    r"(_view_.*|_moved_level\d+_sample\d+|_level\d+_sample\d+|"
    r"_initstate_\d+|_(?:add|light|noise|table|tb)_\d+)$"
)


@dataclass(frozen=True)
class EpisodeSource:
    path: Path
    demo_key: str


def is_noop(action, prev_action=None, threshold=1e-4):
    """Match OpenVLA's no-op filter for regenerated LIBERO datasets."""
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold

    return (
        np.linalg.norm(action[:-1]) < threshold
        and action[-1] == prev_action[-1]
    )


def starts_noop_run(actions, start_index, prev_action, window_size=NOOP_SKIP_WINDOW):
    """Return whether the next window of actions is an uninterrupted no-op run."""
    window = actions[start_index : start_index + window_size]
    if len(window) < window_size:
        return False

    reference_action = prev_action
    for action in window:
        if not is_noop(action, reference_action):
            return False
        reference_action = action
    return True


def count_leading_noops(actions):
    """Count no-op actions at the start of an episode without requiring a full window."""
    num_leading_noops = 0
    prev_action = None

    for action in actions:
        if not is_noop(action, prev_action):
            break
        num_leading_noops += 1
        prev_action = action

    return num_leading_noops


def decode_attr(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def sort_demo_keys(keys):
    def key(name):
        try:
            return (0, int(name.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            return (1, name)

    return sorted(keys, key=key)


def clip_action_array(actions):
    """Clip LIBERO EEF action dimensions while leaving the gripper command unchanged."""
    clipped_actions = np.array(actions, copy=True)
    if clipped_actions.shape[-1] < 7:
        raise ValueError(
            f"Expected LIBERO EEF action dimension >= 7, got {clipped_actions.shape[-1]}"
        )

    clipped_actions[..., :6] = np.clip(clipped_actions[..., :6], -1.0, 1.0)
    return clipped_actions


def get_libero_dummy_action(action_dim):
    """Return a LIBERO no-op action matching the replay action dimension."""
    dummy_action = np.zeros(action_dim, dtype=np.float32)
    dummy_action[-1] = DUMMY_GRIPPER_COMMAND
    return dummy_action


def run_unrecorded_dummy_steps(env, obs, action_dim, num_steps):
    dummy_action = get_libero_dummy_action(action_dim)
    for _ in range(num_steps):
        obs, _, _, _ = env.step(dummy_action.tolist())
    return obs


def canonical_task_name(task_name):
    """Strip LIBERO-plus variation suffixes back to the base task name."""
    task_stem = Path(task_name).stem
    if "_language_" in task_stem:
        task_stem = task_stem.split("_language_", 1)[0]
    task_stem = LIBERO_PLUS_SUFFIX_RE.sub("", task_stem)
    return task_stem


def language_from_task_name(task_name):
    base_task = canonical_task_name(task_name)
    base_task = SCENE_PREFIX_RE.sub("", base_task)
    return " ".join(base_task.split("_"))


def bddl_language_instruction(bddl_file):
    problem_info = BDDLUtils.get_problem_info(bddl_file)
    return problem_info["language_instruction"]


def language_bddl_file(task, bddl_file):
    bddl_path = Path(bddl_file)
    if bddl_path.exists():
        return bddl_path
    if "_language_" in task.name and "_view_" in task.name:
        candidate = bddl_path.with_name(f"{task.name.split('_view_', 1)[0]}.bddl")
        if candidate.exists():
            return candidate
    return bddl_path


def get_official_language_instruction(task, bddl_file):
    """Match official LIBERO language metadata while collapsing plus variants."""
    if "_language_" in task.name:
        return bddl_language_instruction(language_bddl_file(task, bddl_file))
    return language_from_task_name(task.name)


def get_libero_env(task, resolution=IMAGE_RESOLUTION):
    """Create an environment using the same setup as OpenVLA regeneration."""
    bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    task_description = get_official_language_instruction(task, bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(0)
    return env, task_description


def discover_episode_sources(recovery_demo_dir, task_names):
    demo_paths = sorted(recovery_demo_dir.rglob("demo.hdf5"))
    if not demo_paths:
        raise FileNotFoundError(
            f"No demo.hdf5 files found recursively under {recovery_demo_dir}"
        )

    sources_by_task = defaultdict(list)
    for demo_path in demo_paths:
        with h5py.File(demo_path, "r") as source_file:
            if "data" not in source_file:
                raise KeyError(f"Missing /data group in {demo_path}")

            raw_data = source_file["data"]
            if "bddl_file_name" not in raw_data.attrs:
                raise KeyError(f"Missing /data bddl_file_name attribute in {demo_path}")

            task_name = Path(decode_attr(raw_data.attrs["bddl_file_name"])).stem
            if task_name not in task_names:
                raise ValueError(
                    f"Task '{task_name}' from {demo_path} is not in the selected task suite."
                )

            for demo_key in sort_demo_keys(raw_data.keys()):
                raw_demo = raw_data[demo_key]
                if "states" not in raw_demo or "actions" not in raw_demo:
                    raise KeyError(
                        f"Missing states/actions in {demo_path}:/data/{demo_key}"
                    )
                sources_by_task[task_name].append(EpisodeSource(demo_path, demo_key))

    return sources_by_task, demo_paths


def get_robot_state_from_obs(obs):
    return np.concatenate(
        [
            obs["robot0_gripper_qpos"],
            obs["robot0_eef_pos"],
            obs["robot0_eef_quat"],
        ]
    )


def replay_episode(env, source, post_set_init_dummy_steps=0, max_episode_length=None):
    if post_set_init_dummy_steps < 0:
        raise ValueError(
            f"post_set_init_dummy_steps must be >= 0, got {post_set_init_dummy_steps}"
        )
    if max_episode_length is not None and max_episode_length <= 0:
        raise ValueError(
            f"max_episode_length must be > 0, got {max_episode_length}"
        )

    with h5py.File(source.path, "r") as source_file:
        raw_demo = source_file[f"data/{source.demo_key}"]
        orig_actions = np.asarray(raw_demo["actions"][()])
        orig_states = np.asarray(raw_demo["states"][()])

    if len(orig_actions) == 0 or len(orig_states) == 0:
        return None, 0

    replay_actions = clip_action_array(orig_actions)
    start_index = count_leading_noops(replay_actions)
    if start_index >= len(replay_actions):
        return None, start_index
    if start_index >= len(orig_states):
        raise ValueError(
            f"Cannot align replay start at action {start_index}; "
            f"only {len(orig_states)} states are available in {source.path}:/{source.demo_key}"
        )

    env.reset()
    obs = env.set_init_state(orig_states[start_index])
    if post_set_init_dummy_steps:
        obs = run_unrecorded_dummy_steps(
            env, obs, replay_actions.shape[-1], post_set_init_dummy_steps
        )
    replay_initial_state = env.sim.get_state().flatten()

    states = []
    actions = []
    ee_states = []
    gripper_states = []
    joint_states = []
    robot_states = []
    agentview_images = []
    eye_in_hand_images = []
    num_noops = start_index
    done = False
    truncated = False

    for action_index, action in enumerate(
        replay_actions[start_index:], start=start_index
    ):
        prev_action = actions[-1] if actions else None
        if actions and starts_noop_run(replay_actions, action_index, prev_action):
            num_noops += 1
            continue

        states.append(env.sim.get_state().flatten())
        actions.append(action)
        robot_states.append(get_robot_state_from_obs(obs))
        if "robot0_gripper_qpos" in obs:
            gripper_states.append(obs["robot0_gripper_qpos"])
        joint_states.append(obs["robot0_joint_pos"])
        ee_states.append(
            np.hstack(
                (
                    obs["robot0_eef_pos"],
                    T.quat2axisangle(obs["robot0_eef_quat"]),
                )
            )
        )
        agentview_images.append(obs["agentview_image"])
        eye_in_hand_images.append(obs["robot0_eye_in_hand_image"])

        obs, _, done, _ = env.step(action.tolist())
        if max_episode_length is not None and len(actions) >= max_episode_length:
            truncated = action_index < len(replay_actions) - 1
            break

    if not actions:
        return None, num_noops

    replay = {
        "success": bool(done),
        "truncated": bool(truncated),
        "max_episode_length": max_episode_length,
        "initial_state": replay_initial_state,
        "source_initial_state": orig_states[start_index],
        "post_set_init_dummy_steps": int(post_set_init_dummy_steps),
        "states": np.stack(states),
        "actions": np.asarray(actions),
        "ee_states": np.stack(ee_states),
        "gripper_states": np.stack(gripper_states),
        "joint_states": np.stack(joint_states),
        "robot_states": np.stack(robot_states),
        "agentview_images": np.stack(agentview_images),
        "eye_in_hand_images": np.stack(eye_in_hand_images),
    }
    return replay, num_noops


def write_successful_episode(group, demo_key, replay):
    num_samples = len(replay["actions"])
    dones = np.zeros(num_samples, dtype=np.uint8)
    rewards = np.zeros(num_samples, dtype=np.uint8)
    dones[-1] = 1
    rewards[-1] = 1

    episode_group = group.create_group(demo_key)
    obs_group = episode_group.create_group("obs")
    obs_group.create_dataset("gripper_states", data=replay["gripper_states"])
    obs_group.create_dataset("joint_states", data=replay["joint_states"])
    obs_group.create_dataset("ee_states", data=replay["ee_states"])
    obs_group.create_dataset("ee_pos", data=replay["ee_states"][:, :3])
    obs_group.create_dataset("ee_ori", data=replay["ee_states"][:, 3:])
    obs_group.create_dataset("agentview_rgb", data=replay["agentview_images"])
    obs_group.create_dataset("eye_in_hand_rgb", data=replay["eye_in_hand_images"])
    episode_group.create_dataset("actions", data=replay["actions"])
    episode_group.create_dataset("states", data=replay["states"])
    episode_group.create_dataset("robot_states", data=replay["robot_states"])
    episode_group.create_dataset("rewards", data=rewards)
    episode_group.create_dataset("dones", data=dones)
    episode_group.attrs["num_samples"] = num_samples
    episode_group.attrs["post_set_init_dummy_steps"] = int(
        replay["post_set_init_dummy_steps"]
    )


def ensure_outputs_available(output_paths, overwrite):
    existing_paths = [path for path in output_paths if path.exists()]
    if existing_paths and not overwrite:
        formatted_paths = "\n".join(f"  {path}" for path in existing_paths)
        raise FileExistsError(
            "The following output file(s) already exist. Pass --overwrite to "
            f"replace them:\n{formatted_paths}"
        )


def main(args):
    recovery_demo_dir = Path(args.recovery_demo_dir).expanduser().resolve()
    target_dir = Path(args.libero_target_dir).expanduser().resolve()

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_task_suite]()
    tasks_by_name = {task.name: task for task in task_suite.tasks}
    sources_by_task, demo_paths = discover_episode_sources(
        recovery_demo_dir, tasks_by_name
    )

    print(f"Found {len(demo_paths)} raw demo.hdf5 files under {recovery_demo_dir}")
    print(f"Found recovery episodes for {len(sources_by_task)} task(s)")
    for task_name in sorted(sources_by_task):
        print(f"  {task_name}: {len(sources_by_task[task_name])} episode(s)")

    if args.dry_run:
        return

    output_paths = [
        target_dir / f"{task_name}_demo.hdf5" for task_name in sources_by_task
    ]
    ensure_outputs_available(output_paths, args.overwrite)
    target_dir.mkdir(parents=True, exist_ok=True)

    metainfo = {}
    total_replays = 0
    total_success = 0
    total_noops = 0

    for task_name in tqdm.tqdm(sorted(sources_by_task)):
        task = tasks_by_name[task_name]
        sources = sources_by_task[task_name]
        output_path = target_dir / f"{task.name}_demo.hdf5"

        env, task_description = get_libero_env(task, resolution=args.image_resolution)
        task_metadata = {}
        task_success = 0
        task_total_samples = 0

        try:
            with h5py.File(output_path, "w") as output_file:
                group = output_file.create_group("data")
                group.attrs["language_instruction"] = task_description
                group.attrs["libero_task_suite"] = args.libero_task_suite
                group.attrs["task_name"] = task.name
                for attempt_index, source in enumerate(sources):
                    replay, num_noops = replay_episode(
                        env,
                        source,
                        post_set_init_dummy_steps=args.post_set_init_dummy_steps,
                    )
                    total_replays += 1
                    total_noops += num_noops

                    success = replay is not None and replay["success"]
                    attempt_key = f"demo_{attempt_index}"
                    task_metadata[attempt_key] = {
                        "source_file": str(source.path),
                        "source_demo": source.demo_key,
                        "success": bool(success),
                        "post_set_init_dummy_steps": args.post_set_init_dummy_steps,
                        "initial_state": (
                            replay["initial_state"].tolist()
                            if replay is not None
                            else None
                        ),
                    }

                    if not success:
                        continue

                    output_key = f"demo_{task_success}"
                    write_successful_episode(group, output_key, replay)
                    task_total_samples += len(replay["actions"])
                    task_success += 1
                    total_success += 1

                group.attrs["num_demos"] = task_success
                group.attrs["total"] = task_total_samples
        finally:
            env.close()

        metainfo[task_description.replace(" ", "_")] = task_metadata
        print(
            f"Saved {task_success}/{len(sources)} successful demos for "
            f"'{task_description}' at {output_path}"
        )

    metainfo_path = (
        Path(args.metainfo_json).expanduser().resolve()
        if args.metainfo_json
        else target_dir / f"{args.libero_task_suite}_metainfo.json"
    )
    metainfo_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metainfo_path, "w", encoding="utf-8") as metainfo_file:
        json.dump(metainfo, metainfo_file, indent=2)

    print(
        f"Completed: {total_success}/{total_replays} successful demos; "
        f"filtered {total_noops} no-op actions."
    )
    print(f"Regenerated dataset directory: {target_dir}")
    print(f"Metainfo JSON: {metainfo_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Regenerate all recollected LIBERO demos for OpenVLA in one pass."
    )
    parser.add_argument(
        "--libero_task_suite",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
        required=True,
    )
    parser.add_argument(
        "--recovery_demo_dir",
        required=True,
        help=(
            "Directory recursively containing demo.hdf5 files. Each file must contain "
            "/data, /data.attrs['bddl_file_name'], /data/<demo_key>/states, and "
            "/data/<demo_key>/actions."
        ),
    )
    parser.add_argument(
        "--libero_target_dir",
        required=True,
        help="Directory for OpenVLA-compatible regenerated *_demo.hdf5 files.",
    )
    parser.add_argument("--image_resolution", type=int, default=IMAGE_RESOLUTION)
    parser.add_argument("--metainfo_json")
    parser.add_argument(
        "--post_set_init_dummy_steps",
        type=int,
        default=0,
        help=(
            "Number of no-op dummy env.step calls to run immediately after "
            "set_init_state and before recording/replaying actions."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace task output files that already exist.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only discover and group input demos; do not launch environments or write files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
