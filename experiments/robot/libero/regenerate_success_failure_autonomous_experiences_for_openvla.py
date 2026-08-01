"""
Regenerate failed_episodes and success_episodes into OpenVLA-style HDF5 datasets
for VLA-collected autonomous experiences.

This reuses the low-level replay helpers from regenerate_recovery_dataset_for_openvla.py
and routes each replayed episode by its actual replay outcome:

    python experiments/robot/libero/regenerate_success_failure_autonomous_experiences_for_openvla.py \
        --libero_task_suite libero_spatial \
        --overwrite

By default this reads:
    failed_episodes
    success_episodes

and writes replay failures/successes into:
    autonomous_experience_openvla_dataset/libero_spatial_autonomous_failure_no_noops
    autonomous_experience_openvla_dataset/libero_spatial_autonomous_success_no_noops
"""

import argparse
import json
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import tqdm

from regenerate_recovery_dataset_for_openvla import (
    IMAGE_RESOLUTION,
    discover_episode_sources,
    ensure_outputs_available,
    get_libero_env,
    replay_episode,
)

from libero.libero import benchmark


DEFAULT_MAX_EPISODE_LENGTH = 250


@dataclass(frozen=True)
class SourceJob:
    name: str
    source_dir: Path


@dataclass(frozen=True)
class OutcomeDataset:
    name: str
    replay_success: bool
    target_dir: Path
    metainfo_json: Path


def read_source_success(source):
    with h5py.File(source.path, "r") as source_file:
        demo = source_file[f"data/{source.demo_key}"]
        if "success" in demo.attrs:
            return parse_bool(demo.attrs["success"])
        data = source_file["data"]
        if "success" in data.attrs:
            return parse_bool(data.attrs["success"])
    return None


def parse_bool(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(value)

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    return None


def limit_sources_by_episode_count(sources_by_task, max_episodes):
    if max_episodes is None:
        return sources_by_task

    limited = {}
    remaining = max_episodes
    for task_name in sorted(sources_by_task):
        if remaining <= 0:
            break
        sources = sources_by_task[task_name][:remaining]
        if sources:
            limited[task_name] = sources
            remaining -= len(sources)
    return limited


def write_labeled_episode(
    group,
    demo_key,
    replay,
    source,
    label_success,
    source_job,
    source_success,
    num_noops,
):
    num_samples = len(replay["actions"])
    dones = np.zeros(num_samples, dtype=np.uint8)
    rewards = np.zeros(num_samples, dtype=np.float32)
    if label_success:
        dones[-1] = 1
        rewards[-1] = 1.0

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
    episode_group.attrs["success"] = bool(label_success)
    episode_group.attrs["replay_success"] = bool(replay["success"])
    episode_group.attrs["label_source"] = "replay_success"
    episode_group.attrs["source_outcome"] = source_job.name
    episode_group.attrs["post_set_init_dummy_steps"] = int(
        replay["post_set_init_dummy_steps"]
    )
    episode_group.attrs["source_success"] = (
        "unknown" if source_success is None else bool(source_success)
    )
    episode_group.attrs["source_file"] = str(source.path)
    episode_group.attrs["source_demo"] = source.demo_key
    episode_group.attrs["filtered_noops"] = int(num_noops)
    episode_group.attrs["max_episode_length"] = int(replay["max_episode_length"])
    episode_group.attrs["truncated"] = bool(replay.get("truncated", False))


def source_label(source_job):
    if source_job.name == "failure":
        return "failed_episodes"
    if source_job.name == "success":
        return "success_episodes"
    return source_job.name


def count_source_labels(source_entries):
    counts = {"failed_episodes": 0, "success_episodes": 0}
    for source_job, _ in source_entries:
        counts[source_label(source_job)] = counts.get(source_label(source_job), 0) + 1
    return counts


def print_discovery(source_job, sources_by_task, demo_paths, print_task_limit):
    num_episodes = sum(len(sources) for sources in sources_by_task.values())
    print(
        f"[source {source_label(source_job)}] files={len(demo_paths)}, "
        f"tasks={len(sources_by_task)}, episodes={num_episodes}, "
        f"dir={source_job.source_dir}"
    )
    if print_task_limit > 0:
        for task_name in sorted(sources_by_task)[:print_task_limit]:
            print(f"  {task_name}: {len(sources_by_task[task_name])} episode(s)")
        hidden = max(0, len(sources_by_task) - print_task_limit)
        if hidden:
            print(f"  ... {hidden} more task(s)")


def collect_sources(source_jobs, args, tasks_by_name):
    all_sources_by_task = {}
    total_raw_files = 0

    for source_job in source_jobs:
        sources_by_task, demo_paths = discover_episode_sources(
            source_job.source_dir, tasks_by_name
        )
        sources_by_task = limit_sources_by_episode_count(
            sources_by_task, args.max_episodes_per_outcome
        )
        print_discovery(source_job, sources_by_task, demo_paths, args.print_task_limit)

        total_raw_files += len(demo_paths)
        for task_name in sorted(sources_by_task):
            all_sources_by_task.setdefault(task_name, [])
            all_sources_by_task[task_name].extend(
                (source_job, source) for source in sources_by_task[task_name]
            )

    return all_sources_by_task, total_raw_files


def create_data_group(output_file, task, task_description, args, outcome):
    data_group = output_file.create_group("data")
    data_group.attrs["language_instruction"] = task_description
    data_group.attrs["libero_task_suite"] = args.libero_task_suite
    data_group.attrs["task_name"] = task.name
    data_group.attrs["experience_type"] = "vla_autonomous"
    data_group.attrs["dataset_outcome"] = outcome.name
    data_group.attrs["success"] = bool(outcome.replay_success)
    data_group.attrs["replay_success"] = bool(outcome.replay_success)
    data_group.attrs["label_source"] = "replay_success"
    data_group.attrs["max_episode_length"] = int(args.max_episode_length)
    return data_group


def run_routed_jobs(source_jobs, outcome_datasets, args, tasks_by_name):
    sources_by_task, total_raw_files = collect_sources(source_jobs, args, tasks_by_name)

    summary = {
        "raw_files": total_raw_files,
        "tasks": len(sources_by_task),
        "source_failed_episodes": 0,
        "source_success_episodes": 0,
        "replayed": 0,
        "written": 0,
        "written_success": 0,
        "written_failure": 0,
        "replay_success": 0,
        "replay_failure": 0,
        "rerouted": 0,
        "skipped": 0,
        "filtered_noops": 0,
        "truncated": 0,
    }
    for source_entries in sources_by_task.values():
        source_counts = count_source_labels(source_entries)
        summary["source_failed_episodes"] += source_counts["failed_episodes"]
        summary["source_success_episodes"] += source_counts["success_episodes"]

    if args.dry_run:
        return summary

    output_paths = [
        outcome.target_dir / f"{task_name}_demo.hdf5"
        for task_name in sources_by_task
        for outcome in outcome_datasets.values()
    ]
    ensure_outputs_available(output_paths, args.overwrite)
    for outcome in outcome_datasets.values():
        outcome.target_dir.mkdir(parents=True, exist_ok=True)

    metainfo_by_outcome = {outcome.name: {} for outcome in outcome_datasets.values()}
    for task_name in tqdm.tqdm(sorted(sources_by_task), desc="replay-and-route"):
        task = tasks_by_name[task_name]
        env, task_description = get_libero_env(task, resolution=args.image_resolution)
        task_source_counts = count_source_labels(sources_by_task[task_name])
        task_skipped = 0
        task_rerouted = 0
        task_truncated = 0
        outputs = {}

        try:
            with ExitStack() as stack:
                for replay_success, outcome in outcome_datasets.items():
                    output_path = outcome.target_dir / f"{task.name}_demo.hdf5"
                    output_file = stack.enter_context(h5py.File(output_path, "w"))
                    outputs[replay_success] = {
                        "outcome": outcome,
                        "group": create_data_group(
                            output_file, task, task_description, args, outcome
                        ),
                        "metadata": {},
                        "written": 0,
                        "total_samples": 0,
                    }

                for attempt_index, (source_job, source) in enumerate(
                    sources_by_task[task_name]
                ):
                    replay, num_noops = replay_episode(
                        env,
                        source,
                        post_set_init_dummy_steps=args.post_set_init_dummy_steps,
                        max_episode_length=args.max_episode_length,
                    )
                    source_success = read_source_success(source)
                    summary["replayed"] += 1
                    summary["filtered_noops"] += int(num_noops)
                    if replay is not None and replay.get("truncated", False):
                        summary["truncated"] += 1
                        task_truncated += 1

                    replay_success = replay is not None and bool(replay["success"])
                    skipped_reason = None
                    if replay is None:
                        skipped_reason = "empty_after_noop_filter"

                    attempt_key = f"attempt_{attempt_index}"
                    base_metadata = {
                        "source_file": str(source.path),
                        "source_demo": source.demo_key,
                        "source_outcome": source_job.name,
                        "source_success": source_success,
                        "replay_success": replay_success,
                        "skipped_reason": skipped_reason,
                        "filtered_noops": int(num_noops),
                        "post_set_init_dummy_steps": args.post_set_init_dummy_steps,
                        "max_episode_length": args.max_episode_length,
                        "truncated": (
                            bool(replay.get("truncated", False))
                            if replay is not None
                            else False
                        ),
                        "num_samples": (
                            len(replay["actions"]) if replay is not None else 0
                        ),
                    }

                    if skipped_reason is not None:
                        summary["skipped"] += 1
                        task_skipped += 1
                        continue

                    outcome_state = outputs[replay_success]
                    outcome = outcome_state["outcome"]
                    label_success = outcome.replay_success
                    demo_key = f"demo_{outcome_state['written']}"
                    output_metadata = {
                        **base_metadata,
                        "demo_key": demo_key,
                        "label_success": label_success,
                        "routed_outcome": outcome.name,
                        "written": True,
                    }
                    outcome_state["metadata"][attempt_key] = output_metadata

                    write_labeled_episode(
                        outcome_state["group"],
                        demo_key,
                        replay,
                        source,
                        label_success,
                        source_job,
                        source_success,
                        num_noops,
                    )
                    outcome_state["written"] += 1
                    outcome_state["total_samples"] += len(replay["actions"])
                    summary["written"] += 1
                    summary[f"written_{outcome.name}"] += 1
                    summary[f"replay_{outcome.name}"] += 1
                    if (source_job.name == "success") != replay_success:
                        summary["rerouted"] += 1
                        task_rerouted += 1

                task_key = task_description.replace(" ", "_")
                for replay_success, outcome_state in outputs.items():
                    data_group = outcome_state["group"]
                    data_group.attrs["num_demos"] = outcome_state["written"]
                    data_group.attrs["total"] = outcome_state["total_samples"]
                    outcome = outcome_state["outcome"]
                    metainfo_by_outcome[outcome.name][task_key] = outcome_state[
                        "metadata"
                    ]
        finally:
            env.close()

        print(
            f"[task] {task.name}: "
            f"from failed_episodes={task_source_counts['failed_episodes']}, "
            f"success_episodes={task_source_counts['success_episodes']} -> "
            f"success={outputs[True]['written']}, failure={outputs[False]['written']}, "
            f"skipped={task_skipped}, truncated={task_truncated}, "
            f"rerouted={task_rerouted}"
        )

    for outcome in outcome_datasets.values():
        outcome.metainfo_json.parent.mkdir(parents=True, exist_ok=True)
        with open(outcome.metainfo_json, "w", encoding="utf-8") as metainfo_file:
            json.dump(metainfo_by_outcome[outcome.name], metainfo_file, indent=2)
        print(f"[metainfo {outcome.name}] {outcome.metainfo_json}")
    return summary


def build_source_jobs(args):
    source_jobs = []

    if args.only in {"both", "failure"}:
        source_jobs.append(
            SourceJob(
                name="failure",
                source_dir=Path(args.failed_episode_dir).expanduser().resolve(),
            )
        )

    if args.only in {"both", "success"}:
        source_jobs.append(
            SourceJob(
                name="success",
                source_dir=Path(args.success_episode_dir).expanduser().resolve(),
            )
        )

    return source_jobs


def build_outcome_datasets(args):
    output_root = Path(args.output_root).expanduser().resolve()

    failed_target = (
        Path(args.failed_target_dir).expanduser().resolve()
        if args.failed_target_dir
        else output_root / f"{args.libero_task_suite}_autonomous_failure_no_noops"
    )
    success_target = (
        Path(args.success_target_dir).expanduser().resolve()
        if args.success_target_dir
        else output_root / f"{args.libero_task_suite}_autonomous_success_no_noops"
    )

    return {
        False: OutcomeDataset(
            name="failure",
            replay_success=False,
            target_dir=failed_target,
            metainfo_json=failed_target / f"{args.libero_task_suite}_failure_metainfo.json",
        ),
        True: OutcomeDataset(
            name="success",
            replay_success=True,
            target_dir=success_target,
            metainfo_json=success_target / f"{args.libero_task_suite}_success_metainfo.json",
        ),
    }

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate failed/success LIBERO autonomous experiences into "
            "OpenVLA HDF5 datasets routed by actual replay outcome."
        )
    )
    parser.add_argument(
        "--libero_task_suite",
        default="libero_spatial",
        choices=[
            "libero_spatial",
            "libero_object",
            "libero_goal",
            "libero_10",
            "libero_90",
        ],
    )
    parser.add_argument("--failed_episode_dir", default="failed_episodes")
    parser.add_argument("--success_episode_dir", default="success_episodes")
    parser.add_argument("--failed_target_dir")
    parser.add_argument("--success_target_dir")
    parser.add_argument("--output_root", default="autonomous_experience_openvla_dataset")
    parser.add_argument(
        "--only",
        choices=["both", "failure", "success"],
        default="both",
        help=(
            "Source episode directory/directories to read. Replayed episodes are "
            "still routed to success/failure target datasets by replay outcome."
        ),
    )
    parser.add_argument("--image_resolution", type=int, default=IMAGE_RESOLUTION)
    parser.add_argument("--max_episodes_per_outcome", type=int)
    parser.add_argument(
        "--max_episode_length",
        type=int,
        default=DEFAULT_MAX_EPISODE_LENGTH,
        help="Maximum number of recorded steps to keep in each replayed episode.",
    )
    parser.add_argument("--print_task_limit", type=int, default=0)
    parser.add_argument(
        "--post_set_init_dummy_steps",
        type=int,
        default=10,
        help=(
            "Number of no-op dummy env.step calls to run immediately after "
            "set_init_state and before recording/replaying policy actions."
        ),
    )
    parser.add_argument(
        "--strict_replay_success",
        action="store_true",
        help=(
            "Deprecated compatibility flag. Routing is always based on replay outcome."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main(args):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_task_suite]()
    tasks_by_name = {task.name: task for task in task_suite.tasks}

    if args.strict_replay_success:
        print(
            "--strict_replay_success is ignored: episodes are always routed by "
            "actual replay outcome."
        )

    source_jobs = build_source_jobs(args)
    outcome_datasets = build_outcome_datasets(args)
    summary = run_routed_jobs(source_jobs, outcome_datasets, args, tasks_by_name)

    print("Done.")
    print(
        f"  sources: raw_files={summary['raw_files']}, tasks={summary['tasks']}, "
        f"failed_episodes={summary['source_failed_episodes']}, "
        f"success_episodes={summary['source_success_episodes']}"
    )
    print(
        f"  replay: replayed={summary['replayed']}, "
        f"success={summary['replay_success']}, failure={summary['replay_failure']}, "
        f"skipped={summary['skipped']}, truncated={summary['truncated']}, "
        f"filtered_noops={summary['filtered_noops']}"
    )
    print(
        f"  written: success={summary['written_success']}, "
        f"failure={summary['written_failure']}, total={summary['written']}, "
        f"rerouted_from_source_label={summary['rerouted']}"
    )


if __name__ == "__main__":
    main(parse_args())
