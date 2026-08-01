"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import tree_map
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX
from prismatic.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights


SCENE_PREFIX_RE = re.compile(r"^[a-z_]+_scene\d+_", re.IGNORECASE)
LIBERO_PLUS_TASK_SUFFIX_RE = re.compile(
    r"(_view_.*|_moved_level\d+_sample\d+|_level\d+_sample\d+|"
    r"_initstate_\d+|_(?:add|light|noise|table|tb)_\d+)$",
    re.IGNORECASE,
)
LIBERO_PLUS_TEXT_SUFFIX_RE = re.compile(
    r"\s+(?:view(?:\s+[-+]?\d+(?:\.\d+)?)+.*|moved\s+level\d+\s+sample\d+|"
    r"level\d+\s+sample\d+|initstate\s+\d+|(?:add|light|noise|table|tb)\s+\d+)$",
    re.IGNORECASE,
)


def normalize_libero_language_instruction(language_instruction: str) -> str:
    """Collapse LIBERO-plus task metadata suffixes while leaving normal instructions unchanged."""
    language_instruction = " ".join(str(language_instruction).lower().split())
    if not language_instruction:
        return language_instruction

    if "_" in language_instruction:
        task_name = Path(language_instruction).stem
        had_language_suffix = "_language_" in task_name
        if had_language_suffix:
            task_name = task_name.split("_language_", 1)[0]
        stripped_task_name = LIBERO_PLUS_TASK_SUFFIX_RE.sub("", task_name)
        if stripped_task_name != task_name or had_language_suffix or " " not in stripped_task_name:
            stripped_task_name = SCENE_PREFIX_RE.sub("", stripped_task_name)
            return " ".join(part for part in stripped_task_name.split("_") if part)

    return LIBERO_PLUS_TEXT_SUFFIX_RE.sub("", language_instruction).strip()


def _load_dataset_statistics(dataset_statistics: Any) -> Optional[Dict[str, Any]]:
    if dataset_statistics is None:
        return None
    if isinstance(dataset_statistics, (str, Path)):
        with Path(dataset_statistics).open("r") as f:
            return json.load(f)
    return dataset_statistics


def _select_dataset_statistics(dataset_statistics: Optional[Dict[str, Any]], dataset_name: str) -> Optional[Dict[str, Any]]:
    if dataset_statistics is None:
        return None
    if dataset_name in dataset_statistics:
        return dataset_statistics[dataset_name]
    if "shared_bounds" in dataset_statistics:
        return dataset_statistics["shared_bounds"]
    if {"action", "proprio"}.issubset(dataset_statistics):
        return dataset_statistics
    raise KeyError(
        f"Could not find statistics for dataset `{dataset_name}`. "
        f"Available keys: {sorted(dataset_statistics.keys())}"
    )


def _as_scalar_int(value: Any) -> int:
    return int(np.asarray(value).reshape(-1)[0])


def _decode_scalar_string(value: Any) -> str:
    scalar = np.asarray(value).reshape(-1)[0]
    if isinstance(scalar, bytes):
        return scalar.decode("utf-8")
    if isinstance(scalar, np.bytes_):
        return scalar.tobytes().decode("utf-8")
    return str(scalar)


def _current_timestep(observation: Dict[str, Any]) -> int:
    return int(np.asarray(observation["timestep"]).reshape(-1)[-1])


def _semantic_key_from_language(
    language_instruction: str,
    quantile_data: Dict[str, float],
    semantic_task_map: Optional[Dict[str, str]],
) -> str:
    normalized_language = " ".join(str(language_instruction).lower().split())
    if semantic_task_map is not None and normalized_language in semantic_task_map:
        return semantic_task_map[normalized_language]

    direct_key = normalized_language.replace(" ", "_")
    if direct_key in quantile_data:
        return direct_key

    canonical_language = SCENE_PREFIX_RE.sub("", direct_key)
    canonical_language = LIBERO_PLUS_TASK_SUFFIX_RE.sub("", canonical_language)
    canonical_language = " ".join(part for part in canonical_language.split("_") if part)

    for quantile_key in quantile_data:
        canonical_key = SCENE_PREFIX_RE.sub("", str(quantile_key).lower())
        canonical_key = LIBERO_PLUS_TASK_SUFFIX_RE.sub("", canonical_key)
        canonical_key = " ".join(part for part in canonical_key.split("_") if part)
        if canonical_key == canonical_language:
            return quantile_key

    return direct_key


@dataclass
class RLDSValueBatchTransform:
    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Return one current-frame value example without action-token targets."""
        lang = normalize_libero_language_instruction(rlds_batch["task"]["language_instruction"].decode())
        return_dict = {
            "image_primary": Image.fromarray(rlds_batch["observation"]["image_primary"][0]),
            "image_wrist": Image.fromarray(rlds_batch["observation"]["image_wrist"][0]),
            "language_instruction": lang,
            "proprio": torch.as_tensor(rlds_batch["observation"]["proprio"][0], dtype=torch.float32),
            "reward": torch.as_tensor(rlds_batch["reward"], dtype=torch.float32).squeeze(),
            "return_to_go": torch.as_tensor(rlds_batch["return_to_go"], dtype=torch.float32).squeeze(),
            "dataset_name": rlds_batch["dataset_name"],
        }
        if "episode_id" in rlds_batch:
            return_dict["episode_id"] = _as_scalar_int(rlds_batch["episode_id"])
        if "timestep" in rlds_batch["observation"]:
            return_dict["timestep"] = _current_timestep(rlds_batch["observation"])
        return return_dict


@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    use_wrist_image: bool = False
    use_proprio: bool = False
    recap: bool = False
    advantage_data: Optional[Dict[str, Any]] = None
    quantile_data: Optional[Dict[str, float]] = None
    recap_positive_datasets: Tuple[str, ...] = ()
    recap_semantic_task_map: Optional[Dict[str, str]] = None
    recap_negative_loss_weight: float = 0.0

    def build_prompt(self, lang, action_chunk_string):
        prompt_builder = self.prompt_builder_fn("openvla")

        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": action_chunk_string},
        ]
    
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])
        
        return prompt_builder.get_prompt()

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:

        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, current_action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        dataset_name_str = _decode_scalar_string(dataset_name)
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = normalize_libero_language_instruction(rlds_batch["task"]["language_instruction"].decode())
        actions = rlds_batch["action"]
        episode_id = _as_scalar_int(rlds_batch["episode_id"]) if "episode_id" in rlds_batch else None
        timestep = _current_timestep(rlds_batch["observation"]) if "timestep" in rlds_batch["observation"] else None

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens

        # Get future action chunk
        future_actions = rlds_batch["action"][1:]
        future_actions_string = ''.join(self.action_tokenizer(future_actions))

        # Get action chunk string
        current_action_string = self.action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        adv = None
        recap_loss_weight = 1.0

        if self.recap:
            if dataset_name_str in self.recap_positive_datasets:
                adv = "positive"
            else:
                if episode_id is None or timestep is None:
                    raise KeyError("RECAP advantage lookup requires `episode_id` and observation `timestep`.")
                if dataset_name_str not in self.advantage_data:
                    raise KeyError(f"No RECAP advantage table loaded for dataset `{dataset_name_str}`.")

                key = (episode_id, timestep)
                try:
                    advantage, semantic_task = self.advantage_data[dataset_name_str][key]
                except KeyError as exc:
                    raise KeyError(
                        f"No RECAP advantage found for dataset={dataset_name_str}, "
                        f"episode_id={episode_id}, timestep={timestep}."
                    ) from exc

                if semantic_task is None:
                    semantic_task = _semantic_key_from_language(
                        lang,
                        self.quantile_data,
                        self.recap_semantic_task_map,
                    )
                if semantic_task not in self.quantile_data:
                    raise KeyError(f"No RECAP quantile found for semantic task `{semantic_task}`.")

                quantile = self.quantile_data[semantic_task]
                adv = "positive" if advantage >= quantile else "negative"
            recap_loss_weight = self.recap_negative_loss_weight if adv == "negative" else 1.0

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(self.build_prompt(lang, action_chunk_string), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return_dict = dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name, actions=actions)
        if self.recap:
            return_dict["recap_loss_weight"] = np.asarray(recap_loss_weight, dtype=np.float32)
        if episode_id is not None:
            return_dict["episode_id"] = episode_id
        if timestep is not None:
            return_dict["timestep"] = timestep


        # Add additional inputs
        if self.use_wrist_image:
            all_wrist_pixels = []
            for k in rlds_batch["observation"].keys():
                if "wrist" in k:
                    img_wrist = Image.fromarray(rlds_batch["observation"][k][0])
                    pixel_values_wrist = self.image_transform(img_wrist)
                    all_wrist_pixels.append(pixel_values_wrist)
            return_dict["pixel_values_wrist"] = torch.cat(all_wrist_pixels, dim=0)
        if self.use_proprio and "proprio" in rlds_batch["observation"]:
            proprio = rlds_batch["observation"]["proprio"]
            return_dict["proprio"] = proprio

        return return_dict


class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
        future_action_window_size: int = NUM_ACTIONS_CHUNK - 1,
        include_value_targets: bool = False,
        include_metadata: bool = False,
        dataset_statistics: Optional[Union[Dict[str, Any], str, Path]] = None,
        goal_relabeling_strategy: str | None = "uniform",
        use_shared_action_bounds: bool = False,
        use_shared_proprio_bounds: bool = False,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        if "aloha" in self.data_mix:
            load_camera_views = ("primary", "left_wrist", "right_wrist")
        else:
            load_camera_views = ("primary", "wrist")

        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=load_camera_views,
            load_depth=False,
            load_proprio=True,
            load_language=True,
            action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
        )
        if include_value_targets:
            for dataset_kwargs in per_dataset_kwargs:
                dataset_kwargs["include_value_targets"] = True
        if include_metadata:
            for dataset_kwargs in per_dataset_kwargs:
                dataset_kwargs["include_metadata"] = True
        dataset_statistics = _load_dataset_statistics(dataset_statistics)
        if dataset_statistics is not None:
            for dataset_kwargs in per_dataset_kwargs:
                dataset_kwargs["dataset_statistics"] = _select_dataset_statistics(
                    dataset_statistics,
                    dataset_kwargs["name"],
                )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,                                      # If we wanted to feed / predict more than one step
                future_action_window_size=future_action_window_size,
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy=goal_relabeling_strategy,
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
            use_shared_action_bounds=use_shared_action_bounds,
            use_shared_proprio_bounds=use_shared_proprio_bounds,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class RLDSValueDataset(RLDSDataset):
    """RLDS stream for value targets, with one current observation per example."""

    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSValueBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
        use_shared_action_bounds: bool = False,
        use_shared_proprio_bounds: bool = False,
    ) -> None:
        super().__init__(
            data_root_dir,
            data_mix,
            batch_transform,
            resize_resolution,
            shuffle_buffer_size=shuffle_buffer_size,
            train=train,
            image_aug=image_aug,
            future_action_window_size=0,
            include_value_targets=True,
            goal_relabeling_strategy=None,
            use_shared_action_bounds=use_shared_action_bounds,
            use_shared_proprio_bounds=use_shared_proprio_bounds,
        )


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out


class DummyDataset(Dataset):
    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        self.dataset_statistics = {
            "dummy_dataset": {
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }

    def __len__(self):
        # TODO =>> Replace with number of elements in your dataset!
        return 10000

    def __getitem__(self, idx):
        # TODO =>> Load image, action and instruction from disk -- we use dummy values
        image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
        action = np.asarray(np.random.rand(7), dtype=np.float32)
        instruction = "do something spectacular"

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
