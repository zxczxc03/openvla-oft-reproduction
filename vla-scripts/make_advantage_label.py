import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import draccus
import numpy as np
import torch
import tqdm
from transformers import AutoProcessor, Idefics3ForConditionalGeneration

from libero_language_to_task import NORMALIZED_LANGUAGE_TO_BASE_TASKS, normalize_language

from prismatic.models.action_heads import MLPResNet
from prismatic.models.projectors import ProprioProjector
from prismatic.util.data_utils import PaddedCollatorForValueFunction
from prismatic.vla.constants import PROPRIO_DIM
from prismatic.vla.datasets import EpisodicRLDSDataset, RLDSValueBatchTransform


def find_checkpoint_file(checkpoint_dir: Union[str, Path], name_fragment: str) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint path must be a directory: {checkpoint_dir}")

    matches = sorted(
        path
        for path in checkpoint_dir.iterdir()
        if path.is_file() and name_fragment in path.name and "checkpoint" in path.name
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one `{name_fragment}` checkpoint in {checkpoint_dir}, found {len(matches)}."
        )
    return matches[0]


@dataclasses.dataclass
class VfGenerateConfig:
    data_root_dir: Path = Path("datasets/rlds")
    dataset_name: str = "libero_spatial_no_noops"
    pretrained_checkpoint: Union[str, Path] = ""

    advantage_data_dir: Path = Path("./advantage_data")
    split: str = "train"
    image_size: int = 512
    batch_size: int = 128
    max_length: int = 512
    max_episodes: Optional[int] = None
    dataset_statistics_path: Optional[Path] = None

    # Keep advantage generation deterministic. Training-time augmentation should not be used for labels.
    image_aug: bool = False
    use_shared_proprio_bounds: bool = True

    # `mc`: return_to_go - V(s). `n_step`: scaled n-step TD advantage.
    advantage_method: str = "n_step"
    n_step: int = 30
    gamma: float = 1.0
    return_scale: float = 250.0


def decode_scalar_string(value: Any) -> str:
    scalar = np.asarray(value).reshape(-1)[0]
    if isinstance(scalar, bytes):
        return scalar.decode("utf-8")
    if isinstance(scalar, np.bytes_):
        return scalar.tobytes().decode("utf-8")
    return str(scalar)


def load_dataset_statistics_for_labels(cfg: VfGenerateConfig, *, verbose: bool) -> Optional[Dict[str, Any]]:
    statistics_path = cfg.dataset_statistics_path
    if statistics_path is None and cfg.pretrained_checkpoint:
        candidate = Path(cfg.pretrained_checkpoint) / "dataset_statistics.json"
        if candidate.exists():
            statistics_path = candidate

    if statistics_path is None:
        if cfg.use_shared_proprio_bounds and verbose:
            print(
                "No dataset_statistics.json found for advantage generation; "
                "shared bounds will be recomputed from the requested label dataset."
            )
        return None

    with Path(statistics_path).open("r") as f:
        dataset_statistics = json.load(f)
    if verbose:
        print(f"Using dataset statistics for advantage labels: {statistics_path}")
    return dataset_statistics


def initialize_model(cfg: VfGenerateConfig, device: torch.device, *, verbose: bool):
    cfg.pretrained_checkpoint = str(cfg.pretrained_checkpoint).rstrip("/")
    if not cfg.pretrained_checkpoint:
        raise ValueError("`pretrained_checkpoint` must point to a value-function checkpoint directory.")

    if verbose:
        print(f"Loading value-function checkpoint: {cfg.pretrained_checkpoint}")
    processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint)

    vlm = Idefics3ForConditionalGeneration.from_pretrained(
        cfg.pretrained_checkpoint,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    text_config = getattr(vlm.config, "text_config", vlm.config)
    llm_dim = text_config.hidden_size

    value_head = MLPResNet(num_blocks=2, input_dim=llm_dim, hidden_dim=llm_dim, output_dim=1)
    proprio_projector = ProprioProjector(llm_dim=llm_dim, proprio_dim=PROPRIO_DIM)

    proprio_tokens = ["<proprio_0>"]
    additional_special_tokens = list(
        processor.tokenizer.special_tokens_map.get("additional_special_tokens", [])
    )
    for token in proprio_tokens:
        if token not in additional_special_tokens:
            additional_special_tokens.append(token)
    processor.tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
    vlm.resize_token_embeddings(len(processor.tokenizer))

    value_head_checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, "value_head")
    value_head.load_state_dict(torch.load(value_head_checkpoint_path, map_location="cpu", weights_only=True))
    proprio_projector_checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, "proprio_projector")
    proprio_projector.load_state_dict(
        torch.load(proprio_projector_checkpoint_path, map_location="cpu", weights_only=True)
    )

    vlm.eval().to(device)
    value_head.eval().to(device)
    proprio_projector.eval().to(device)

    return processor, vlm, proprio_projector, value_head


@torch.no_grad()
def predict_batch_values(
    vlm,
    proprio_projector,
    value_head,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    proprio_token_id: int,
) -> np.ndarray:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    pixel_values = batch["pixel_values"].to(device)
    proprio = batch["proprio"].to(device)
    pixel_attention_mask = batch.get("pixel_attention_mask")
    if pixel_attention_mask is not None:
        pixel_attention_mask = pixel_attention_mask.to(device)

    mask = input_ids.eq(proprio_token_id)
    if not torch.all(mask.sum(dim=1) == 1):
        raise ValueError("Each sample should contain exactly one `<proprio_0>` token.")

    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        proprio_embeddings = proprio_projector(proprio)

        def inject_proprio_embedding(_module, _inputs, token_embeddings):
            return torch.where(
                mask.unsqueeze(-1),
                proprio_embeddings.unsqueeze(1).to(dtype=token_embeddings.dtype),
                token_embeddings,
            )

        embedding_layer = vlm.get_input_embeddings()
        hook_handle = embedding_layer.register_forward_hook(inject_proprio_embedding)
        try:
            outputs = vlm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                pixel_attention_mask=pixel_attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        finally:
            hook_handle.remove()

        hidden = outputs.hidden_states[-1]
        last_token_idx = attention_mask.sum(dim=1) - 1
        pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_token_idx]
        value_logits = value_head(pooled).squeeze(-1)

    pred_value = torch.sigmoid(value_logits.float()) - 1.0
    return pred_value.detach().cpu().numpy().astype(np.float32)


def predict_episode_values(
    episode: List[Dict[str, Any]],
    collator: PaddedCollatorForValueFunction,
    vlm,
    proprio_projector,
    value_head,
    device: torch.device,
    proprio_token_id: int,
    batch_size: int,
) -> np.ndarray:
    values = []
    for start in range(0, len(episode), batch_size):
        batch = collator(episode[start : start + batch_size])
        values.append(
            predict_batch_values(
                vlm=vlm,
                proprio_projector=proprio_projector,
                value_head=value_head,
                batch=batch,
                device=device,
                proprio_token_id=proprio_token_id,
            )
        )
    return np.concatenate(values, axis=0)


def compute_advantage(
    rewards: np.ndarray,
    return_to_go: np.ndarray,
    values: np.ndarray,
    cfg: VfGenerateConfig,
) -> np.ndarray:
    if cfg.advantage_method == "mc":
        return (return_to_go - values).astype(np.float32)

    if cfg.advantage_method != "n_step":
        raise ValueError("`advantage_method` must be one of: `mc`, `n_step`.")
    if cfg.n_step <= 0:
        raise ValueError("`n_step` must be positive when using `advantage_method=n_step`.")

    advantages = np.zeros_like(values, dtype=np.float32)
    for timestep in range(len(values)):
        horizon = min(cfg.n_step, len(values) - timestep)
        scaled_return = 0.0
        discount = 1.0
        for offset in range(horizon):
            scaled_return += discount * rewards[timestep + offset] / cfg.return_scale
            discount *= cfg.gamma

        bootstrap_timestep = timestep + horizon
        if bootstrap_timestep < len(values):
            scaled_return += discount * values[bootstrap_timestep]

        advantages[timestep] = scaled_return - values[timestep]
    return advantages


def episode_output_path(output_dir: Path, episode_id: int) -> Path:
    return output_dir / f"episode_{episode_id:06d}.npz"


def safe_filename_component(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)
    return safe.strip("._") or "dataset"


def write_episode_advantage(
    path: Path,
    *,
    dataset_name: str,
    split: str,
    episode_id: int,
    timesteps: np.ndarray,
    rewards: np.ndarray,
    return_to_go: np.ndarray,
    values: np.ndarray,
    advantages: np.ndarray,
    cfg: VfGenerateConfig,
) -> None:
    np.savez_compressed(
        path,
        dataset_name=np.asarray(dataset_name),
        split=np.asarray(split),
        episode_id=np.asarray(episode_id, dtype=np.int64),
        timestep=timesteps.astype(np.int64),
        reward=rewards.astype(np.float32),
        return_to_go=return_to_go.astype(np.float32),
        value=values.astype(np.float32),
        advantage=advantages.astype(np.float32),
        advantage_method=np.asarray(cfg.advantage_method),
        pretrained_checkpoint=np.asarray(str(cfg.pretrained_checkpoint)),
    )


def build_manifest_record(
    *,
    dataset_name: str,
    split: str,
    episode_id: int,
    path: Path,
    num_steps: int,
) -> Dict[str, Any]:
    return {
        "dataset_name": dataset_name,
        "split": split,
        "episode_id": episode_id,
        "path": str(path),
        "num_steps": num_steps,
    }


def table_output_path(output_dir: Path, dataset_name: str) -> Path:
    return output_dir / f"advantage_table_{safe_filename_component(dataset_name)}.npz"


def _load_optional_step_array(
    data: np.lib.npyio.NpzFile,
    key: str,
    dtype: np.dtype,
    num_steps: int,
) -> Optional[np.ndarray]:
    if key not in data.files:
        return None
    array = np.asarray(data[key], dtype=dtype)
    if array.shape[0] != num_steps:
        raise ValueError(f"`{key}` has {array.shape[0]} rows, expected {num_steps}.")
    return array


def write_advantage_tables_by_dataset(
    output_dir: Path,
    manifest_records: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    records_by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    for record in manifest_records:
        records_by_dataset.setdefault(record["dataset_name"], []).append(record)

    table_index: Dict[str, Dict[str, Any]] = {}

    for dataset_name, records in sorted(records_by_dataset.items()):
        episode_ids: List[np.ndarray] = []
        timesteps: List[np.ndarray] = []
        advantages: List[np.ndarray] = []
        per_episode_ids: List[int] = []
        episode_starts: List[int] = []
        episode_lengths: List[int] = []
        num_samples = 0

        for record in sorted(records, key=lambda item: item["episode_id"]):
            episode_path = Path(record["path"])
            with np.load(episode_path, allow_pickle=False) as data:
                advantage = np.asarray(data["advantage"], dtype=np.float32)
                num_steps = int(advantage.shape[0])
                episode_id = int(np.asarray(data["episode_id"]).reshape(-1)[0])
                timestep = _load_optional_step_array(data, "timestep", np.dtype(np.int64), num_steps)
                if timestep is None:
                    timestep = np.arange(num_steps, dtype=np.int64)

                episode_ids.append(np.full(num_steps, episode_id, dtype=np.int64))
                timesteps.append(timestep)
                advantages.append(advantage)
                per_episode_ids.append(episode_id)
                episode_starts.append(num_samples)
                episode_lengths.append(num_steps)

                num_samples += num_steps

        if not advantages:
            continue

        payload: Dict[str, Any] = {
            "dataset_name": np.asarray(dataset_name),
            "split": np.asarray(records[0]["split"]),
            "episode_id": np.concatenate(episode_ids, axis=0),
            "timestep": np.concatenate(timesteps, axis=0),
            "advantage": np.concatenate(advantages, axis=0),
            "episode_index_episode_id": np.asarray(per_episode_ids, dtype=np.int64),
            "episode_start_index": np.asarray(episode_starts, dtype=np.int64),
            "episode_length": np.asarray(episode_lengths, dtype=np.int64),
        }

        table_path = table_output_path(output_dir, dataset_name)
        np.savez_compressed(table_path, **payload)
        table_index[dataset_name] = {
            "path": str(table_path),
            "split": records[0]["split"],
            "num_episodes": len(records),
            "num_samples": int(num_samples),
        }

    return table_index


@draccus.wrap()
def eval_vf(cfg: VfGenerateConfig):
    if cfg.split not in {"train", "val"}:
        raise ValueError("`split` must be `train` or `val`.")
    if cfg.batch_size <= 0:
        raise ValueError("`batch_size` must be positive.")

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    processor, vlm, proprio_projector, value_head = initialize_model(
        cfg,
        device,
        verbose=True,
    )
    dataset_statistics = load_dataset_statistics_for_labels(
        cfg,
        verbose=True,
    )
    proprio_token_id = processor.tokenizer.convert_tokens_to_ids("<proprio_0>")

    processor_image_size = getattr(getattr(processor, "image_processor", None), "size", {})
    if isinstance(processor_image_size, dict):
        effective_image_size = int(processor_image_size.get("longest_edge", cfg.image_size))
    else:
        effective_image_size = int(getattr(processor_image_size, "longest_edge", cfg.image_size))

    batch_transform = RLDSValueBatchTransform()
    dataset = EpisodicRLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=(effective_image_size, effective_image_size),
        train=cfg.split == "train",
        image_aug=cfg.image_aug,
        future_action_window_size=0,
        include_value_targets=True,
        include_metadata=True,
        dataset_statistics=dataset_statistics,
        goal_relabeling_strategy=None,
        use_shared_proprio_bounds=cfg.use_shared_proprio_bounds,
    )
    collator = PaddedCollatorForValueFunction(
        processor=processor,
        max_length=cfg.max_length,
        num_proprio_tokens=1,
    )

    output_dir = cfg.advantage_data_dir / cfg.dataset_name / cfg.split
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_records: List[Dict[str, Any]] = []

    progress = tqdm.tqdm(
        dataset,
        total=len(dataset),
        desc="Advantage labels",
    )

    advantages_by_semantic = {}

    for episode_index, episode in enumerate(progress):
        if cfg.max_episodes is not None and episode_index >= cfg.max_episodes:
            break
        if not episode:
            continue
        if "episode_id" not in episode[0] or "timestep" not in episode[0]:
            raise ValueError("Episode examples must contain `episode_id` and `timestep` metadata.")

        episode_id = int(episode[0]["episode_id"])
        dataset_name = decode_scalar_string(episode[0]["dataset_name"])
        output_path = episode_output_path(output_dir, episode_id)

        episode_ids = np.asarray([int(step["episode_id"]) for step in episode], dtype=np.int64)
        if not np.all(episode_ids == episode_id):
            raise ValueError(f"Episode yielded mixed episode ids: {np.unique(episode_ids)}")

        timesteps = np.asarray([int(step["timestep"]) for step in episode], dtype=np.int64)
        rewards = np.asarray([float(torch.as_tensor(step["reward"]).item()) for step in episode], dtype=np.float32)
        return_to_go = np.asarray(
            [float(torch.as_tensor(step["return_to_go"]).item()) for step in episode],
            dtype=np.float32,
        )

        values = predict_episode_values(
            episode=episode,
            collator=collator,
            vlm=vlm,
            proprio_projector=proprio_projector,
            value_head=value_head,
            device=device,
            proprio_token_id=proprio_token_id,
            batch_size=cfg.batch_size,
        )
        advantages = compute_advantage(rewards, return_to_go, values, cfg)
        
        language = normalize_language(episode[0]['language_instruction'])
        base_task = NORMALIZED_LANGUAGE_TO_BASE_TASKS[language][0] if language in NORMALIZED_LANGUAGE_TO_BASE_TASKS.keys() else language
        base_task = base_task.replace(' ', '_')
        advantages_by_semantic.setdefault(base_task, []).append(advantages)

        write_episode_advantage(
            output_path,
            dataset_name=dataset_name,
            split=cfg.split,
            episode_id=episode_id,
            timesteps=timesteps,
            rewards=rewards,
            return_to_go=return_to_go,
            values=values,
            advantages=advantages,
            cfg=cfg,
        )
        manifest_records.append(
            build_manifest_record(
                dataset_name=dataset_name,
                split=cfg.split,
                episode_id=episode_id,
                path=output_path,
                num_steps=len(episode),
            )
        )

    manifest_records.sort(key=lambda record: (record["dataset_name"], record["episode_id"]))

    semantic_path = output_dir / "advantages_by_semantic.npy"
    np.save(semantic_path, advantages_by_semantic, allow_pickle=True)

    table_index = write_advantage_tables_by_dataset(output_dir, manifest_records)
    total_samples = sum(table["num_samples"] for table in table_index.values())

    print(
        f"Saved advantage tables to {output_dir} "
        f"({len(table_index)} tables, {len(manifest_records)} episodes, {total_samples} samples)."
    )
    print(f"Saved semantic advantages to {semantic_path}.")


if __name__ == "__main__":
    eval_vf()
