import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import draccus
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image, ImageDraw, ImageFont

from libero_language_to_task import NORMALIZED_LANGUAGE_TO_BASE_TASKS, normalize_language
from prismatic.vla.datasets import EpisodicRLDSDataset, RLDSValueBatchTransform

"""
example:
python vla-scripts/visualize_advantage.py \
  --data_root_dir /root/autodl-tmp/openvla-oft/LIBERO_datasets \
  --dataset_name libero_spatial \
  --split train \
  --episode_id 850
"""



# regenerate_recovery_dataset_for_openvla.py creates OffScreenRenderEnv without
# overriding control_freq. LIBERO's ControlEnv / BDDLBaseDomain default to 20 Hz.
DEFAULT_LIBERO_CONTROL_FPS = 20


@dataclasses.dataclass
class VisualizeAdvantageConfig:
    data_root_dir: Path = Path("datasets/rlds")
    dataset_name: str = "libero_spatial"
    split: str = "train"
    episode_id: int = 850

    advantage_data_dir: Path = Path("advantage_data")
    advantage_path: Optional[Path] = None
    output_dir: Path = Path("rollout_video")

    image_size: int = 256
    fps: int = DEFAULT_LIBERO_CONTROL_FPS
    return_scale: float = 250.0
    use_advantage_metadata: bool = True
    match_episode_by: str = "episode_id"  # one of: episode_id, index
    strict_length: bool = False

    write_camera_videos: bool = True
    write_combined_video: bool = True
    write_annotated_video: bool = True
    write_plot: bool = True


def decode_scalar(value: Any) -> Any:
    scalar = np.asarray(value).reshape(-1)[0]
    if isinstance(scalar, bytes):
        return scalar.decode("utf-8")
    if isinstance(scalar, np.bytes_):
        return scalar.tobytes().decode("utf-8")
    return scalar.item() if hasattr(scalar, "item") else scalar


def format_float(value: float) -> str:
    return f"{float(value):+.3f}"


def decode_text(value: Any) -> str:
    scalar = np.asarray(value).reshape(-1)[0]
    if isinstance(scalar, bytes):
        return scalar.decode("utf-8")
    if isinstance(scalar, np.bytes_):
        return scalar.tobytes().decode("utf-8")
    return str(scalar)


def semantic_task_from_language(language_instruction: str) -> Tuple[str, str]:
    normalized = normalize_language(language_instruction)
    base_tasks = NORMALIZED_LANGUAGE_TO_BASE_TASKS.get(normalized)
    semantic_task = base_tasks[0] if base_tasks else normalized
    return normalized, semantic_task


def episode_filename(episode_id: int) -> str:
    return f"episode_{episode_id:06d}.npz"


def resolve_advantage_path(cfg: VisualizeAdvantageConfig) -> Path:
    if cfg.advantage_path is not None:
        path = Path(cfg.advantage_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Advantage file does not exist: {path}")
        return path

    candidate = (
        Path(cfg.advantage_data_dir).expanduser()
        / cfg.dataset_name
        / cfg.split
        / episode_filename(cfg.episode_id)
    )
    if candidate.exists():
        return candidate

    local_candidate = Path(episode_filename(cfg.episode_id))
    if local_candidate.exists():
        return local_candidate

    raise FileNotFoundError(
        "Could not find advantage data. Tried:\n"
        f"  {candidate}\n"
        f"  {local_candidate}\n"
        "Pass --advantage_path to visualize a specific .npz file."
    )


def load_advantage_data(path: Path) -> Dict[str, Any]:
    required_keys = ("timestep", "advantage", "value", "return_to_go")
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in required_keys if key not in data.files]
        if missing:
            raise KeyError(f"{path} is missing required field(s): {missing}")

        result: Dict[str, Any] = {
            "timestep": np.asarray(data["timestep"], dtype=np.int64),
            "advantage": np.asarray(data["advantage"], dtype=np.float32),
            "value": np.asarray(data["value"], dtype=np.float32),
            "return_to_go": np.asarray(data["return_to_go"], dtype=np.float32),
        }
        if "reward" in data.files:
            result["reward"] = np.asarray(data["reward"], dtype=np.float32)

        for key in ("dataset_name", "split", "episode_id", "advantage_method", "pretrained_checkpoint"):
            if key in data.files:
                result[key] = decode_scalar(data[key])

    lengths = {key: len(result[key]) for key in required_keys}
    if "reward" in result:
        lengths["reward"] = len(result["reward"])
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Advantage arrays must have the same length, got {lengths}")

    return result


def resolve_dataset_identity(
    cfg: VisualizeAdvantageConfig,
    advantage_data: Dict[str, Any],
) -> Tuple[str, str, int]:
    dataset_name = cfg.dataset_name
    split = cfg.split
    episode_id = cfg.episode_id

    if cfg.use_advantage_metadata:
        dataset_name = str(advantage_data.get("dataset_name", dataset_name))
        split = str(advantage_data.get("split", split))
        episode_id = int(advantage_data.get("episode_id", episode_id))

    return dataset_name, split, episode_id


def build_dataset(cfg: VisualizeAdvantageConfig, dataset_name: str, split: str) -> EpisodicRLDSDataset:
    batch_transform = RLDSValueBatchTransform()
    return EpisodicRLDSDataset(
        Path(cfg.data_root_dir).expanduser(),
        dataset_name,
        batch_transform,
        resize_resolution=(cfg.image_size, cfg.image_size),
        train=split == "train",
        image_aug=False,
        future_action_window_size=0,
        include_value_targets=True,
        include_metadata=True,
        goal_relabeling_strategy=None,
    )


def get_episode(
    dataset: EpisodicRLDSDataset,
    episode_id: int,
    match_episode_by: str,
) -> Tuple[int, List[Dict[str, Any]]]:
    if match_episode_by not in {"episode_id", "index"}:
        raise ValueError("match_episode_by must be one of: episode_id, index")

    for episode_index, episode in enumerate(dataset):
        if not episode:
            continue
        if match_episode_by == "index":
            if episode_index == episode_id:
                return episode_index, episode
            continue

        step_episode_id = episode[0].get("episode_id")
        if step_episode_id is not None and int(step_episode_id) == episode_id:
            return episode_index, episode

    raise ValueError(f"Could not find episode {episode_id} by {match_episode_by}.")


def to_pil_image(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)

    if image.ndim == 3 and image.shape[0] in {1, 3} and image.shape[-1] not in {1, 3}:
        image = np.moveaxis(image, 0, -1)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0) * 255.0
    image = image.astype(np.uint8)
    if image.ndim == 2:
        return Image.fromarray(image, mode="L").convert("RGB")
    return Image.fromarray(image).convert("RGB")


def ensure_mp4_path(output_path: Path) -> Path:
    output_path = Path(output_path)
    return output_path if output_path.suffix.lower() == ".mp4" else output_path.with_suffix(".mp4")


def image2Video(images: List[Any], output_path: Path, fps: int) -> Path:
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if len(images) == 0:
        raise ValueError("the number of images is 0")

    output_path = ensure_mp4_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    first = to_pil_image(images[0])
    width, height = first.size
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    for image in images:
        frame = to_pil_image(image)
        if frame.size != (width, height):
            frame = frame.resize((width, height), Image.BILINEAR)
        writer.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"Saved video to {output_path}")
    return output_path


def image2Vedio(images: List[Any], output_path: Path, fps: int) -> Path:
    return image2Video(images, output_path, fps)


def draw_label(image: Image.Image, text: str) -> Image.Image:
    frame = image.copy()
    draw = ImageDraw.Draw(frame)
    font = ImageFont.load_default()
    padding = 5
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    draw.rectangle((0, 0, text_w + padding * 2, text_h + padding * 2), fill=(0, 0, 0))
    draw.text((padding, padding), text, fill=(255, 255, 255), font=font)
    return frame


def make_camera_grid(primary: Image.Image, wrist: Optional[Image.Image]) -> Image.Image:
    primary = to_pil_image(primary)
    if wrist is None:
        return draw_label(primary, "primary")

    wrist = to_pil_image(wrist)
    if wrist.height != primary.height:
        wrist_width = int(round(wrist.width * (primary.height / wrist.height)))
        wrist = wrist.resize((wrist_width, primary.height), Image.BILINEAR)

    canvas = Image.new("RGB", (primary.width + wrist.width, primary.height), (0, 0, 0))
    canvas.paste(draw_label(primary, "primary"), (0, 0))
    canvas.paste(draw_label(wrist, "wrist"), (primary.width, 0))
    return canvas


def save_advantage_plot(
    advantage_data: Dict[str, Any],
    output_path: Path,
    title: Optional[str] = None,
    return_scale: float = 250.0,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timesteps = advantage_data["timestep"]

    plt.figure(figsize=(10, 4.8), dpi=140)
    plt.plot(timesteps, advantage_data["advantage"], label="advantage", linewidth=1.8)
    plt.plot(timesteps, advantage_data["value"], label="value", linewidth=1.4)
    plt.plot(timesteps, advantage_data["return_to_go"], label="return_to_go", linewidth=1.4)
    if "reward" in advantage_data:
        if return_scale <= 0:
            raise ValueError(f"return_scale must be positive, got {return_scale}")
        scaled_reward = advantage_data["reward"] / return_scale
        plt.plot(
            timesteps,
            scaled_reward,
            label=f"reward / {return_scale:g}",
            linewidth=1.0,
            alpha=0.55,
        )
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    plt.xlabel("timestep")
    plt.ylabel("score")
    if title:
        plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to {output_path}")
    return output_path


def render_metric_panel(
    advantage_data: Dict[str, Any],
    frame_index: int,
    size: Tuple[int, int],
) -> Image.Image:
    width, height = size
    dpi = 100
    fig = plt.Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    timesteps = advantage_data["timestep"]
    current_timestep = timesteps[frame_index]
    ax.plot(timesteps, advantage_data["advantage"], label="adv", linewidth=1.5)
    ax.plot(timesteps, advantage_data["value"], label="value", linewidth=1.1)
    ax.plot(timesteps, advantage_data["return_to_go"], label="rtg", linewidth=1.1)
    ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.3)
    ax.axvline(current_timestep, color="red", linewidth=1.1, alpha=0.8)
    ax.set_xlim(timesteps[0], timesteps[-1])
    ax.tick_params(labelsize=7)
    ax.set_xlabel("timestep", fontsize=8)
    ax.set_ylabel("score", fontsize=8)
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout(pad=0.35)

    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return Image.fromarray(rgba[:, :, :3]).convert("RGB")


def make_annotated_frames(
    primary_images: List[Image.Image],
    wrist_images: List[Image.Image],
    advantage_data: Dict[str, Any],
    semantic_task: Optional[str] = None,
) -> List[Image.Image]:
    frames: List[Image.Image] = []
    has_wrist = len(wrist_images) == len(primary_images)

    for frame_index, primary in enumerate(primary_images):
        wrist = wrist_images[frame_index] if has_wrist else None
        camera_grid = make_camera_grid(primary, wrist)
        plot_panel = render_metric_panel(advantage_data, frame_index, (camera_grid.width, camera_grid.height))

        header_h = 44 if semantic_task else 28
        canvas = Image.new(
            "RGB",
            (camera_grid.width, header_h + camera_grid.height + plot_panel.height),
            (255, 255, 255),
        )
        timestep = int(advantage_data["timestep"][frame_index])
        header = (
            f"t={timestep}  "
            f"A={format_float(advantage_data['advantage'][frame_index])}  "
            f"V={format_float(advantage_data['value'][frame_index])}  "
            f"RTG={format_float(advantage_data['return_to_go'][frame_index])}"
        )
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), header, fill=(0, 0, 0), font=ImageFont.load_default())
        if semantic_task:
            draw.text((8, 24), f"task={semantic_task}", fill=(0, 0, 0), font=ImageFont.load_default())
        canvas.paste(camera_grid, (0, header_h))
        canvas.paste(plot_panel, (0, header_h + camera_grid.height))
        frames.append(canvas)

    return frames


def truncate_to_common_length(
    primary_images: List[Image.Image],
    wrist_images: List[Image.Image],
    advantage_data: Dict[str, Any],
    strict: bool,
) -> Tuple[List[Image.Image], List[Image.Image], Dict[str, Any]]:
    lengths = {
        "primary_images": len(primary_images),
        "advantage": len(advantage_data["advantage"]),
    }
    if wrist_images:
        lengths["wrist_images"] = len(wrist_images)

    common_length = min(lengths.values())
    if len(set(lengths.values())) == 1:
        return primary_images, wrist_images, advantage_data

    message = f"Episode/image/advantage lengths differ: {lengths}."
    if strict:
        raise ValueError(message)
    print(f"Warning: {message} Truncating all streams to {common_length}.")

    truncated = dict(advantage_data)
    for key in ("timestep", "advantage", "value", "return_to_go", "reward"):
        if key in truncated:
            truncated[key] = truncated[key][:common_length]
    return primary_images[:common_length], wrist_images[:common_length], truncated


@draccus.wrap()
def main(cfg: VisualizeAdvantageConfig) -> None:
    advantage_path = resolve_advantage_path(cfg)
    advantage_data = load_advantage_data(advantage_path)
    dataset_name, split, episode_id = resolve_dataset_identity(cfg, advantage_data)

    print(f"Loading advantage data: {advantage_path}")
    print(f"Dataset: {dataset_name} | split: {split} | episode_id: {episode_id} | fps: {cfg.fps}")

    dataset = build_dataset(cfg, dataset_name, split)
    matched_index, episode = get_episode(dataset, episode_id, cfg.match_episode_by)
    print(f"Matched dataset episode index: {matched_index} ({len(episode)} steps)")

    language_instruction = decode_text(episode[0]["language_instruction"])
    normalized_language, semantic_task = semantic_task_from_language(language_instruction)
    print(f"Language: {language_instruction}")
    print(f"Normalized language: {normalized_language}")
    print(f"Semantic task: {semantic_task}")

    primary_images = [to_pil_image(step["image_primary"]) for step in episode]
    wrist_images = [to_pil_image(step["image_wrist"]) for step in episode if "image_wrist" in step]
    primary_images, wrist_images, advantage_data = truncate_to_common_length(
        primary_images,
        wrist_images,
        advantage_data,
        strict=cfg.strict_length,
    )

    episode_dir = Path(cfg.output_dir) / dataset_name / split / f"episode_{episode_id:06d}"
    episode_dir.mkdir(parents=True, exist_ok=True)

    if cfg.write_plot:
        save_advantage_plot(
            advantage_data,
            episode_dir / "advantage.png",
            title=semantic_task,
            return_scale=cfg.return_scale,
        )

    if cfg.write_camera_videos:
        image2Video(primary_images, episode_dir / "primary.mp4", cfg.fps)
        if wrist_images:
            image2Video(wrist_images, episode_dir / "wrist.mp4", cfg.fps)

    if cfg.write_combined_video:
        combined_frames = [
            make_camera_grid(primary, wrist_images[index] if wrist_images else None)
            for index, primary in enumerate(primary_images)
        ]
        image2Video(combined_frames, episode_dir / "combined_cameras.mp4", cfg.fps)

    if cfg.write_annotated_video:
        annotated_frames = make_annotated_frames(primary_images, wrist_images, advantage_data, semantic_task)
        image2Video(annotated_frames, episode_dir / "advantage_overlay.mp4", cfg.fps)

    print(f"Saved visualization outputs under {episode_dir}")


if __name__ == "__main__":
    main()
