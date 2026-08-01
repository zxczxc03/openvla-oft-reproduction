from pathlib import Path
import numpy as np

advantage_data_dir: Path = Path("advantage_data")
datasets_list: list[str] = [
    "libero_plus_spatial_autonomous_failure",
    "libero_plus_spatial_autonomous_success",
]
split: str = "train"
advantages_by_semantic = {}
quantile_by_semantic = {}
positive_fraction = 0.4

for dataset in datasets_list:
    dataset_advantage_path = advantage_data_dir / dataset / split
    per_dataset = np.load(
        dataset_advantage_path / "advantages_by_semantic.npy",
        allow_pickle=True,
    ).item()

    for key, episode_advantages in per_dataset.items():
        if not episode_advantages:
            continue
        flat = np.concatenate(episode_advantages, axis=0)
        advantages_by_semantic.setdefault(key, []).extend(flat.tolist())

for key, values in advantages_by_semantic.items():
    quantile_by_semantic[key] = float(np.quantile(values, 1 - positive_fraction))

np.save(advantage_data_dir / f"quantile_by_semantic_{split}.npy", quantile_by_semantic, allow_pickle=True)
    
