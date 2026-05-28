'''
ft_value_function.py

Fine-tunes SmolVLM to predict value via LoRA

'''
import os
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import draccus
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.parallel import DistributedDataParallel as DDP
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoProcessor, Idefics3ForConditionalGeneration

import wandb

from prismatic.models.action_heads import MLPResNet
from prismatic.models.projectors import ProprioProjector
from prismatic.util.data_utils import PaddedCollatorForValueFunction
from prismatic.vla.constants import PROPRIO_DIM
from prismatic.vla.datasets import RLDSValueBatchTransform, RLDSValueDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics


os.environ["TOKENIZERS_PARALLELISM"] = "false"



@dataclass
class FinetuneConfig:
    # fmt: off
    vlm_path: str = "HuggingFaceTB/SmolVLM-500M-Instruct"             # Path to SmolVLM model (on HuggingFace Hub or stored locally)

    # Dataset
    data_root_dir: Path = Path("datasets/rlds")      # Directory containing RLDS datasets
    dataset_name: str = "libero_spatial_no_noops"    # Dataset whose transform creates value targets
    run_root_dir: Path = Path("runs")                # Path to directory to store logs & checkpoints
    shuffle_buffer_size: int = 100_000               # Dataloader shuffle buffer size (can reduce if OOM errors occur)

    # Training configuration
    batch_size: int = 8                              # Batch size per device (total batch size = batch_size * num GPUs)
    learning_rate: float = 5e-4                      # Learning rate
    lr_warmup_steps: int = 0                         # Number of steps to warm up learning rate (from 10% to 100%)
    num_steps_before_decay: int = 100_000            # Number of steps before LR decays by 10x
    grad_accumulation_steps: int = 1                 # Number of gradient accumulation steps
    max_steps: int = 200_000                         # Max number of training steps
    use_val_set: bool = False                        # If True, uses validation set and log validation metrics
    val_freq: int = 10_000                           # (When `use_val_set==True`) Validation set logging frequency in steps
    val_time_limit: int = 180                        # (When `use_val_set==True`) Time limit for computing validation metrics
    save_freq: int = 10_000                          # Checkpoint saving frequency in steps
    save_latest_checkpoint_only: bool = False        # If True, saves only 1 checkpoint, overwriting latest checkpoint
                                                     #   (If False, saves all checkpoints)
    resume: bool = False                             # If True, resumes from checkpoint
    resume_step: Optional[int] = None                # (When `resume==True`) Step number that we are resuming from
    image_aug: bool = True                           # If True, trains with image augmentations (HIGHLY RECOMMENDED)

    # LoRA
    use_lora: bool = True                            # If True, uses LoRA fine-tuning
    lora_rank: int = 32                              # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                        # Dropout applied to LoRA weights
    merge_lora_during_training: bool = True          # If True, merges LoRA weights and saves result during training
                                                     #   Note: Merging can be very slow on some machines. If so, set to
                                                     #         False and merge final checkpoint offline!

    # Logging
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    run_id_override: Optional[str] = None            # Optional string to override the run ID with
    wandb_log_freq: int = 10                         # WandB logging frequency in steps

def get_run_id(cfg) -> str:
    """
    Generates or retrieves an identifier string for an experiment run.

    Args:
        cfg (FinetuneConfig): Training configuration.

    Returns:
        str: Experiment run ID.
    """
    if cfg.run_id_override is not None:
        # Override the run ID with the user-provided ID
        run_id = cfg.run_id_override
    elif cfg.resume:
        # Override run ID with the previous resumed run's ID
        run_id = cfg.vlm_path.split("/")[-1]
        # Remove the "--XXX_chkpt" suffix from the run ID if it exists
        if "chkpt" in run_id.split("--")[-1]:
            run_id = "--".join(run_id.split("--")[:-1])
    else:
        run_id = (
            f"{cfg.vlm_path.split('/')[-1]}+{cfg.dataset_name}"
            f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
            f"+lr-{cfg.learning_rate}"
        )
        if cfg.use_lora:
            run_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
        if cfg.image_aug:
            run_id += "--image_aug"
        if cfg.run_id_note is not None:
            run_id += f"--{cfg.run_id_note}"
    return run_id

def wrap_ddp(module: nn.Module, device_id: int, find_unused: bool = False) -> DDP:
    """
    Wrap a module with DistributedDataParallel.

    Args:
        module (nn.Module): PyTorch module.
        device_id (str): Device ID.
        find_unused (bool): Whether to detect parameters without gradients in distributed training.

    Returns:
        DistributedDataParallel: PyTorch module wrapped with DDP.
    """
    return DDP(module, device_ids=[device_id], find_unused_parameters=find_unused, gradient_as_bucket_view=True)


def unwrap_module(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, DDP) else module


def count_parameters(module: nn.Module, name: str) -> None:
    num_params = sum(param.numel() for param in module.parameters() if param.requires_grad)
    print(f"# trainable params in {name}: {num_params}")


def compute_smoothened_metrics(metrics_deques) -> dict:
    return {name: sum(values) / len(values) for name, values in metrics_deques.items() if values}


def log_metrics_to_wandb(metrics: dict, prefix: str, step: int) -> None:
    display_names = {
        "loss_value": "Loss",
        "mae": "MAE",
        "rmse": "RMSE",
        # "pred_value_mean": "Pred Value Mean",
        # "pred_value_min": "Pred Value Min",
        # "pred_value_max": "Pred Value Max",
        # "target_value_mean": "Target Value Mean",
        # "target_value_min": "Target Value Min",
        # "target_value_max": "Target Value Max",
        # "gradient_norm": "Gradient Norm",
        # "learning_rate": "Learning Rate",
        # "batches_count": "Batches Count",
    }
    wandb.log(
        {f"{prefix}/{display_names.get(name, name.replace('_', ' ').title())}": value for name, value in metrics.items()},
        step=step,
    )


def load_training_state(checkpoint_dir: Path) -> dict:
    state_path = checkpoint_dir / "training_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"Could not resume value training: missing `{state_path}`.")
    return torch.load(state_path, map_location="cpu", weights_only=False)


def load_module_checkpoint(module: nn.Module, module_name: str, checkpoint_dir: Path, suffix: str) -> None:
    checkpoint_path = checkpoint_dir / f"{module_name}--{suffix}"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Could not resume value training: missing `{checkpoint_path}`.")
    module.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))


def run_forward_pass(
    vlm,
    value_head,
    proprio_projector,
    batch,
    device_id,
    proprio_token_id,
):
    ground_truth_values = batch["value_label"].to(device_id).float().squeeze(-1)
    if torch.any((ground_truth_values < -1.0) | (ground_truth_values > 0.0)):
        raise ValueError(
            "Value targets must be in [-1, 0] for the current sigmoid value head. "
            "Normalize the targets or change the output parameterization."
        )
    input_ids = batch["input_ids"].to(device_id)
    attention_mask = batch["attention_mask"].to(device_id)
    pixel_values = batch["pixel_values"].to(device_id)
    proprio = batch["proprio"].to(device_id)
    pixel_attention_mask = batch.get("pixel_attention_mask")
    if pixel_attention_mask is not None:
        pixel_attention_mask = pixel_attention_mask.to(device_id)

    mask = input_ids.eq(proprio_token_id)
    if not torch.all(mask.sum(dim=1) == 1):
        raise ValueError("Each sample should contain exactly one `<proprio_0>` token.")

    with torch.autocast("cuda", dtype=torch.bfloat16):
        proprio_embeddings = proprio_projector(proprio)

        # Keep `input_ids` in the Idefics3 forward path so image token replacement
        # still works, while overriding just the learned proprio token embedding.
        def inject_proprio_embedding(_module, _inputs, token_embeddings):
            return torch.where(
                mask.unsqueeze(-1),
                proprio_embeddings.unsqueeze(1).to(dtype=token_embeddings.dtype),
                token_embeddings,
            )

        embedding_layer = unwrap_module(vlm).get_input_embeddings()
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
        pooled = hidden[
            torch.arange(hidden.shape[0], device=hidden.device),
            last_token_idx,
        ]
        value_logits = value_head(pooled).squeeze(-1)

    pred_value = torch.sigmoid(value_logits.float()) - 1
    loss = F.mse_loss(pred_value, ground_truth_values)
    absolute_error = (pred_value - ground_truth_values).abs()
    metrics = {
        "loss_value": loss.detach().item(),
        "mae": absolute_error.mean().detach().item(),
        "rmse": torch.sqrt(F.mse_loss(pred_value, ground_truth_values)).detach().item(),
        # "pred_value_mean": pred_value.mean().detach().item(),
        # "pred_value_min": pred_value.min().detach().item(),
        # "pred_value_max": pred_value.max().detach().item(),
        # "target_value_mean": ground_truth_values.mean().detach().item(),
        # "target_value_min": ground_truth_values.min().detach().item(),
        # "target_value_max": ground_truth_values.max().detach().item(),
    }

    return loss, metrics


def save_training_checkpoint(
    cfg: FinetuneConfig,
    run_dir: Path,
    log_step: int,
    base_vlm_path: str,
    vlm: nn.Module,
    processor,
    value_head: nn.Module,
    proprio_projector: nn.Module,
    optimizer: AdamW,
    scheduler: MultiStepLR,
    train_dataset: RLDSValueDataset,
    distributed_state: PartialState,
) -> None:
    if cfg.save_latest_checkpoint_only:
        checkpoint_dir = run_dir
        checkpoint_suffix = "latest_checkpoint.pt"
    else:
        checkpoint_dir = Path(str(run_dir) + f"--{log_step}_chkpt")
        checkpoint_suffix = f"{log_step}_checkpoint.pt"
    adapter_dir = checkpoint_dir / "lora_adapter"

    if distributed_state.is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        adapter_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving Value Model Checkpoint for Step {log_step} at: {checkpoint_dir}")

        processor.save_pretrained(checkpoint_dir)
        unwrap_module(vlm).save_pretrained(adapter_dir)
        torch.save(unwrap_module(value_head).state_dict(), checkpoint_dir / f"value_head--{checkpoint_suffix}")
        torch.save(
            unwrap_module(proprio_projector).state_dict(),
            checkpoint_dir / f"proprio_projector--{checkpoint_suffix}",
        )
        torch.save(
            {
                "step": log_step,
                "base_vlm_path": str(base_vlm_path),
                "checkpoint_suffix": checkpoint_suffix,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            checkpoint_dir / "training_state.pt",
        )
        save_dataset_statistics(train_dataset.dataset_statistics, checkpoint_dir)

    distributed_state.wait_for_everyone()

    if distributed_state.is_main_process and cfg.merge_lora_during_training:
        base_vlm = Idefics3ForConditionalGeneration.from_pretrained(base_vlm_path, torch_dtype=torch.bfloat16)
        base_vlm.resize_token_embeddings(len(processor.tokenizer))
        merged_vlm = PeftModel.from_pretrained(base_vlm, adapter_dir).merge_and_unload()
        merged_vlm.save_pretrained(checkpoint_dir)
        del base_vlm, merged_vlm
        print(f"Saved merged value VLM for Step {log_step} at: {checkpoint_dir}")

    distributed_state.wait_for_everyone()


def run_validation(
    vlm: nn.Module,
    value_head: nn.Module,
    proprio_projector: nn.Module,
    val_dataloader: DataLoader,
    device_id: int,
    proprio_token_id: int,
    log_step: int,
    distributed_state: PartialState,
    val_time_limit: int,
) -> None:
    val_start_time = time.time()
    vlm.eval()
    value_head.eval()
    proprio_projector.eval()
    metric_totals = {}
    batches_count = 0

    with torch.no_grad():
        for batch in val_dataloader:
            _, metrics = run_forward_pass(
                vlm=vlm,
                value_head=value_head,
                proprio_projector=proprio_projector,
                batch=batch,
                device_id=device_id,
                proprio_token_id=proprio_token_id,
            )
            for metric_name, value in metrics.items():
                metric_totals[metric_name] = metric_totals.get(metric_name, 0.0) + value
            batches_count += 1
            if time.time() - val_start_time > val_time_limit:
                break

    if distributed_state.is_main_process and batches_count > 0:
        avg_metrics = {name: value / batches_count for name, value in metric_totals.items()}
        avg_metrics["batches_count"] = batches_count
        log_metrics_to_wandb(avg_metrics, "Value Val", log_step)


@draccus.wrap()
def finetune_value_function(cfg: FinetuneConfig):
    assert cfg.use_lora, "Only LoRA fine-tuning is supported. Please set --use_lora=True!"
    if cfg.grad_accumulation_steps < 1:
        raise ValueError("`grad_accumulation_steps` must be at least 1.")
    if cfg.max_steps < 1 or cfg.save_freq < 1 or cfg.wandb_log_freq < 1:
        raise ValueError("`max_steps`, `save_freq`, and `wandb_log_freq` must be positive.")
    if cfg.use_val_set and cfg.val_freq < 1:
        raise ValueError("`val_freq` must be positive when validation is enabled.")

    cfg.vlm_path = cfg.vlm_path.rstrip("/")
    print(f"Fine-tuning SmolVLM Model `{cfg.vlm_path}` on `{cfg.dataset_name}`")

    run_id = get_run_id(cfg)
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)

    resume_dir = Path(cfg.vlm_path) if cfg.resume else None
    resume_state = load_training_state(resume_dir) if resume_dir is not None else None
    start_step = int(resume_state["step"]) if resume_state is not None else 0
    if cfg.resume_step is not None and cfg.resume_step != start_step:
        raise ValueError(
            f"`resume_step={cfg.resume_step}` does not match the checkpoint state step `{start_step}`."
        )
    base_vlm_path = resume_state["base_vlm_path"] if resume_state is not None else cfg.vlm_path
    if start_step >= cfg.max_steps:
        print(f"Checkpoint is already at step {start_step}, which meets `max_steps={cfg.max_steps}`.")
        return

    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    if distributed_state.is_main_process:
        wandb_config = {
            name: str(value) if isinstance(value, Path) else value for name, value in asdict(cfg).items()
        }
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{run_id}", config=wandb_config)

    processor_path = str(resume_dir) if resume_dir is not None else base_vlm_path
    processor = AutoProcessor.from_pretrained(processor_path)
    processor.tokenizer.padding_side = "right"
    vlm = Idefics3ForConditionalGeneration.from_pretrained(
        base_vlm_path,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2",
    )
    text_config = getattr(vlm.config, "text_config", vlm.config)
    llm_dim = text_config.hidden_size
    value_head = MLPResNet(num_blocks=2, input_dim=llm_dim, hidden_dim=llm_dim, output_dim=1)
    proprio_projector = ProprioProjector(llm_dim=llm_dim, proprio_dim=PROPRIO_DIM)
    count_parameters(value_head, "value_head")
    count_parameters(proprio_projector, "proprio_projector")

    proprio_tokens = ["<proprio_0>"]
    additional_special_tokens = list(processor.tokenizer.additional_special_tokens)
    for token in proprio_tokens:
        if token not in additional_special_tokens:
            additional_special_tokens.append(token)
    processor.tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
    vlm.resize_token_embeddings(len(processor.tokenizer))
    proprio_token_id = processor.tokenizer.convert_tokens_to_ids(proprio_tokens[0])

    if resume_dir is not None:
        adapter_dir = resume_dir / "lora_adapter"
        if not adapter_dir.exists():
            raise FileNotFoundError(f"Could not resume value training: missing `{adapter_dir}`.")
        vlm = PeftModel.from_pretrained(vlm, adapter_dir, is_trainable=True)
        load_module_checkpoint(value_head, "value_head", resume_dir, resume_state["checkpoint_suffix"])
        load_module_checkpoint(proprio_projector, "proprio_projector", resume_dir, resume_state["checkpoint_suffix"])
    else:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules=["down_proj", "o_proj", "k_proj", "q_proj", "gate_proj", "up_proj", "v_proj"],
            init_lora_weights="gaussian",
        )
        vlm = get_peft_model(vlm, lora_config)
    vlm.print_trainable_parameters()

    vlm = vlm.to(device_id)
    value_head = value_head.to(device_id)
    proprio_projector = proprio_projector.to(device_id)
    vlm = wrap_ddp(vlm, device_id, find_unused=True)
    value_head = wrap_ddp(value_head, device_id, find_unused=True)
    proprio_projector = wrap_ddp(proprio_projector, device_id, find_unused=True)

    trainable_params = [param for param in vlm.parameters() if param.requires_grad]
    trainable_params += [param for param in value_head.parameters() if param.requires_grad]
    trainable_params += [param for param in proprio_projector.parameters() if param.requires_grad]
    print(f"# total trainable params: {sum(p.numel() for p in trainable_params)}")
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)
    original_lr = optimizer.param_groups[0]["lr"]
    scheduler = MultiStepLR(
        optimizer,
        milestones=[cfg.num_steps_before_decay],
        gamma=0.1,
    )
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])

    batch_transform = RLDSValueBatchTransform()
    train_dataset = RLDSValueDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=(256, 256),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )
    if cfg.use_val_set:
        val_dataset = RLDSValueDataset(
            cfg.data_root_dir,
            cfg.dataset_name,
            batch_transform,
            resize_resolution=(256, 256),
            shuffle_buffer_size=max(cfg.shuffle_buffer_size // 10, 1),
            train=False,
            image_aug=False,
        )

    collator = PaddedCollatorForValueFunction(processor=processor, max_length=512, num_proprio_tokens=1)
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
    )
    if cfg.use_val_set:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,
        )

    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)
        wandb.log(
            {
                "Value Train/Trainable Parameters": sum(param.numel() for param in trainable_params),
                "Value Train/Effective Batch Size": cfg.batch_size
                * cfg.grad_accumulation_steps
                * distributed_state.num_processes,
            },
            step=start_step,
        )

    metric_names = (
        "loss_value",
        "mae",
        "rmse",
        "pred_value_mean",
        "pred_value_min",
        "pred_value_max",
        "target_value_mean",
        "target_value_min",
        "target_value_max",
    )
    recent_metrics = {name: deque(maxlen=cfg.grad_accumulation_steps) for name in metric_names}
    global_step = start_step
    last_saved_step = None

    with tqdm.tqdm(total=cfg.max_steps, initial=start_step, leave=False) as progress:
        vlm.train()
        value_head.train()
        proprio_projector.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            loss, metrics = run_forward_pass(
                vlm=vlm,
                value_head=value_head,
                proprio_projector=proprio_projector,
                batch=batch,
                device_id=device_id,
                proprio_token_id=proprio_token_id,
            )

            (loss / cfg.grad_accumulation_steps).backward()
            for metric_name, value in metrics.items():
                recent_metrics[metric_name].append(value)

            if (batch_idx + 1) % cfg.grad_accumulation_steps != 0:
                continue

            if cfg.lr_warmup_steps > 0 and global_step < cfg.lr_warmup_steps:
                lr_progress = min((global_step + 1) / cfg.lr_warmup_steps, 1.0)
                current_lr = original_lr * (0.1 + 0.9 * lr_progress)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr

            parameters_with_grad = [param for param in trainable_params if param.grad is not None]
            gradient_norm = (
                torch.linalg.vector_norm(
                    torch.stack([param.grad.detach().float().norm(2) for param in parameters_with_grad]), ord=2
                ).item()
                if parameters_with_grad
                else 0.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            progress.update()

            smoothened_metrics = compute_smoothened_metrics(recent_metrics)
            smoothened_metrics["gradient_norm"] = gradient_norm
            smoothened_metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
            progress.set_postfix(loss=f"{smoothened_metrics['loss_value']:.4f}")
            if distributed_state.is_main_process and (
                global_step % cfg.wandb_log_freq == 0 or global_step >= cfg.max_steps
            ):
                log_metrics_to_wandb(smoothened_metrics, "Value Train", global_step)

            should_save = global_step % cfg.save_freq == 0 or global_step >= cfg.max_steps
            if should_save:
                save_training_checkpoint(
                    cfg=cfg,
                    run_dir=run_dir,
                    log_step=global_step,
                    base_vlm_path=base_vlm_path,
                    vlm=vlm,
                    processor=processor,
                    value_head=value_head,
                    proprio_projector=proprio_projector,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    train_dataset=train_dataset,
                    distributed_state=distributed_state,
                )
                last_saved_step = global_step

            if cfg.use_val_set and global_step % cfg.val_freq == 0:
                run_validation(
                    vlm=vlm,
                    value_head=value_head,
                    proprio_projector=proprio_projector,
                    val_dataloader=val_dataloader,
                    device_id=device_id,
                    proprio_token_id=proprio_token_id,
                    log_step=global_step,
                    distributed_state=distributed_state,
                    val_time_limit=cfg.val_time_limit,
                )
                vlm.train()
                value_head.train()
                proprio_projector.train()

            if global_step >= cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break

    if global_step > start_step and last_saved_step != global_step:
        save_training_checkpoint(
            cfg=cfg,
            run_dir=run_dir,
            log_step=global_step,
            base_vlm_path=base_vlm_path,
            vlm=vlm,
            processor=processor,
            value_head=value_head,
            proprio_projector=proprio_projector,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dataset=train_dataset,
            distributed_state=distributed_state,
        )

    if distributed_state.is_main_process:
        wandb.finish()

if __name__ == "__main__":
    finetune_value_function()
