#!/usr/bin/env python3
"""Training and evaluation utilities for DeepSpeed/distributed training.

Includes: process/rank helpers, DeepSpeed config patching, model checkpoint saving,
perplexity evaluation, and max_steps handling. Training config logging is in logger.py.
"""

import os
import shutil
import torch
import torch.distributed as dist
from typing import Optional, Union
from pathlib import Path

from deepspeed import get_accelerator

from logger import log_info, log_error, log_warning
from logger import get_logger
from tqdm import tqdm

logger = get_logger()


def is_main_process() -> bool:
    """Check if current process is main process."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def get_world_size() -> int:
    """Get world size for distributed training."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", 1))


def cleanup_distributed():
    """Clean up distributed process group."""
    try:
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            dist.destroy_process_group()
            log_info(f"🧹 Distributed cleanup completed for rank {rank}")
    except Exception as e:
        log_error(f"❌ Error during distributed cleanup: {e}")


def patch_deepspeed_config(
    ds_config: dict,
    model_dtype,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    world_size: int,
    learning_rate: Optional[float] = None,
    num_training_steps: Optional[int] = None,
    warmup_ratio: Optional[float] = None,
    grad_clip_value: Optional[float] = None,
) -> tuple:
    """
    Patch DeepSpeed config dict for dtype, batch, scheduler, gradient clipping, etc.
    Args:
        ds_config: DeepSpeed config dict
        model_dtype: torch.dtype
        per_device_train_batch_size: int
        gradient_accumulation_steps: int
        world_size: int
        learning_rate: float (optional)
        num_training_steps: int (optional)
        warmup_ratio: float (optional)
        grad_clip_value: float (optional)
    Returns:
        Patched ds_config, dtype_str ("fp16", "bf16", "fp32")
    """
    import torch

    # 1. Dtype
    if model_dtype == torch.bfloat16:
        dtype_str = "bf16"
        ds_config["bf16"] = {"enabled": True}
        ds_config.pop("fp16", None)
    elif model_dtype == torch.float16:
        dtype_str = "fp16"
        ds_config["fp16"] = {"enabled": True}
        ds_config.pop("bf16", None)
    elif model_dtype == torch.float32:
        dtype_str = "fp32"
        ds_config.pop("fp16", None)
        ds_config.pop("bf16", None)
    else:
        raise ValueError(f"Unsupported dtype: {model_dtype}")

    # 2. Batch size
    ds_config["gradient_accumulation_steps"] = gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = per_device_train_batch_size
    ds_config["train_batch_size"] = (
        per_device_train_batch_size * world_size * gradient_accumulation_steps
    )

    # 3. Scheduler - handle all "auto" string params
    if "scheduler" in ds_config and "params" in ds_config["scheduler"]:
        params = ds_config["scheduler"]["params"]

        # handle warmup_min_lr
        if "warmup_min_lr" in params:
            if isinstance(params["warmup_min_lr"], str):
                params["warmup_min_lr"] = 0.0

        # handle warmup_max_lr
        if "warmup_max_lr" in params:
            if isinstance(params["warmup_max_lr"], str):
                if learning_rate is not None:
                    params["warmup_max_lr"] = float(learning_rate)
                else:
                    params["warmup_max_lr"] = 2e-5  # default learning rate

        # handle min_lr
        if "min_lr" in params:
            if isinstance(params["min_lr"], str):
                params["min_lr"] = 0.0

        # handle warmup_num_steps
        if "warmup_num_steps" in params:
            if isinstance(params["warmup_num_steps"], str):
                if num_training_steps is not None and warmup_ratio is not None:
                    warmup_steps = int(num_training_steps * warmup_ratio)
                    params["warmup_num_steps"] = warmup_steps
                else:
                    params["warmup_num_steps"] = 100  # default warmup steps

        # remove params not supported by deepspeed scheduler
        if "total_num_steps" in params:
            params.pop("total_num_steps")

    # 4. Gradient clipping - handle "auto" string
    if "gradient_clipping" in ds_config:
        if isinstance(ds_config["gradient_clipping"], str):
            if grad_clip_value is not None:
                ds_config["gradient_clipping"] = float(grad_clip_value)
            else:
                ds_config["gradient_clipping"] = 1.0  # default gradient clip value
    elif grad_clip_value is not None:
        ds_config["gradient_clipping"] = float(grad_clip_value)

    return ds_config, dtype_str


def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor


def handle_max_steps(args, global_step: int, training_finished: bool) -> bool:
    """Curriculum learning: only check max_steps in training loop; eval and save at each stage end."""
    if getattr(args, "max_steps", None) is not None and global_step >= args.max_steps:
        if is_main_process():
            logger.info(f"🎯 Reached max steps ({args.max_steps}). Stopping...")
        return True
    return training_finished


def evaluate_perplexity(
    model,
    dataloader,
    device: Optional[torch.device],
) -> float:
    """Evaluate model perplexity on given dataloader.

    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on

    Returns:
        Perplexity score (float)
    """
    model.eval()
    # Token-weighted accumulation for correct global avg loss and perplexity
    total_loss_sum = 0.0
    total_tokens = 0
    try:
        dataloader_len = len(dataloader)
    except (TypeError, AttributeError):
        dataloader_len = None

    dataloader_iter = tqdm(
        dataloader,
        desc="Evaluating",
        ncols=100,
        total=dataloader_len,
        disable=not is_main_process()
    )

    batch_count = 0
    for batch_idx, batch in enumerate(dataloader_iter):
        # Move batch to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)

        # Count only tokens that contribute to loss (labels != -100), matching model's answer-only loss
        batch_tokens = (batch["labels"] != -100).sum().item()

        with torch.no_grad():
            outputs = model(**batch, use_cache=False)

        loss = outputs.loss
        # Token-weighted: total_loss = sum(loss_per_batch * batch_tokens) / total_tokens
        total_loss_sum += loss.item() * batch_tokens
        total_tokens += batch_tokens
        batch_count += 1
        if dataloader_len is not None and batch_count >= dataloader_len:
            break

    # Ensure all processes have processed at least some data
    if total_tokens == 0:
        log_warning(
            "No evaluation data processed - this may indicate an issue "
            "with the dataloader"
        )
        return float('inf')

    # In distributed: all_reduce(SUM) total_loss_sum and total_tokens, then average.
    # Place loss_tensor on current GPU to avoid ZeRO-3 param offload to CPU causing NCCL all_reduce error.
    if torch.distributed.is_initialized():
        reduce_device = torch.device(
            get_accelerator().device_name(),
            dist.get_rank() % torch.cuda.device_count(),
        )
        loss_tensor = torch.tensor([total_loss_sum, float(total_tokens)], device=reduce_device, dtype=torch.float64)
        torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
        total_loss_sum = loss_tensor[0].item()
        total_tokens = int(loss_tensor[1].item())
    avg_loss = total_loss_sum / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return perplexity


def _ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists and return Path object."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cleanup_old_checkpoints(checkpoint_dir: Path, save_total_limit: int):
    """Remove old checkpoints to maintain save_total_limit."""
    if save_total_limit <= 0:
        return

    # Find all step directories
    step_dirs = []
    for item in checkpoint_dir.iterdir():
        if item.is_dir() and item.name.startswith("step-"):
            try:
                step_num = int(item.name.split("-")[1])
                step_dirs.append((step_num, item))
            except (ValueError, IndexError):
                continue

    # Sort by step number and remove oldest if exceeding limit
    step_dirs.sort(key=lambda x: x[0])
    while len(step_dirs) > save_total_limit:
        _, old_dir = step_dirs.pop(0)
        if is_main_process():
            logger.info(f"🗑️  Removing old checkpoint: {old_dir}")
        shutil.rmtree(old_dir, ignore_errors=True)


def _save_hf_model_zero3_compatible(model_engine, output_dir: Path, tokenizer=None):
    """Save HuggingFace model with ZeRO-3 compatibility."""
    try:
        # Check if we're using ZeRO-3 (parameter partitioning)
        if hasattr(model_engine, 'zero_optimization_partition_weights') and \
           model_engine.zero_optimization_partition_weights():
            # ZeRO-3: Use DeepSpeed's save method for parameter gathering
            if is_main_process():
                logger.info("🔄 ZeRO-3 detected: Gathering distributed parameters...")

            # Save model state dict in a way that works with ZeRO-3
            model_engine.save_16bit_model(str(output_dir), "pytorch_model.bin")

            # Save config manually since save_16bit_model doesn't save it
            if hasattr(model_engine.module, 'config'):
                model_engine.module.config.save_pretrained(str(output_dir))

            if is_main_process():
                logger.info("✅ ZeRO-3 model saved successfully")
        else:
            # ZeRO-1/2 or no ZeRO: Direct save
            if is_main_process():
                logger.info("🔄 Saving model in standard mode...")
            model_engine.module.save_pretrained(str(output_dir))
            if is_main_process():
                logger.info("✅ Standard model saved successfully")

        # Save tokenizer if provided
        if tokenizer is not None and is_main_process():
            tokenizer.save_pretrained(str(output_dir))
            logger.info("✅ Tokenizer saved successfully")

    except Exception as e:
        if is_main_process():
            logger.error(f"❌ Failed to save HuggingFace model: {e}")
        raise


def save_model_checkpoint(
    model_engine,
    tokenizer,
    output_dir: str,
    step: Optional[int] = None,
    is_final: bool = False,
    save_strategy: str = "deepspeed",
    save_total_limit: int = 3,
):
    """
    Unified model saving function that handles different scenarios.

    Args:
        model_engine: DeepSpeed model engine
        tokenizer: Tokenizer to save alongside model
        output_dir: Base output directory
        step: Current training step (None for final save)
        is_final: Whether this is the final save
        save_strategy: "deepspeed", "hf", or "both"
        save_total_limit: Maximum number of checkpoints to keep (0 = unlimited)
    """
    if not is_main_process():
        # Only main process handles directory creation and cleanup
        # But all processes participate in model saving
        pass

    output_path = Path(output_dir)

    try:
        if is_final:
            # Final save: save in root directory and final subdirectory
            final_dir = _ensure_directory(output_path / "final")

            if save_strategy in ["deepspeed", "both"]:
                if is_main_process():
                    logger.info("💾 Saving final DeepSpeed checkpoint...")
                model_engine.save_checkpoint(str(output_path / "final" / "deepspeed"))

            if save_strategy in ["hf", "both"]:
                if is_main_process():
                    logger.info("💾 Saving final HuggingFace model...")
                _save_hf_model_zero3_compatible(model_engine, final_dir, tokenizer)

        else:
            # Intermediate save
            if step is None:
                raise ValueError("Step must be provided for intermediate saves")

            step_name = f"step-{step}"

            if save_strategy in ["deepspeed", "both"]:
                checkpoint_dir = _ensure_directory(output_path / "checkpoints")
                if is_main_process():
                    logger.info(f"💾 Saving DeepSpeed checkpoint at step {step}...")
                model_engine.save_checkpoint(str(checkpoint_dir / step_name))

                # Cleanup old checkpoints
                if is_main_process():
                    _cleanup_old_checkpoints(checkpoint_dir, save_total_limit)

            if save_strategy in ["hf", "both"]:
                hf_dir = _ensure_directory(output_path / "hf_models" / step_name)
                if is_main_process():
                    logger.info(f"💾 Saving HuggingFace model at step {step}...")
                _save_hf_model_zero3_compatible(model_engine, hf_dir, tokenizer)

                # Cleanup old HF models
                if is_main_process():
                    _cleanup_old_checkpoints(output_path / "hf_models", save_total_limit)

        if is_main_process():
            logger.info("✅ Model checkpoint saved successfully")

    except Exception as e:
        if is_main_process():
            logger.error(f"❌ Failed to save model checkpoint: {e}")
        raise
