"""
Centralized logging utility with configurable level and format.
"""

import logging
import sys
import os
from typing import Optional

try:
    from utils import is_main_process
except ImportError:
    def is_main_process() -> bool:
        return True


class TrainingLogger:
    """Centralized training logger with multi-level and optional file output."""

    _instance: Optional["TrainingLogger"] = None
    _initialized: bool = False

    def __new__(cls) -> "TrainingLogger":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._setup_logger()
            TrainingLogger._initialized = True

    def _setup_logger(self):
        self.logger = logging.getLogger("training")
        self.logger.setLevel(logging.INFO)

        if self.logger.handlers:
            return

        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        log_file = os.environ.get("LOG_FILE")
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def set_level(self, level: str):
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        if level.upper() in level_map:
            self.logger.setLevel(level_map[level.upper()])
            for handler in self.logger.handlers:
                handler.setLevel(level_map[level.upper()])

    def debug(self, message: str, *args, **kwargs):
        if is_main_process():
            self.logger.debug(message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs):
        if is_main_process():
            self.logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs):
        if is_main_process():
            self.logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs):
        if is_main_process():
            self.logger.error(message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs):
        if is_main_process():
            self.logger.critical(message, *args, **kwargs)


_global_logger: Optional[TrainingLogger] = None


def get_logger() -> TrainingLogger:
    global _global_logger
    if _global_logger is None:
        _global_logger = TrainingLogger()
    return _global_logger


def set_log_level(level: str):
    get_logger().set_level(level)


def log_debug(message: str, *args, **kwargs):
    if is_main_process():
        get_logger().debug(message, *args, **kwargs)


def log_info(message: str, *args, **kwargs):
    if is_main_process():
        get_logger().info(message, *args, **kwargs)


def log_warning(message: str, *args, **kwargs):
    if is_main_process():
        get_logger().warning(message, *args, **kwargs)


def log_error(message: str, *args, **kwargs):
    if is_main_process():
        get_logger().error(message, *args, **kwargs)


def log_critical(message: str, *args, **kwargs):
    if is_main_process():
        get_logger().critical(message, *args, **kwargs)


def log_training_parameters(args, model, tokenizer, ds_config, num_training_steps=None):
    """Log all training parameters (aligned with deepspeed_train.py args)."""
    if not is_main_process():
        return
    import torch
    try:
        from utils import get_world_size
    except ImportError:
        def get_world_size() -> int:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                return torch.distributed.get_world_size()
            return int(os.environ.get("WORLD_SIZE", "1"))

    logger = get_logger()
    logger.info("=" * 80)
    logger.info(
        "🚀 TRAINING CONFIGURATION in rank %d",
        torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
    )
    logger.info("=" * 80)

    # Model & Data
    logger.info("📋 MODEL & DATA CONFIGURATION")
    logger.info("-" * 40)
    logger.info(f"  Model Path:           {args.model_name_or_path}")
    logger.info(f"  Dataset:              {args.dataset}")
    logger.info(f"  Dataset Split:        {args.dataset_split}")
    if getattr(args, "finetune_data_format", None):
        logger.info(f"  Finetune Data Format: {args.finetune_data_format}")
    logger.info(f"  Block Size:           {args.block_size}")
    logger.info(f"  Model Dtype:          {args.torch_dtype}")
    logger.info(f"  Tokenizer Vocab Size: {tokenizer.vocab_size}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Model Parameters:     {total_params:,}")
    logger.info(f"  Trainable Parameters: {trainable_params:,}")
    logger.info("")

    # Curriculum Learning
    if hasattr(args, "cl_k_ratio"):
        logger.info("📚 CURRICULUM LEARNING")
        logger.info("-" * 40)
        logger.info(f"  CL K Ratio:           {args.cl_k_ratio}")
        logger.info(f"  CL Rehearsal Ratio:   {args.cl_rehearsal_ratio}")
        logger.info("")

    # Training Hyperparameters
    logger.info("⚙️  TRAINING HYPERPARAMETERS")
    logger.info("-" * 40)
    logger.info(f"  Learning Rate:        {args.learning_rate}")
    logger.info(f"  Number of Epochs:     {args.num_epochs}")
    if getattr(args, "max_steps", None) is not None:
        logger.info(f"  Max Steps:            {args.max_steps:,}")
        logger.info("  Training Strategy:    Stop at min(epochs, max_steps)")
    else:
        logger.info("  Training Strategy:    Complete all epochs")
    logger.info(f"  Per Device Train Batch: {args.per_device_train_batch_size}")
    logger.info(f"  Per Device Eval Batch:  {args.per_device_eval_batch_size}")
    logger.info(f"  Gradient Accum Steps: {args.gradient_accumulation_steps}")
    logger.info(f"  Warmup Ratio:         {args.warmup_ratio}")
    logger.info(f"  Grad Clip Value:      {args.grad_clip_value}")
    logger.info(f"  Random Seed:          {args.seed}")
    model_size = sum(p.numel() for p in model.parameters()) / 1e9
    trainable_model_size = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9
    logger.info(f"  Trainable Model Size: {trainable_model_size:.2f}B")
    logger.info(f"  Total Model Size:     {model_size:.2f}B")
    if num_training_steps:
        logger.info(f"  Total Training Steps: {num_training_steps:,}")
        if getattr(args, "max_steps", None) is not None:
            logger.info(f"  (Limited by max_steps: {args.max_steps:,})")
    logger.info("")

    # Data
    logger.info("📊 DATA CONFIGURATION")
    logger.info("-" * 40)
    logger.info(f"  Validation Split Ratio: {args.val_split_ratio}")
    logger.info(f"  Dataloader Num Workers: {args.dataloader_num_workers}")
    dataset_length = getattr(args, "dataset_length", None)
    if dataset_length is not None:
        logger.info(f"  Dataset Length:       {dataset_length:,}")
        if getattr(args, "val_split_ratio", 0) > 0:
            train_samples = int(dataset_length * (1 - args.val_split_ratio))
            val_samples = dataset_length - train_samples
            logger.info(f"  Training Samples:     {train_samples:,}")
            logger.info(f"  Validation Samples:   {val_samples:,}")
    logger.info("")

    # DeepSpeed
    logger.info("⚡ DEEPSPEED CONFIGURATION")
    logger.info("-" * 40)
    logger.info(f"  Config File:          {args.deepspeed}")
    zero_config = ds_config.get("zero_optimization", {})
    stage = zero_config.get("stage", "N/A")
    opt_device = zero_config.get("offload_optimizer", {}).get("device", "N/A")
    param_device = zero_config.get("offload_param", {}).get("device", "N/A")
    logger.info(f"  Zero Stage:           {stage}")
    logger.info(f"  Offload Optimizer:    {opt_device}")
    logger.info(f"  Offload Parameters:   {param_device}")
    logger.info(f"  Train Batch Size:     {ds_config.get('train_batch_size', 'N/A')}")
    logger.info(f"  Micro Batch Per GPU:  {ds_config.get('train_micro_batch_size_per_gpu', 'N/A')}")
    logger.info(f"  Grad Accum Steps:     {ds_config.get('gradient_accumulation_steps', 'N/A')}")
    logger.info(f"  Gradient Clipping:    {ds_config.get('gradient_clipping', 'N/A')}")
    sched = ds_config.get("scheduler", {})
    if sched:
        logger.info(f"  Scheduler Type:       {sched.get('type', 'N/A')}")
        sched_params = sched.get("params", {})
        if sched_params:
            logger.info(f"  Scheduler Params:    {sched_params}")
    logger.info(f"  Gradient Checkpointing: {args.is_grad_checkpointing}")
    logger.info("")

    # System & Logging
    logger.info("💻 SYSTEM & LOGGING")
    logger.info("-" * 40)
    world_size = get_world_size() if torch.distributed.is_initialized() else 1
    global_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size
    logger.info(f"  World Size:           {world_size}")
    logger.info(f"  Global Batch Size:    {global_batch}")
    logger.info(f"  Output Directory:     {args.output_dir}")
    logger.info(f"  Eval Steps:           {getattr(args, 'eval_steps', 'N/A')}")
    logger.info(f"  Save Steps:           {getattr(args, 'save_steps', 'N/A')}")
    logger.info(f"  Save Strategy:        {args.save_strategy}")
    logger.info(f"  Save Total Limit:     {args.save_total_limit}")
    if getattr(args, "mlflow_tracking_uri", None):
        logger.info(f"  MLflow Tracking URI: {args.mlflow_tracking_uri}")
        logger.info(f"  MLflow Experiment:   {getattr(args, 'mlflow_experiment', 'N/A')}")
        logger.info(f"  MLflow Run Name:     {getattr(args, 'mlflow_run_name', 'N/A')}")
    else:
        logger.info("  MLflow:              disabled")
    logger.info("=" * 80)
