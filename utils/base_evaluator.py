#!/usr/bin/env python3
"""
Base evaluator for math/QA datasets: load data, build prompt, generate, extract answer, check correctness.
Subclass and override the hook methods to implement dataset-specific behavior.
"""

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import torch.distributed as dist
from tqdm import tqdm

from utils.timeout import time_limit, EvalTimeout


def get_accelerator():
    try:
        from deepspeed import get_accelerator as _get_accelerator
        return _get_accelerator()
    except ImportError:
        class _CudaAccelerator:
            def device_name(self):
                return "cuda"
        return _CudaAccelerator()

from utils import is_main_process
from utils.logger import get_logger


class BaseMathEvaluator(ABC):
    """
    Parent class for evaluation datasets. Handles:
    - Loading test samples (jsonl)
    - Building prompts from question
    - Running model generation in batches (with distributed sharding)
    - Extracting answer from completion
    - Deciding correctness (prediction vs ground truth)
    """

    # Subclass can override for tqdm and logging
    eval_name: str = "Eval"
    per_sample_timeout: float = 10.0   # Per-sample eval timeout (seconds); <=0 disables

    @abstractmethod
    def get_question_key(self) -> str:
        """Key in sample dict for the question (e.g. 'question' or 'problem')."""
        pass

    @abstractmethod
    def get_ground_truth_key(self) -> str:
        """Key in sample dict for the ground-truth answer (e.g. 'answer')."""
        pass

    def load_dataset(self, test_path: str) -> List[Dict[str, Any]]:
        """Load test samples from a jsonl file. Override if format differs."""
        samples: List[Dict[str, Any]] = []
        with Path(test_path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                samples.append(json.loads(line))
        return samples

    def get_question(self, sample: Dict[str, Any]) -> str:
        """Get the question string from a sample."""
        return sample.get(self.get_question_key(), "")

    def get_ground_truth_raw(self, sample: Dict[str, Any]) -> str:
        """Get the raw ground-truth answer from a sample (before any extraction/normalization)."""
        return sample.get(self.get_ground_truth_key(), "") or ""

    @abstractmethod
    def build_prompt(self, tokenizer, question: str) -> str:
        """Build the full prompt (system + user) for generation."""
        pass

    @abstractmethod
    def extract_answer(self, completion: str) -> Optional[str]:
        """Extract the model's final answer from the completion text (e.g. after ####)."""
        pass

    def extract_ground_truth(self, sample: Dict[str, Any]) -> Optional[str]:
        """
        Get ground truth for comparison. Default: use raw value; override if
        the dataset stores a different format (e.g. full solution with ####).
        """
        raw = self.get_ground_truth_raw(sample)
        if not raw:
            return None
        # If the dataset already stores a plain answer, return it; else subclasses can parse.
        return raw.strip()

    @abstractmethod
    def is_correct(self, prediction: Optional[str], ground_truth: Optional[str], sample: Dict[str, Any]) -> bool:
        """Return True if prediction matches ground truth (both may be normalized)."""
        pass

    def get_detail_record(
        self,
        sample: Dict[str, Any],
        completion: str,
        prediction: Optional[str],
        ground_truth: Optional[str],
        is_correct: bool,
    ) -> Dict[str, Any]:
        """Build one entry for detailed results. Override to add dataset-specific fields."""
        qkey = self.get_question_key()
        gkey = self.get_ground_truth_key()
        return {
            qkey: self.get_question(sample),
            gkey: self.get_ground_truth_raw(sample),
            "ground_truth_extracted": ground_truth,
            "prediction_extracted": prediction,
            "completion": completion,
            "is_correct": is_correct,
        }

    def get_sync_prompt_question(self) -> str:
        """Minimal question for sync batches when rank has no data (DeepSpeed ZeRO needs same collectives)."""
        return "1"

    def evaluate(
        self,
        model,
        tokenizer,
        test_path: str,
        device: Optional[torch.device],
        batch_size: int = 1,
        max_new_tokens: int = 1024,
        return_details: bool = False,
        save_results_path: Optional[str] = None,
    ):
        """
        Main evaluation: load dataset, shard by rank, run generation in batches,
        extract answers, check correctness, all-reduce and return accuracy.
        When save_results_path is set: each rank writes part file, rank 0 merges (no all_gather).
        """
        model.eval()
        base_model = model.module if hasattr(model, "module") else model

        original_use_cache = getattr(base_model.config, "use_cache", True)
        base_model.config.use_cache = True

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"

        all_samples = self.load_dataset(test_path)
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            local_samples = all_samples[rank::world_size]
        else:
            rank = 0
            world_size = 1
            local_samples = all_samples

        local_correct = 0
        local_total = 0
        detailed_results: List[Dict[str, Any]] = []

        # In distributed mode with DeepSpeed ZeRO, all ranks must run same number of
        # model forward/generate calls (they trigger internal collectives).
        if dist.is_available() and dist.is_initialized():
            my_batches = max(1, (len(local_samples) + batch_size - 1) // batch_size)
            max_batches_t = torch.tensor(
                [my_batches],
                dtype=torch.long,
                device=torch.device(
                    get_accelerator().device_name(),
                    dist.get_rank() % torch.cuda.device_count(),
                ),
            )
            dist.all_reduce(max_batches_t, op=dist.ReduceOp.MAX)
            max_batches = int(max_batches_t.item())
        else:
            max_batches = max(1, (len(local_samples) + batch_size - 1) // batch_size)

        try:
            iterator = tqdm(
                range(max_batches),
                desc=f"Evaluating {self.eval_name}",
                ncols=100,
                disable=not is_main_process(),
            )
            for batch_idx in iterator:
                is_sync_batch = batch_idx * batch_size >= len(local_samples)
                if is_sync_batch:
                    # Dummy batch for collective sync (no metrics)
                    sync_q = self.get_sync_prompt_question()
                    prompts = [self.build_prompt(tokenizer, sync_q)]
                else:
                    start = batch_idx * batch_size
                    batch_samples = local_samples[start : start + batch_size]
                    prompts = [
                        self.build_prompt(tokenizer, self.get_question(s))
                        for s in batch_samples
                    ]

                encoded = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                encoded = {k: v.to(device, non_blocking=True) for k, v in encoded.items()}

                with torch.no_grad():
                    gen_max = 1 if is_sync_batch else max_new_tokens
                    generated = base_model.generate(
                        **encoded,
                        max_new_tokens=gen_max,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                prompt_width = encoded["input_ids"].shape[1]
                completions = tokenizer.batch_decode(
                    generated[:, prompt_width:],
                    skip_special_tokens=True,
                )
                if is_sync_batch:
                    continue
                batch_samples = local_samples[
                    batch_idx * batch_size : batch_idx * batch_size + batch_size
                ]
                # for sample, completion in zip(batch_samples, completions):
                #     prediction = self.extract_answer(completion)
                #     ground_truth = self.extract_ground_truth(sample)
                #     correct = self.is_correct(prediction, ground_truth, sample)
                #     local_correct += int(correct)
                #     local_total += 1
                for sample, completion in zip(batch_samples, completions):
                    prediction = self.extract_answer(completion)
                    ground_truth = self.extract_ground_truth(sample)
                    try:
                        with time_limit(self.per_sample_timeout):
                            correct = self.is_correct(prediction, ground_truth, sample)
                    except EvalTimeout:
                        get_logger().warning(
                            f"{self.eval_name} sample timed out "
                            f"after {self.per_sample_timeout}s, marked incorrect"
                        )
                        correct = False
                    local_correct += int(correct)
                    local_total += 1

                    if return_details or save_results_path:
                        detailed_results.append(
                            self.get_detail_record(
                                sample, completion, prediction, ground_truth, correct
                            )
                        )
        finally:
            tokenizer.padding_side = original_padding_side
            base_model.config.use_cache = original_use_cache
            model.train()

        if dist.is_available() and dist.is_initialized():
            reduce_device = torch.device(
                get_accelerator().device_name(),
                dist.get_rank() % torch.cuda.device_count(),
            )
            metrics = torch.tensor(
                [float(local_correct), float(local_total)],
                device=reduce_device,
                dtype=torch.float64,
            )
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            local_correct = int(metrics[0].item())
            local_total = int(metrics[1].item())

            if save_results_path:
                # Each rank with data writes its part file (no all_gather)
                if detailed_results:
                    part_path = Path(save_results_path).parent / (
                        Path(save_results_path).stem + f"_part{rank}.jsonl"
                    )
                    part_path.parent.mkdir(parents=True, exist_ok=True)
                    with part_path.open("w", encoding="utf-8") as f:
                        for result in detailed_results:
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")
                # All ranks must barrier for sync (including ranks with no local samples)
                dist.barrier()
                # Rank 0 merges part files into final file
                if rank == 0:
                    out_path = Path(save_results_path)
                    with out_path.open("w", encoding="utf-8") as outf:
                        for r in range(world_size):
                            p = out_path.parent / (out_path.stem + f"_part{r}.jsonl")
                            if p.exists():
                                with p.open("r", encoding="utf-8") as inf:
                                    outf.write(inf.read())
                                p.unlink()
                    get_logger().info(f"Evaluation results saved to {out_path}")
                detailed_results = []
            elif return_details and not save_results_path:
                # Legacy: all_gather when return_details without save (kept for compatibility)
                gathered_results = [None] * world_size
                dist.all_gather_object(gathered_results, detailed_results)
                detailed_results = []
                for rank_results in gathered_results:
                    if rank_results is not None:
                        detailed_results.extend(rank_results)
        else:
            # Non-distributed: early return when no data to avoid division by zero
            if local_total == 0:
                if return_details:
                    return 0.0, []
                return 0.0
            # Non-distributed: write directly when save_results_path is set
            if save_results_path and detailed_results:
                out_path = Path(save_results_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with out_path.open("w", encoding="utf-8") as f:
                    for result in detailed_results:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                get_logger().info(f"Evaluation results saved to {out_path}")
                detailed_results = []

        accuracy = local_correct / max(local_total, 1)

        if return_details:
            return accuracy, detailed_results
        return accuracy


class BaseCodeEvaluator(ABC):
    """
    Base class for code-generation benchmarks (HumanEval, MBPP, etc.):
    load parquet/jsonl samples, shard by rank, generate, extract code, run tests.
    Mirrors BaseMathEvaluator distributed ZeRO sync batches and optional save_results_path.
    """

    eval_name: str = "CodeEval"
    per_sample_timeout: float = 15.0

    @abstractmethod
    def build_prompt(self, tokenizer, item: Dict[str, Any]) -> str:
        pass

    @abstractmethod
    def build_test_code(self, item: Dict[str, Any], generated_code: str):
        """Build executable test closure or namespace+tests from generated code."""
        pass

    @abstractmethod
    def run_test(self, test_code) -> Dict[str, Any]:
        """Run tests; return dict with at least ``pass``: bool."""
        pass

    @abstractmethod
    def extract_code_block(self, text: str) -> str:
        pass

    @abstractmethod
    def is_correct(self, generated_code: Optional[str], item: Dict[str, Any]) -> bool:
        pass

    @abstractmethod
    def get_sync_dummy_item(self) -> Dict[str, Any]:
        """Minimal sample for ZeRO alignment batches when a rank has no local data (not scored)."""
        pass

    # def load_dataset(self, test_path: str) -> List[Dict[str, Any]]:
    #     """Default: HuggingFace parquet. Subclasses may override (e.g. MBPP HF fallback)."""
    #     from datasets import load_dataset

    #     ds = load_dataset("parquet", data_files=test_path)
    #     key = "train" if "train" in ds else next(iter(ds))
    #     split = ds[key]
    #     if hasattr(split, "to_list"):
    #         return split.to_list()
    #     return [split[i] for i in range(len(split))]
    def load_dataset(self, test_path: str) -> List[Dict[str, Any]]:
        """Load code benchmark samples from jsonl/json/parquet."""
        suffix = Path(test_path).suffix.lower()

        if suffix in {".jsonl", ".json"}:
            samples: List[Dict[str, Any]] = []
            with Path(test_path).open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    samples.append(json.loads(line))
            return samples

        if suffix == ".parquet":
            from datasets import load_dataset

            ds = load_dataset("parquet", data_files=test_path)
            key = "train" if "train" in ds else next(iter(ds))
            split = ds[key]
            if hasattr(split, "to_list"):
                return split.to_list()
            return [split[i] for i in range(len(split))]

        raise ValueError(f"Unsupported dataset format: {test_path}")

    def get_detail_record(
        self,
        sample: Dict[str, Any],
        completion: str,
        correct: bool,
    ) -> Dict[str, Any]:
        return {
            "completion": completion,
            "is_correct": correct,
        }

    def evaluate(
        self,
        model,
        tokenizer,
        test_path: str,
        device: Optional[torch.device],
        batch_size: int = 1,
        max_new_tokens: int = 1024,
        return_details: bool = False,
        save_results_path: Optional[str] = None,
    ):
        model.eval()
        base_model = model.module if hasattr(model, "module") else model

        original_use_cache = getattr(base_model.config, "use_cache", True)
        base_model.config.use_cache = True

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"

        all_samples = self.load_dataset(test_path)
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            local_samples = all_samples[rank::world_size]
        else:
            rank = 0
            world_size = 1
            local_samples = all_samples

        local_correct = 0
        local_total = 0
        detailed_results: List[Dict[str, Any]] = []

        if dist.is_available() and dist.is_initialized():
            my_batches = max(1, (len(local_samples) + batch_size - 1) // batch_size)
            max_batches_t = torch.tensor(
                [my_batches],
                dtype=torch.long,
                device=torch.device(
                    get_accelerator().device_name(),
                    dist.get_rank() % max(1, torch.cuda.device_count()),
                ),
            )
            dist.all_reduce(max_batches_t, op=dist.ReduceOp.MAX)
            max_batches = int(max_batches_t.item())
        else:
            max_batches = max(1, (len(local_samples) + batch_size - 1) // batch_size)

        sync_item = self.get_sync_dummy_item()

        try:
            iterator = tqdm(
                range(max_batches),
                desc=f"Evaluating {self.eval_name}",
                ncols=100,
                disable=not is_main_process(),
            )
            for batch_idx in iterator:
                is_sync_batch = batch_idx * batch_size >= len(local_samples)
                if is_sync_batch:
                    prompts = [self.build_prompt(tokenizer, sync_item)]
                else:
                    start = batch_idx * batch_size
                    batch_samples = local_samples[start : start + batch_size]
                    prompts = [self.build_prompt(tokenizer, s) for s in batch_samples]

                encoded = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=4096,
                )
                encoded = {k: v.to(device, non_blocking=True) for k, v in encoded.items()}

                with torch.no_grad():
                    gen_max = 1 if is_sync_batch else max_new_tokens
                    generated = base_model.generate(
                        **encoded,
                        max_new_tokens=gen_max,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                prompt_width = encoded["input_ids"].shape[1]
                completions = tokenizer.batch_decode(
                    generated[:, prompt_width:],
                    skip_special_tokens=True,
                )
                if is_sync_batch:
                    continue

                batch_samples = local_samples[
                    batch_idx * batch_size : batch_idx * batch_size + batch_size
                ]
                for sample, completion in zip(batch_samples, completions):
                    try:
                        with time_limit(self.per_sample_timeout):
                            extracted = self.extract_code_block(completion)
                            correct = self.is_correct(extracted, sample)
                    except EvalTimeout:
                        get_logger().warning(
                            f"{self.eval_name} sample timed out "
                            f"after {self.per_sample_timeout}s, marked incorrect"
                        )
                        correct = False
                    except BaseException as e:
                        if isinstance(e, KeyboardInterrupt):
                            raise e
                        get_logger().warning(f"Error evaluating {self.eval_name} sample: {e}")
                        correct = False
                    local_correct += int(correct)
                    local_total += 1

                    if return_details or save_results_path:
                        rec = self.get_detail_record(sample, completion, correct)
                        detailed_results.append(rec)
        finally:
            tokenizer.padding_side = original_padding_side
            base_model.config.use_cache = original_use_cache
            model.train()

        if dist.is_available() and dist.is_initialized():
            reduce_device = torch.device(
                get_accelerator().device_name(),
                dist.get_rank() % max(1, torch.cuda.device_count()),
            )
            metrics = torch.tensor(
                [float(local_correct), float(local_total)],
                device=reduce_device,
                dtype=torch.float64,
            )
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            local_correct = int(metrics[0].item())
            local_total = int(metrics[1].item())

            if save_results_path:
                if detailed_results:
                    part_path = Path(save_results_path).parent / (
                        Path(save_results_path).stem + f"_part{rank}.jsonl"
                    )
                    part_path.parent.mkdir(parents=True, exist_ok=True)
                    with part_path.open("w", encoding="utf-8") as f:
                        for result in detailed_results:
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")
                dist.barrier()
                if rank == 0:
                    out_path = Path(save_results_path)
                    with out_path.open("w", encoding="utf-8") as outf:
                        for r in range(world_size):
                            p = out_path.parent / (out_path.stem + f"_part{r}.jsonl")
                            if p.exists():
                                with p.open("r", encoding="utf-8") as inf:
                                    outf.write(inf.read())
                                p.unlink()
                    get_logger().info(f"Evaluation results saved to {out_path}")
                detailed_results = []
            elif return_details and not save_results_path:
                gathered_results = [None] * world_size
                dist.all_gather_object(gathered_results, detailed_results)
                detailed_results = []
                for rank_results in gathered_results:
                    if rank_results is not None:
                        detailed_results.extend(rank_results)
        else:
            if local_total == 0:
                if return_details:
                    return 0.0, []
                return 0.0
            if save_results_path and detailed_results:
                out_path = Path(save_results_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with out_path.open("w", encoding="utf-8") as f:
                    for result in detailed_results:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                get_logger().info(f"Evaluation results saved to {out_path}")
                detailed_results = []

        accuracy = local_correct / max(local_total, 1)

        if return_details:
            return accuracy, detailed_results
        return accuracy

