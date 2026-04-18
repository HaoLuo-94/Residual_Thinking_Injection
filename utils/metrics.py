"""
Evaluation metrics and utilities.

Provides standardized evaluation functions for comparing model outputs
against ground truth answers.
"""
import re
import json
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field


# Punctuation marks to strip during normalization
PUNCTUATION = [
    '.</s>', '.\n', '.', ';', '!', ',', '?', '\n',
    '</s>', '<pad>', ' ', ':', '"', "'", '(', ')',
    '[', ']', '{', '}', '-', '_', '/', '\\', '*', 
    '&', '^', '%', '$', '#', '@', '~', '`', '|', 
    '<', '>', '=', '+'
]


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract the answer from \\boxed{} format.
    
    Supports multiple boxed formats:
    - \\boxed{answer}
    - \\boxed:{answer}
    - boxed{answer}
    - boxed:{answer}
    
    Args:
        text: Input text that may contain boxed answer.
    
    Returns:
        Extracted answer string if found, None otherwise.
    """
    if not text:
        return None
    
    # Pattern to match various boxed formats:
    # \\boxed{...}, \\boxed:{...}, boxed{...}, boxed:{...}
    # Also handles nested braces by matching the outermost pair
    patterns = [
        r'\\boxed\s*[:\{]\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',  # \\boxed:{...} or \\boxed{{...}}
        r'\\boxed\s*[:\{]\s*([^{}\s][^{}\n]*)',  # \\boxed:{answer} without braces
        r'boxed\s*[:\{]\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',  # boxed:{...}
        r'boxed\s*[:\{]\s*([^{}\s][^{}\n]*)',  # boxed:{answer} without braces
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # Return the last match (usually the final answer)
            return matches[-1].strip()
    
    return None


def normalize_answer(answer: str) -> str:
    """
    Normalize answer string for comparison.
    
    Performs:
    1. Strip whitespace
    2. Convert to lowercase
    3. Take only the first line/sentence if multi-line
    4. Remove punctuation
    
    Args:
        answer: Raw answer string.
    
    Returns:
        Normalized answer string.
    """
    if not answer:
        return ""
    
    # Take first line and convert to lowercase
    ans = str(answer).strip().lower().split('\n')[0]
    
    # Remove all punctuation from the list
    for p in PUNCTUATION:
        ans = ans.replace(p.lower(), "")
    
    return ans.strip()


def exact_match(prediction: str, reference: str) -> bool:
    """
    Check if prediction exactly matches reference after normalization.
    
    Args:
        prediction: Model output.
        reference: Ground truth.
    
    Returns:
        True if normalized strings match exactly.
    """
    return normalize_answer(prediction) == normalize_answer(reference)


def compute_accuracy(prediction: str, reference: str) -> bool:
    """
    Compute whether prediction matches reference.
    
    Uses flexible matching:
    1. First check for boxed format and extract answer from it
    2. Check if reference is contained in the extracted/prediction answer
    3. Exact match after normalization
    4. Reference contained at start of prediction
    5. Single-character reference matching
    
    Args:
        prediction: Model output string.
        reference: Ground truth answer.
    
    Returns:
        True if prediction is considered correct.
    """
    norm_ref = normalize_answer(reference)
    
    # Empty reference - cannot match
    if not norm_ref:
        return False
    
    # Priority 1: Check for boxed format and extract answer
    boxed_answer = extract_boxed_answer(prediction)
    if boxed_answer is not None:
        norm_boxed = normalize_answer(boxed_answer)
        # Exact match
        if norm_boxed == norm_ref:
            return True
        # For boxed answer, if extraction succeeded but doesn't match, still try fallback
    
    # Fallback: Standard matching without boxed extraction
    norm_pred = normalize_answer(prediction)
    
    # Exact match
    if norm_pred == norm_ref:
        return True
    # Single character matching (for A/B/C/D style answers)
    if len(norm_ref) <= 2:
        # Check first characters
        if norm_pred and norm_pred[0] == norm_ref[0]:
            return True
    
    return False


def compute_ifeval_accuracy(prediction: str, reference: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Compute IFEval accuracy by verifying instruction-following.
    
    IFEval requires checking whether the model's response satisfies
    all verifiable instructions in the prompt (e.g., word count,
    format requirements, keyword inclusion, etc.).
    
    Args:
        prediction: Model output string.
        reference: JSON-encoded instruction metadata containing:
            - instruction_id_list: List of instruction IDs to verify
            - kwargs: List of parameters for each instruction
    
    Returns:
        Tuple of (is_correct, details) where:
            - is_correct: True if ALL instructions are satisfied
            - details: Dict with per-instruction results
    """
    try:
        # Import IFEval verification functions
        from utils.ifeval_instructions import verify_all_instructions
        
        # Parse the reference metadata
        try:
            metadata = json.loads(reference)
        except (json.JSONDecodeError, TypeError):
            # If not valid JSON or not a string, fall back to standard evaluation
            return compute_accuracy(prediction, reference), {}
            
        # Check if this is an IFEval task
        if not isinstance(metadata, dict) or metadata.get("task_type") != "ifeval":
            # Not an IFEval task, fall back to standard evaluation
            return compute_accuracy(prediction, reference), {}
        
        instruction_id_list = metadata.get("instruction_id_list", [])
        kwargs_list = metadata.get("kwargs", [])
        
        # Verify all instructions
        all_satisfied, results = verify_all_instructions(
            prediction, instruction_id_list, kwargs_list
        )
        
        return all_satisfied, {
            "instruction_results": results,
            "total_instructions": len(instruction_id_list),
            "satisfied_instructions": sum(results)
        }
    
    except (json.JSONDecodeError, ImportError, AttributeError, TypeError) as e:
        # If any other error occurs, fall back to standard evaluation
        return compute_accuracy(prediction, reference), {"error": str(e)}


def is_ifeval_reference(reference: str) -> bool:
    """
    Check if a reference is IFEval format (JSON with task_type).
    
    Args:
        reference: The reference string to check.
    
    Returns:
        True if reference is IFEval format, False otherwise.
    """
    try:
        if not isinstance(reference, str):
            return False
        metadata = json.loads(reference)
        if not isinstance(metadata, dict):
            return False
        return metadata.get("task_type") == "ifeval"
    except (json.JSONDecodeError, TypeError, AttributeError):
        return False


@dataclass
class EvaluationResult:
    """
    Results from evaluation run.
    
    Attributes:
        accuracy: Overall accuracy (correct / total).
        total: Total number of samples.
        correct: Number of correct predictions.
        predictions: List of (prediction, reference, is_correct) tuples.
        ifeval_metrics: Optional IFEval-specific metrics (only for IFEval tasks).
    """
    accuracy: float = 0.0
    total: int = 0
    correct: int = 0
    predictions: List[Tuple[str, str, bool]] = field(default_factory=list)
    ifeval_metrics: Optional[Dict[str, Any]] = None


class Evaluator:
    """
    Evaluation handler for model outputs.
    
    Provides:
    - Single sample evaluation
    - Batch evaluation
    - Result aggregation and reporting
    - IFEval-specific evaluation
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize evaluator.
        
        Args:
            verbose: Whether to print per-sample results.
        """
        self.verbose = verbose
        self.reset()
    
    def reset(self):
        """Reset evaluation state."""
        self.predictions = []
        self.correct_count = 0
        self.total_count = 0
        # IFEval-specific tracking
        self.ifeval_mode = False
        self.ifeval_instruction_results = []
        self.ifeval_total_instructions = 0
        self.ifeval_satisfied_instructions = 0
    
    def evaluate_single(
        self,
        prediction: str,
        reference: str,
        record: bool = True
    ) -> bool:
        """
        Evaluate a single prediction.
        
        Args:
            prediction: Model output.
            reference: Ground truth.
            record: Whether to record this evaluation.
        
        Returns:
            True if prediction is correct.
        """
        # Check if this is an IFEval evaluation
        if is_ifeval_reference(reference):
            return self._evaluate_ifeval_single(prediction, reference, record)
        
        is_correct = compute_accuracy(prediction, reference)
        
        if record:
            self.predictions.append((prediction, reference, is_correct))
            self.total_count += 1
            if is_correct:
                self.correct_count += 1
        
        if self.verbose:
            status = "✓" if is_correct else "✗"
            # Only truncate and add ellipsis if prediction is longer than 50 chars
            pred_display = prediction[:128] + "..." if len(prediction) > 50 else prediction
            print(f"{status} Pred: '{pred_display}' | Ref: '{reference}'")
        
        return is_correct
    
    def _evaluate_ifeval_single(
        self,
        prediction: str,
        reference: str,
        record: bool = True
    ) -> bool:
        """
        Evaluate a single IFEval prediction.
        
        Args:
            prediction: Model output.
            reference: JSON-encoded instruction metadata.
            record: Whether to record this evaluation.
        
        Returns:
            True if ALL instructions are satisfied.
        """
        self.ifeval_mode = True
        is_correct, details = compute_ifeval_accuracy(prediction, reference)
        
        if record:
            self.predictions.append((prediction, reference, is_correct))
            self.total_count += 1
            if is_correct:
                self.correct_count += 1
            
            # Track instruction-level statistics
            if "instruction_results" in details:
                self.ifeval_instruction_results.append(details["instruction_results"])
                self.ifeval_total_instructions += details.get("total_instructions", 0)
                self.ifeval_satisfied_instructions += details.get("satisfied_instructions", 0)
        
        if self.verbose:
            status = "✓" if is_correct else "✗"
            pred_display = prediction[:50] + "..." if len(prediction) > 50 else prediction
            satisfied = details.get("satisfied_instructions", 0)
            total = details.get("total_instructions", 0)
            print(f"{status} Pred: '{pred_display}' | Instructions: {satisfied}/{total}")
        
        return is_correct
    
    def evaluate_batch(
        self,
        predictions: List[str],
        references: List[str]
    ) -> List[bool]:
        """
        Evaluate a batch of predictions.
        
        Args:
            predictions: List of model outputs.
            references: List of ground truths.
        
        Returns:
            List of correctness flags.
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")
        
        results = []
        for pred, ref in zip(predictions, references):
            results.append(self.evaluate_single(pred, ref))
        
        return results
    
    def get_accuracy(self) -> float:
        """Get current accuracy."""
        if self.total_count == 0:
            return 0.0
        return self.correct_count / self.total_count
    
    def get_result(self) -> EvaluationResult:
        """Get full evaluation result."""
        result = EvaluationResult(
            accuracy=self.get_accuracy(),
            total=self.total_count,
            correct=self.correct_count,
            predictions=self.predictions.copy()
        )
        
        # Add IFEval-specific metrics if in IFEval mode
        if self.ifeval_mode:
            result.ifeval_metrics = {
                "prompt_strict_accuracy": self.get_accuracy(),  # All instructions satisfied
                "instruction_accuracy": (
                    self.ifeval_satisfied_instructions / self.ifeval_total_instructions
                    if self.ifeval_total_instructions > 0 else 0.0
                ),
                "total_prompts": self.total_count,
                "prompts_all_correct": self.correct_count,
                "total_instructions": self.ifeval_total_instructions,
                "satisfied_instructions": self.ifeval_satisfied_instructions
            }
        
        return result
    
    def print_summary(self, name: str = "Evaluation"):
        """Print evaluation summary."""
        acc = self.get_accuracy()
        print(f"\n{'='*50}")
        print(f"{name} Results")
        print(f"{'='*50}")
        print(f"Accuracy: {acc:.4f} ({self.correct_count}/{self.total_count})")
        print(f"{'='*50}\n")


def compare_results(
    baseline: EvaluationResult,
    intervention: EvaluationResult,
    name: str = "Comparison"
) -> Dict:
    """
    Compare baseline and intervention results.
    
    Args:
        baseline: Baseline evaluation result.
        intervention: Intervention evaluation result.
        name: Name for the comparison.
    
    Returns:
        Dictionary with comparison metrics.
    """
    delta = intervention.accuracy - baseline.accuracy
    relative_improvement = delta / baseline.accuracy if baseline.accuracy > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"{name}")
    print(f"{'='*50}")
    print(f"Baseline:     {baseline.accuracy:.4f}")
    print(f"Intervention: {intervention.accuracy:.4f}")
    print(f"Δ Absolute:   {delta:+.4f}")
    print(f"Δ Relative:   {relative_improvement:+.2%}")
    print(f"{'='*50}\n")
    
    return {
        "baseline": baseline.accuracy,
        "intervention": intervention.accuracy,
        "delta_absolute": delta,
        "delta_relative": relative_improvement
    }
