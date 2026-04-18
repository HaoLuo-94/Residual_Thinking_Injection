"""
Dataset loader module.

Provides unified interface for loading various datasets for evaluation.
Supports: COPA, SST-2, SST-5, BoolQ, MMLU, XNLI, WinoGrande, GSM8K, MATH-500, TruthfulQA, IFEval, Spider.
"""
import os
import json
import csv
import random
import re
import itertools
import torch.distributed as dist
from typing import List, Tuple, Optional, Dict, Any, Set
from pathlib import Path
from collections import defaultdict


# Type alias for dataset format: (question, correct_answer, wrong_answer)
DataSample = Tuple[str, str, str]


class DatasetLoader:
    """
    Unified dataset loader supporting multiple benchmark datasets.

    Each dataset is loaded into a standard format:
    [(prompt, correct_answer, wrong_answer), ...]

    This format enables:
    - Direct use in activation intervention methods
    - Rollout-based pair generation
    - Consistent evaluation
    """

    def __init__(self, data_root: str = "./data", model_name: str = ""):
        """
        Initialize dataset loader.

        Args:
            data_root: Root directory containing dataset folders.
            model_name: Name/path of the model (used for prompt format selection).
        """
        self.data_root = Path(data_root)
        self.model_name = model_name.lower() if model_name else ""

        # Spider-specific attributes
        self.spider_dir = self.data_root.parent / "shell" / "spider"
        self.spider_tables_file = self.spider_dir / "tables.json"
        self.spider_databases_dir = self.spider_dir / "database"

        # Load spider tables data
        self.spider_tables = {}
        if self.spider_tables_file.exists():
            with open(self.spider_tables_file, 'r', encoding='utf-8') as f:
                self.spider_tables = {db['db_id']: db for db in json.load(f)}
    
    def load(
        self, 
        dataset_name: str,
        split: str = "both",
        max_train: Optional[int] = None,
        max_test: Optional[int] = None
    ) -> Tuple[List[DataSample], List[DataSample]]:
        """
        Load a dataset by name.
        
        Args:
            dataset_name: Name of dataset ('copa', 'sst2', 'boolq', etc.).
            split: Which split to load ('train', 'test', 'both').
            max_train: Maximum training samples (None for all).
            max_test: Maximum test samples (None for all).
        
        Returns:
            Tuple of (train_data, test_data).
        """
        loaders = {
            "copa": self.load_copa,
            "sst2": self.load_sst2,
            "sst5": self.load_sst5,
            "boolq": self.load_boolq,
            "mmlu": self.load_mmlu,
            "xnli": self.load_xnli,
            "winogrande": self.load_winogrande,
            "gsm8k": self.load_gsm8k,
            "math500": self.load_math500,
            "truthfulqa": self.load_truthfulqa,
            "ifeval": self.load_ifeval,
            "spider": self.load_spider,
        }
        
        name = dataset_name.lower()
        if name not in loaders:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(loaders.keys())}")
        
        train_data, test_data = loaders[name]()
        
        # Apply sample limits
        if max_train and len(train_data) > max_train:
            train_data = train_data[:max_train]
        if max_test and len(test_data) > max_test:
            test_data = test_data[:max_test]
        
        # Only print on rank 0 in distributed mode
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            print(f"Loaded {dataset_name}: {len(train_data)} train, {len(test_data)} test samples", flush=True)
        
        return train_data, test_data
    
    def load_copa(self) -> Tuple[List[DataSample], List[DataSample]]:
        """Load COPA (Choice of Plausible Alternatives) dataset."""
        train_data, test_data = [], []

        # Load training data
        train_path = self.data_root / "xcopa" / "train.csv"
        if train_path.exists():
            with open(train_path, "r", encoding="utf-8") as f:
                lines = f.readlines()[1:1501]  # Skip header, limit samples

            for line in lines:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue
                label, premise, question = parts[0].strip(), parts[2].strip(), parts[3].strip()
                choice1, choice2 = parts[4].strip(), parts[5].strip()

                prompt = (
                    f"Question:\n{premise} Based on the previous passage, "
                    f"choose the most reasonable {question}.\n"
                    f"A:{choice1}\nB:{choice2}\n\nAnswer:\n"
                )

                if int(label) == 0:
                    train_data.append((prompt, "A", "B"))
                else:
                    train_data.append((prompt, "B", "A"))

        # Load test data
        test_path = self.data_root / "xcopa" / "test.en.jsonl"
        if test_path.exists():
            with open(test_path, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    premise = item['premise']
                    question = item['question']
                    choice1, choice2 = item['choice1'], item['choice2']
                    label = item['label']

                    prompt = (
                        f"Question:\n{premise} Based on the previous passage"
                        f"choose the most reasonable {question}.\n"
                        f"A:{choice1}\nB:{choice2}\n\nAnswer:\n"
                    )

                    if int(label) == 0:
                        test_data.append((prompt, "A", "B"))
                    else:
                        test_data.append((prompt, "B", "A"))

        return train_data, test_data

    def load_sst2(self) -> Tuple[List[DataSample], List[DataSample]]:
        """Load SST-2 (Stanford Sentiment Treebank - Binary) dataset."""
        train_data, test_data = [], []
        
        def process_item(item: Dict) -> DataSample:
            text = item['text'].strip()
            label = item['label_text'].strip()
            
            prompt = (
                f"Consider the sentiment expression in this sentence and respond briefly "
                f"with 'positive' or 'negative'.\n\n{text}\n\nAnswer:"
            )
            
            if label == 'positive':
                return (prompt, "Positive", "Negative")
            else:
                return (prompt, "Negative", "Positive")
        
        # Load training data
        train_path = self.data_root / "SST" / "sst2" / "train.jsonl"
        if train_path.exists():
            with open(train_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= 2000:  # Limit training samples
                        break
                    train_data.append(process_item(json.loads(line)))
        
        # Load test data
        test_path = self.data_root / "SST" / "sst2" / "test.jsonl"
        if test_path.exists():
            with open(test_path, "r", encoding="utf-8") as f:
                for line in f:
                    test_data.append(process_item(json.loads(line)))
        
        return train_data, test_data
    
    def load_sst5(self) -> Tuple[List[DataSample], List[DataSample]]:
        """Load SST-5 (5-class sentiment) dataset."""
        train_data, test_data = [], []
        
        opposite_map = {
            'positive': 'negative',
            'very positive': 'very negative',
            'negative': 'positive',
            'very negative': 'very positive',
            'neutral': None  # Random choice
        }
        
        # Check if model is Qwen to determine prompt format
        is_qwen = 'qwen' in self.model_name.lower()
        def process_item(item: Dict) -> DataSample:
            text = item['text'].strip()
            label = item['label_text'].strip()
            
            if is_qwen:
                # Use boxed format for Qwen models
                prompt = (
                    "You are given a sentence.\n"
                    "Determine the sentiment of the sentence.\n\n"
                    "Rules:\n"
                    "1. Choose exactly one label from: very positive, positive, neutral, negative, very negative.\n"
                    "2. Do not add any extra text in the final answer.\n"
                    "3. Put the final answer inside \\boxed{{}}.\n\n"
                    f"Sentence: {text}\n\n"
                    "Final Answer:"
                )
            else:
                # Use simple format for other models
                prompt = (
                    f"Question:\nDetermine the sentiment of the following sentence. "
                    f"Choose one label from: very positive, positive, neutral, negative, very negative.\n"
                    f"Sentence: {text}\n\nAnswer:\n"
                )
            
            wrong = opposite_map.get(label)
            if wrong is None:
                wrong = random.choice(['positive', 'negative'])
            
            return (prompt, label, wrong)
        
        # Load training data
        train_path = self.data_root / "SST" / "sst5" / "train.jsonl"
        if train_path.exists():
            with open(train_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= 2000:
                        break
                    train_data.append(process_item(json.loads(line)))
        
        # Load test data
        test_path = self.data_root / "SST" / "sst5" / "test.jsonl"
        if test_path.exists():
            with open(test_path, "r", encoding="utf-8") as f:
                for line in f:
                    test_data.append(process_item(json.loads(line)))
        
        return train_data, test_data
    
    def load_boolq(self) -> Tuple[List[DataSample], List[DataSample]]:
        """Load BoolQ (Boolean Questions) dataset."""
        train_data, test_data = [], []
        
        def process_item(item: Dict) -> DataSample:
            question = item['question'].strip()
            passage = item['passage'].strip()
            answer = str(item['answer']).strip()
            
            prompt = (
                f"Is the answer to the question encapsulated in the passage? "
                f"Please confirm with 'yes' or 'no'.\n\n"
                f"Passage: {passage}\n\nQuestion: {question}\n\nAnswer:"
            )
            
            if answer == 'True':
                return (prompt, "Yes", "No")
            else:
                return (prompt, "No", "Yes")
        
        # Load training data
        train_path = self.data_root / "boolq" / "train.jsonl"
        if train_path.exists():
            with open(train_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= 2000:
                        break
                    train_data.append(process_item(json.loads(line)))
        
        # Load test data (dev set)
        test_path = self.data_root / "boolq" / "dev.jsonl"
        if test_path.exists():
            with open(test_path, "r", encoding="utf-8") as f:
                for line in f:
                    test_data.append(process_item(json.loads(line)))
        
        return train_data, test_data
    
    def load_mmlu(self, subject: str = "all") -> Tuple[List[DataSample], List[DataSample]]:
        """Load MMLU (Massive Multitask Language Understanding) dataset."""
        all_data = []
        
        mmlu_dir = self.data_root / "mmlu" / "test"
        if not mmlu_dir.exists():
            return [], []
        
        def process_row(row: List[str]) -> Optional[DataSample]:
            if len(row) < 6:
                return None
            
            question = row[0]
            options = row[1:5]
            correct = row[5]
            
            prompt = (
                f"Question: {question}\n"
                f"Which of the following answers is correct?\n"
                f"A. {options[0]}\nB. {options[1]}\nC. {options[2]}\nD. {options[3]}\n"
                f"State the letter corresponding to the correct answer.\nAnswer:"
            )
            
            wrong_choices = ['A', 'B', 'C', 'D']
            if correct in wrong_choices:
                wrong_choices.remove(correct)
            wrong = random.choice(wrong_choices)
            
            return (prompt, correct, wrong)
        
        # Load from all subject files or specific subject
        for csv_file in sorted(mmlu_dir.glob("*.csv")):
            if subject != "all" and subject.lower() not in csv_file.stem.lower():
                continue
            
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i >= 500:  # Limit per subject
                        break
                    sample = process_row(row)
                    if sample:
                        all_data.append(sample)
        
        # Split into train and test to avoid direct overlap
        if len(all_data) > 200:
            random.seed(42)
            random.shuffle(all_data)
            split_idx = min(len(all_data) // 2, 1000)
            train_data = all_data[:split_idx]
            test_data = all_data[split_idx:]
        else:
            train_data = all_data
            test_data = all_data.copy()
            
        return train_data, test_data
    
    def load_xnli(self) -> Tuple[List[DataSample], List[DataSample]]:
        """Load XNLI (Cross-lingual NLI) dataset - English subset."""
        train_data, test_data = [], []
        
        label_map = {
            'entailment': ('True', ['False', 'Inconclusive']),
            'contradiction': ('False', ['True', 'Inconclusive']),
            'neutral': ('Inconclusive', ['True', 'False'])
        }
        
        def process_line(parts: List[str]) -> Optional[DataSample]:
            if len(parts) < 8:
                return None
            
            lang, label = parts[0], parts[1]
            premise, hypothesis = parts[6], parts[7]
            
            if lang != 'en':
                return None
            if 'qwen' in self.model_name.lower():
                prompt = (
                    "You are given a premise and a hypothesis.\n"
                    "Determine whether the hypothesis is **more likely to be true, "
                    "false, or inconclusive** based only on the information in the "
                    "premise.\n\n"
                    "Rules:\n"
                    "1. Choose exactly one label from: True, False, Inconclusive.\n"
                    "2. Do not add any extra text in the final answer.\n"
                    "3. Put the final answer inside \\boxed{{}}.\n\n"
                    f"Premise: {premise}\n"
                    f"Hypothesis: {hypothesis}\n\n"
                    "Final Answer:"
                )
            else:
                prompt = (
                    f"Question:\nDetermine whether the hypothesis is **more likely to be true, "
                    f"false, or inconclusive** based only on the information in the "
                    f"premise.\n\n"
                    f"Premise: {premise}\n"
                    f"Hypothesis: {hypothesis}\n\n"
                    f"Answer:"
                )
            
            if label in label_map:
                correct, wrong_options = label_map[label]
                wrong = random.choice(wrong_options)
                return (prompt, correct, wrong)
            return None
        
        # Load dev data as train
        dev_path = self.data_root / "mnli" / "xnli.dev.tsv"
        if dev_path.exists():
            with open(dev_path, "r", encoding="utf-8") as f:
                for line in f.readlines()[1:]:  # Skip header
                    sample = process_line(line.split('\t'))
                    if sample:
                        train_data.append(sample)
        
        # Load test data
        test_path = self.data_root / "mnli" / "xnli.test.tsv"
        if test_path.exists():
            with open(test_path, "r", encoding="utf-8") as f:
                for line in f.readlines()[1:]:
                    sample = process_line(line.split('\t'))
                    if sample:
                        test_data.append(sample)
        
        return train_data, test_data
    
    def load_winogrande(self) -> Tuple[List[DataSample], List[DataSample]]:
        """Load WinoGrande (commonsense reasoning) dataset."""
        train_data, test_data = [], []
        
        def process_item(item: Dict) -> DataSample:
            sentence = item['sentence'].strip()
            option1 = item['option1'].strip()
            option2 = item['option2'].strip()
            answer = item['answer'].strip()
            
            prompt = (
                f"Please fill in the blanks. Write A or B as the answer.\n\n"
                f"Sentence: {sentence}\nA. {option1}\nB. {option2}\nAnswer:"
            )
            
            if answer == '1':
                return (prompt, "A", "B")
            else:
                return (prompt, "B", "A")
        
        # Load training data
        train_path = self.data_root / "winogrande" / "train_m.jsonl"
        if train_path.exists():
            with open(train_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= 2000:
                        break
                    train_data.append(process_item(json.loads(line)))
        
        # Load test data (using dev split)
        test_path = self.data_root / "winogrande" / "dev.jsonl"
        if test_path.exists():
            with open(test_path, "r", encoding="utf-8") as f:
                for line in f:
                    test_data.append(process_item(json.loads(line)))
        
        return train_data, test_data

    def load_gsm8k(self) -> Tuple[List[DataSample], List[DataSample]]:
        """Load GSM8K (Grade School Math 8K) dataset."""
        train_data, test_data = [], []
        
        def process_item(item: Dict) -> DataSample:
            question = item['question'].strip()
            answer_text = item['answer'].strip()

            # Extract the final numeric answer after ####
            correct_answer = answer_text.split("####")[-1].strip()

            prompt = (
                "You are given a math problem.\n"
                "Solve the problem step by step and provide the final numeric answer.\n\n"
                "Rules:\n"
                "1. The final answer should be a number only.\n"
                "2. Do not add any extra text in the final answer.\n"
                "3. Put the final answer inside \\boxed{{}}.\n\n"
                f"Question: {question}\n\n"
            )

            # Generate a simple wrong answer by adding a random offset
            try:
                val = int(correct_answer.replace(',', ''))
                wrong_val = val + random.randint(1, 100)
                wrong_answer = str(wrong_val)
                wrong_answer = "\\boxed{" + wrong_answer + "}"
            except ValueError:
                # Fallback if answer is not a simple integer
                wrong_answer = "0"

            return (prompt, correct_answer, wrong_answer)

        # Load training data
        train_path = self.data_root / "gsm8k" / "train.jsonl"
        if train_path.exists():
            with open(train_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= 2000:  # Limit training samples
                        break
                    train_data.append(process_item(json.loads(line)))

        # Load test data
        test_path = self.data_root / "gsm8k" / "test.jsonl"
        if test_path.exists():
            with open(test_path, "r", encoding="utf-8") as f:
                for line in f:
                    test_data.append(process_item(json.loads(line)))

        return train_data, test_data

    def load_math500(self) -> Tuple[List[DataSample], List[DataSample]]:
        """
        Load MATH-500 dataset.
        
        MATH-500 is a subset of 500 problems selected by OpenAI from the MATH dataset.
        It covers various mathematical topics including algebra, geometry, number theory,
        counting and probability, intermediate algebra, precalculus, and prealgebra.
        """
        train_data, test_data = [], []

        def process_item(item: Dict) -> DataSample:
            question = item['question'].strip()
            correct_answer = item['answer'].strip()

            prompt = (
                "You are given a math problem.\n"
                "Solve the problem step by step and provide the final answer.\n\n"
                "Rules:\n"
                "1. Show your reasoning process clearly.\n"
                "2. Put the final answer inside \\boxed{{}}.\n\n"
                f"Question: {question}\n\n"
            )

            # Generate a simple wrong answer for contrastive learning
            # For MATH-500, answers can be complex (fractions, expressions, etc.)
            # We use a placeholder wrong answer
            wrong_answer = "0"
            wrong_answer = "\\boxed{" + wrong_answer + "}"

            return (prompt, correct_answer, wrong_answer)

        # MATH-500 only has test set (500 problems)
        # We use a portion for training and the rest for testing
        test_path = self.data_root / "math500" / "test.jsonl"
        if test_path.exists():
            all_data = []
            with open(test_path, "r", encoding="utf-8") as f:
                for line in f:
                    all_data.append(process_item(json.loads(line)))
            
            # Split: first 100 samples for training, rest for testing
            if len(all_data) > 100:
                train_data = all_data[:100]
                test_data = all_data[100:]
            else:
                train_data = all_data
                test_data = []

        return train_data, test_data
    
    def load_truthfulqa(self) -> Tuple[List[DataSample], List[DataSample]]:
        """
        Load TruthfulQA dataset in Multiple-choice format.
        
        TruthfulQA is a benchmark to measure whether a language model is truthful
        in generating answers to questions. The MC (Multiple-choice) format provides
        questions with multiple answer options, where only one is correct.
        """
        train_data, test_data = [], []
        
        mc_task_path = self.data_root / "truthfulqa" / "mc_task.json"
        
        if not mc_task_path.exists():
            print(f"Warning: TruthfulQA dataset not found at {mc_task_path}")
            return [], []
        
        def process_item(item: Dict) -> Optional[DataSample]:
            """
            Process a TruthfulQA MC item into standard format.
            
            Format: {
                "Question": "...",
                "Correct Answers": ["..."],
                "Incorrect Answers": ["...", "...", ...]
            }
            """
            question = item.get("Question", "").strip()
            correct_answers = item.get("Correct Answers", [])
            incorrect_answers = item.get("Incorrect Answers", [])
            
            if not question or not correct_answers or not incorrect_answers:
                return None
            
            # Use the first correct answer
            correct_answer = correct_answers[0].strip()
            
            # Select 3 incorrect answers to form 4 options (A, B, C, D)
            # If there are fewer than 3 incorrect answers, pad with duplicates
            selected_incorrect = incorrect_answers[:3] if len(incorrect_answers) >= 3 else incorrect_answers
            
            # Ensure we have exactly 4 options
            # If we have fewer than 3 incorrect answers, pad by repeating
            while len(selected_incorrect) < 3:
                selected_incorrect.append(selected_incorrect[-1] if selected_incorrect else correct_answer)
            
            # Combine correct and incorrect answers, then shuffle
            all_options = [correct_answer] + selected_incorrect
            random.shuffle(all_options)
            
            # Find the index of correct answer after shuffling
            correct_idx = all_options.index(correct_answer)
            correct_letter = ['A', 'B', 'C', 'D'][correct_idx]
            
            # Select a wrong answer letter for contrast
            wrong_choices = ['A', 'B', 'C', 'D']
            wrong_choices.remove(correct_letter)
            wrong_letter = random.choice(wrong_choices)
            if 'qwen' in self.model_name.lower():
                prompt = (
                    f"Question: {question}\n"
                    f"Which of the following answers is correct?\n"
                    f"A. {all_options[0]}\nB. {all_options[1]}\n"
                    f"C. {all_options[2]}\nD. {all_options[3]}\n"
                    f"Do not add any extra text in the final answer.\n"
                    f"Put the final answer inside \\boxed{{}}.\n give your answer in the beginning of the your sentence.\n"
                    f"Answer:"
                )
            else:
                # Build prompt similar to MMLU format
                prompt = (
                    f"Question: {question}\n"
                    f"Which of the following answers is correct?\n"
                    f"A. {all_options[0]}\nB. {all_options[1]}\n"
                    f"C. {all_options[2]}\nD. {all_options[3]}\n"
                    f"State the letter corresponding to the correct answer.\nAnswer:"
                )
            
            return (prompt, correct_letter, wrong_letter)
        
        # Load data from JSON file
        with open(mc_task_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Process all items
        all_data = []
        for item in data:
            sample = process_item(item)
            if sample:
                all_data.append(sample)
        
        # Split into train and test sets: first 100 samples for training, rest for testing
        if len(all_data) > 100:
            train_data = all_data[:100]
            test_data = all_data[100:]
        else:
            # If dataset has 100 or fewer samples, use all for training
            train_data = all_data
            test_data = []

        return train_data, test_data

    def load_ifeval(self) -> Tuple[List[DataSample], List[DataSample]]:
        """
        Load IFEval (Instruction-Following Evaluation) dataset.
        
        IFEval is a benchmark for evaluating instruction-following capabilities
        of language models. It contains prompts with verifiable instructions
        that can be automatically checked (e.g., "write at least 300 words",
        "do not use commas", etc.).
        
        The dataset has 541 samples with 25 types of verifiable instructions.
        
        Note: IFEval requires special evaluation logic - the correct answer is
        not a simple string but rather a set of verifiable constraints.
        For compatibility with the standard (prompt, correct, wrong) format,
        we encode the instruction metadata as a JSON string in the correct_answer field.
        """
        train_data, test_data = [], []
        
        ifeval_path = self.data_root / "ifeval" / "test.jsonl"
        
        if not ifeval_path.exists():
            print(f"Warning: IFEval dataset not found at {ifeval_path}")
            return [], []
        
        def process_item(item: Dict) -> Optional[DataSample]:
            """
            Process an IFEval item into standard format.
            
            Format: {
                "key": int,
                "prompt": str,
                "instruction_id_list": List[str],
                "kwargs": List[Dict]
            }
            """
            prompt = item.get("prompt", "").strip()
            instruction_id_list = item.get("instruction_id_list", [])
            kwargs = item.get("kwargs", [])
            key = item.get("key", 0)
            
            if not prompt or not instruction_id_list:
                return None
            
            # Encode the instruction metadata as JSON for evaluation
            # The evaluation code will decode this to verify the response
            metadata = {
                "key": key,
                "instruction_id_list": instruction_id_list,
                "kwargs": kwargs,
                "task_type": "ifeval"  # Mark this as IFEval task for special handling
            }
            correct_answer = json.dumps(metadata, ensure_ascii=False)
            
            # For wrong answer, we use a placeholder
            # (IFEval doesn't use contrastive pairs in the same way)
            wrong_answer = "INVALID_RESPONSE"
            
            return (prompt, correct_answer, wrong_answer)
        
        # Load data from JSONL file
        all_data = []
        with open(ifeval_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    sample = process_item(item)
                    if sample:
                        all_data.append(sample)
                except json.JSONDecodeError:
                    continue
        
        # IFEval is typically used as a test-only benchmark
        # We split into train/test for compatibility with the framework
        # Use first 100 samples for training (steering vector construction)
        # and the rest for testing
        if len(all_data) > 100:
            train_data = all_data[:100]
            test_data = all_data[100:]
        else:
            train_data = all_data
            test_data = all_data.copy()
        
        return train_data, test_data

    # ==================== Spider SQL Evaluation Methods ====================

    def extract_dbid(self, question: str) -> str:
        """Extract database ID from question."""
        if "|" in question:
            return question.split("|")[0]
        return question

    def get_database_overview(self, db_id: str) -> str:
        """
        Generate formatted overview of database structure

        Args:
            db_id: Database identifier

        Returns:
            Formatted string containing database schema information
        """
        if db_id not in self.spider_tables:
            return f"Database {db_id} not found in tables data"

        db_info = self.spider_tables[db_id]
        overview_parts = []

        # Database overview section
        db_overview = db_info.get('db_overview', 'No overview available')
        overview_parts.append(f"Database Overview: {db_overview}")
        overview_parts.append("")

        # Table listing
        overview_parts.append("Tables:")
        table_names = db_info.get('table_names_original', [])
        for table_name in table_names:
            overview_parts.append(f"- {table_name}")
        overview_parts.append("")

        # Column details
        overview_parts.append("Columns:")
        column_names_original = db_info.get('column_names_original', [])
        column_descriptions = db_info.get('column_descriptions', [])
        column_types = db_info.get('column_types', [])

        # Process each column
        for i, column_info in enumerate(column_names_original):
            if i == 0:  # Skip index column
                continue

            table_idx, column_name = column_info
            if table_idx < 0 or table_idx >= len(table_names):
                continue

            table_name = table_names[table_idx]

            # Build column description
            description = ""
            if i < len(column_descriptions) and column_descriptions[i]:
                description = column_descriptions[i]
            elif i < len(column_types) and column_types[i]:
                description = f"{column_types[i]} type"
            else:
                description = "No description available"

            overview_parts.append(f"- {table_name}.{column_name}: {description}")

        return "\n".join(overview_parts)

    def permute_tuple(self, element: Tuple, perm: Tuple) -> Tuple:
        """Reorder tuple elements based on permutation indices"""
        assert len(element) == len(perm)
        return tuple([element[i] for i in perm])

    def unorder_row(self, row: Tuple) -> Tuple:
        """Create sorted version of row for comparison"""
        return tuple(sorted(row, key=lambda x: str(x) + str(type(x))))

    def quick_rej(self, result1: List[Tuple], result2: List[Tuple], order_matters: bool) -> bool:
        """Quick comparison using unordered rows"""
        s1 = [self.unorder_row(row) for row in result1]
        s2 = [self.unorder_row(row) for row in result2]
        if order_matters:
            return s1 == s2
        else:
            return set(s1) == set(s2)

    def multiset_eq(self, l1: List, l2: List) -> bool:
        """Check if two lists contain same elements with same frequencies"""
        if len(l1) != len(l2):
            return False
        d = defaultdict(int)
        for e in l1:
            d[e] = d[e] + 1
        for e in l2:
            d[e] = d[e] - 1
            if d[e] < 0:
                return False
        return True

    def get_constraint_permutation(self, tab1_sets_by_columns: List[Set], result2: List[Tuple]):
        """Generate column permutations considering value constraints"""
        num_cols = len(result2[0])
        perm_constraints = [{i for i in range(num_cols)} for _ in range(num_cols)]
        if num_cols <= 3:
            return itertools.product(*perm_constraints)

        # Use random sampling to reduce permutation space
        for _ in range(min(20, len(result2))):
            random_tab2_row = random.choice(result2)
            for tab1_col in range(num_cols):
                for tab2_col in set(perm_constraints[tab1_col]):
                    if random_tab2_row[tab2_col] not in tab1_sets_by_columns[tab1_col]:
                        perm_constraints[tab1_col].remove(tab2_col)
        return itertools.product(*perm_constraints)

    def result_eq(self, result1: List[Tuple], result2: List[Tuple], order_matters: bool) -> bool:
        """Compare query results with column permutation support"""
        if len(result1) == 0 and len(result2) == 0:
            return True

        if len(result1) != len(result2):
            return False

        num_cols = len(result1[0])
        if len(result2[0]) != num_cols:
            return False

        # Initial quick comparison
        if not self.quick_rej(result1, result2, order_matters):
            return False

        if result1 == result2:
            return True

        # Skip large result sets for performance
        if len(result2) > 200 or len(result1) > 200:
            return False

        tab1_sets_by_columns = [{row[i] for row in result1} for i in range(num_cols)]

        # Test different column orderings
        for perm in self.get_constraint_permutation(tab1_sets_by_columns, result2):
            if len(perm) != len(set(perm)):
                continue
            if num_cols == 1:
                result2_perm = result2
            else:
                result2_perm = [self.permute_tuple(element, perm) for element in result2]

            if order_matters:
                if result1 == result2_perm:
                    return True
            else:
                if set(result1) == set(result2_perm) and self.multiset_eq(result1, result2_perm):
                    return True
        return False

    def postprocess_query(self, query: str) -> str:
        """Fix common SQL formatting issues"""
        query = query.replace('> =', '>=').replace('< =', '<=').replace('! =', '!=')
        return query

    def remove_distinct(self, s: str) -> str:
        """Remove DISTINCT keyword from SQL query"""
        return re.sub(r'\bDISTINCT\b', '', s, flags=re.IGNORECASE)

    def replace_cur_year(self, query: str) -> str:
        """Replace dynamic year functions with static value"""
        return re.sub(r"YEAR\s*\(\s*CURDATE\s*\(\s*\)\s*\)\s*", "2020", query, flags=re.IGNORECASE)

    # def get_cursor_from_path(self, sqlite_path: str):
    #     """Create database connection cursor"""
    #     try:
    #         if not os.path.exists(sqlite_path):
    #             print(f"Opening a new connection: {sqlite_path}")
    #         connection = sqlite3.connect(sqlite_path)
    #         connection.text_factory = lambda b: b.decode(errors="ignore")
    #         return connection.cursor()
    #     except Exception as e:
    #         print(f"Error connecting to {sqlite_path}: {e}")
    #         raise e

    def exec_on_db_sync(self, sqlite_path: str, query: str) -> Tuple[str, Any]:
        """Execute SQL query and return results"""
        query = self.replace_cur_year(query)
        cursor = self.get_cursor_from_path(sqlite_path)
        try:
            cursor.execute(query)
            result = cursor.fetchall()
            cursor.connection.close()
            return "result", result
        except Exception as e:
            cursor.connection.close()
            return "exception", e

    def extract_sql_from_answer(self, answer: str) -> str:
        """Extract SQL query from text response"""
        answer = answer.strip()

        # Check for SQL code blocks
        code_block = re.search(r"```sql(.*?)```", answer, re.DOTALL | re.IGNORECASE)
        if code_block:
            sql = code_block.group(1).strip()
            return self._extract_complete_sql(sql)

        # Find SQL keywords in text
        sql_keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP", "WITH"]
        positions = [(kw, answer.upper().find(kw)) for kw in sql_keywords]
        positions = [(kw, pos) for kw, pos in positions if pos != -1]

        if not positions:
            return answer

        _, start = min(positions, key=lambda x: x[1])
        sql_text = answer[start:].strip()

        return self._extract_complete_sql(sql_text)

    def _extract_complete_sql(self, sql: str) -> str:
        """Extract complete SQL statement handling quotes and semicolons"""
        result = []
        in_single_quote = False
        in_double_quote = False

        for ch in sql:
            result.append(ch)

            if ch == "'" and not in_double_quote:
                in_single_quote = not in_single_quote
            elif ch == '"' and not in_single_quote:
                in_double_quote = not in_double_quote
            elif ch == ";" and not in_single_quote and not in_double_quote:
                break

        cleaned = "".join(result).strip()

        # Close any open quotes
        if in_single_quote:
            cleaned += "'"
        if in_double_quote:
            cleaned += '"'

        # Remove trailing semicolon
        if cleaned.endswith(";"):
            cleaned = cleaned[:-1].strip()

        return cleaned

    def get_database_paths(self, db_id: str) -> List[str]:
        """Find all database files for given database ID"""
        # Check primary database path
        db_path = os.path.join(str(self.spider_databases_dir), f"{db_id}/{db_id}.sqlite")
        if os.path.exists(db_path):
            db_dir = os.path.dirname(db_path)
        else:
            # Check alternative path
            db_path = os.path.join(str(self.spider_databases_dir), f"{db_id}.sqlite")
            if os.path.exists(db_path):
                db_dir = os.path.dirname(db_path)
            else:
                return []

        # Find all SQLite files in directory
        db_paths = []
        for item in os.listdir(db_dir):
            if item.endswith('.sqlite'):
                db_paths.append(os.path.join(db_dir, item))

        return db_paths if db_paths else [db_path]

    def evaluate_execution(self, db_id: str, predicted_sql: str, gold_sql: str) -> bool:
        """
        Compare predicted SQL against gold standard by execution results

        Args:
            db_id: Target database identifier
            predicted_sql: Generated SQL query to test
            gold_sql: Reference SQL query

        Returns:
            True if queries produce equivalent results, False otherwise
        """
        # Normalize query formatting
        p_str = self.postprocess_query(predicted_sql)
        g_str = self.postprocess_query(gold_sql)

        # Remove DISTINCT keyword
        p_str = self.remove_distinct(p_str)
        g_str = self.remove_distinct(g_str)

        # Determine if row order affects correctness
        order_matters = 'order by' in g_str.lower()

        # Locate database files
        db_paths = self.get_database_paths(db_id)
        if not db_paths:
            print(f"No databases found for {db_id}")
            return False

        # Execute and compare on all database instances
        for db_path in db_paths:
            # Execute reference query
            g_flag, g_denotation = self.exec_on_db_sync(db_path, g_str)
            if g_flag == 'exception':
                print(f"Reference query failed on {db_path}: {g_denotation}")
                return False

            # Execute predicted query
            p_flag, p_denotation = self.exec_on_db_sync(db_path, p_str)
            if p_flag == 'exception':
                return False

            # Compare execution results
            if not self.result_eq(g_denotation, p_denotation, order_matters):
                return False

        return True

    # ==================== Spider Prompt Processing Methods ====================

    def sql_prompt_postprocessor(self, prompt: str) -> str:
        """
        Add database schema information to SQL prompts.

        Args:
            prompt: Original prompt (format: "dbid|question")

        Returns:
            Prompt with database schema information prepended
        """
        # Extract database ID from prompt (format: "dbid|question")
        db_id = self.extract_dbid(prompt)
        # Get database overview
        db_overview = self.get_database_overview(db_id)
        # Prepend database schema information to the prompt
        return (f"{db_overview}\n\nQuestion: {prompt},given the database "
                f"overview, you should generate the SQL query to answer the question.")

    def create_sql_reward_fn(self):
        """
        Create a SQL reward function for the current instance.

        Returns:
            Reward function with signature (question, answer1, answer2, standard_answer) -> List[float]
        """
        def sql_reward_fn(question: str, answer1: str, answer2: str, standard_answer: str) -> List[float]:
            """
            Evaluate SQL generation quality by comparing execution results.

            Args:
                question: The question containing database ID
                answer1: First SQL answer to compare
                answer2: Second SQL answer to compare
                standard_answer: Standard answer for evaluation

            Returns:
                [1.0, 0.0] if answer1 correct and answer2 wrong
                [0.0, 1.0] if answer2 correct and answer1 wrong
                [0.5, 0.5] if both correct or both wrong
            """
            sql1 = self.extract_sql_from_answer(answer1)
            sql2 = self.extract_sql_from_answer(answer2)

            db_id = self.extract_dbid(question)
            result1 = self.evaluate_execution(db_id, sql1, standard_answer)
            result2 = self.evaluate_execution(db_id, sql2, standard_answer)

            # Return reward based on correctness: [reward1, reward2]
            if result1 and not result2:
                return [1.0, 0.0]
            elif not result1 and result2:
                return [0.0, 1.0]
            else:
                return [0.5, 0.5]

        return sql_reward_fn

    def load_spider(self) -> Tuple[List[DataSample], List[DataSample]]:
        """
        Load Spider (SQL generation) dataset.

        Spider is a large-scale complex and cross-domain semantic parsing
        and text-to-SQL dataset. This method loads the dataset and formats
        it for chat mode evaluation.
        """
        train_data, test_data = [], []

        def process_item(item: Dict) -> Optional[DataSample]:
            """
            Process a Spider item into standard format.

            Args:
                item: Dictionary with 'question' and 'answer' keys

            Returns:
                Tuple of (prompt, correct_answer, wrong_answer) or None if invalid
            """
            question = item.get('question', '').strip()
            correct_sql = item.get('answer', '').strip()

            if not question or not correct_sql:
                return None

            # Generate prompt with database schema information
            try:
                prompt = self.sql_prompt_postprocessor(question)
            except Exception as e:
                print(f"Warning: Failed to generate prompt for question '{question}': {e}")
                return None

            # For wrong answer, we use a simple incorrect SQL query
            # This is a basic approach - in practice, you might want more sophisticated wrong answers
            wrong_sql = "SELECT * FROM non_existent_table"

            return (prompt, correct_sql, wrong_sql)

        # Load training data
        train_path = self.spider_dir / "train_spider.jsonl"
        if train_path.exists():
            with open(train_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= 2000:  # Limit training samples for efficiency
                        break
                    try:
                        item = json.loads(line)
                        sample = process_item(item)
                        if sample:
                            train_data.append(sample)
                    except json.JSONDecodeError:
                        continue

        # Load test data (dev set)
        test_path = self.spider_dir / "dev.jsonl"
        if test_path.exists():
            with open(test_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        sample = process_item(item)
                        if sample:
                            test_data.append(sample)
                    except json.JSONDecodeError:
                        continue

        return train_data, test_data
