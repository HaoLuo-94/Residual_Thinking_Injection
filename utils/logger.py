"""Lightweight logger for base_evaluator and related modules."""

from __future__ import annotations

import logging
import sys
from typing import Optional


class TrainingLogger:
    _instance: Optional["TrainingLogger"] = None
    _initialized: bool = False

    def __new__(cls) -> "TrainingLogger":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self.logger = logging.getLogger("residual_injection.utils")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            self.logger.addHandler(handler)
        TrainingLogger._initialized = True

    def info(self, message: str, *args, **kwargs) -> None:
        self.logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs) -> None:
        self.logger.warning(message, *args, **kwargs)


def get_logger() -> TrainingLogger:
    return TrainingLogger()
