"""Project evaluation utilities (prompt assembly, answer extraction, correctness checks)."""


def is_main_process() -> bool:
    """Always True in non-distributed batch runs."""
    return True
