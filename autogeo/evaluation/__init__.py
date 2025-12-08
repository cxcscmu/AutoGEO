"""
Evaluation module for AutoGEO.
"""
from .evaluator import autogeo_evaluation
from .aggregate_results import aggregate_json_files
from .generative_engine import generate_answer_gemini, generate_answer_gpt, generate_answer_claude

__all__ = [
    'autogeo_evaluation',
    'aggregate_json_files',
    'generate_answer_gemini',
    'generate_answer_gpt',
    'generate_answer_claude',
]

