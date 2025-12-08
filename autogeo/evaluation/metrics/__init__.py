"""
Evaluation metrics module for AutoGEO.

Provides metrics for evaluating rewritten documents:
- GEO score: Visibility metrics (position, token count, citation frequency)
- GEU score: Utility metrics (citation quality, keypoint coverage, response quality)
"""
from .geo_score import (
    get_num_words,
    extract_citations_new,
    impression_wordpos_count_simple,
    impression_word_count_simple,
    impression_pos_count_simple,
)
from .geu_score import (
    preprocess_data_for_evaluation,
    calculate_citation_quality,
    calculate_quality_dimensions,
    calculate_keypoint_coverage,
    process_single_question,
    evaluate_ge_utility,
    geu_score,
)

__all__ = [
    # GEO score functions
    'get_num_words',
    'extract_citations_new',
    'impression_wordpos_count_simple',
    'impression_word_count_simple',
    'impression_pos_count_simple',
    # GEU score functions
    'preprocess_data_for_evaluation',
    'calculate_citation_quality',
    'calculate_quality_dimensions',
    'calculate_keypoint_coverage',
    'process_single_question',
    'evaluate_ge_utility',
    'geu_score',
]


