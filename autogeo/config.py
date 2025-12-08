"""
Configuration module for AutoGEO datasets and models.
"""
from typing import Dict
from enum import Enum


class Dataset(str, Enum):
    """Supported datasets."""
    RESEARCHY_GEO = "Researchy-GEO"
    ECOMMERCE = "E-commerce"
    GEO_BENCH = "GEO-Bench"


class LLMName(str, Enum):
    """Supported LLM names."""
    GEMINI = "gemini"
    GPT = "gpt"
    CLAUDE = "claude"


# Dataset configuration mapping
DATASET_CONFIGS: Dict[str, Dict[str, str]] = {
    Dataset.RESEARCHY_GEO: {
        "hf_dataset_name": "Researchy-GEO",
        "data_dir": "data/Researchy-GEO",
        "train_dir": "data/Researchy-GEO/train",
        "test_dir": "data/Researchy-GEO/test",
        "rl_dir": "data/Researchy-GEO/RL",
        "rule_candidate_file": "rule_candidate.json",
        "finetune_file": "finetune.json",
        "inference_file": "inference.json",
        "grpo_input_file": "grpo_input.json",
        "grpo_eval_file": "grpo_eval.json",
    },
    Dataset.ECOMMERCE: {
        "hf_dataset_name": "E-commerce",
        "data_dir": "data/E-commerce",
        "train_dir": "data/E-commerce/train",
        "test_dir": "data/E-commerce/test",
        "rl_dir": "data/E-commerce/RL",
        "rule_candidate_file": "rule_candidate.json",
        "finetune_file": "finetune.json",
        "inference_file": "inference.json",
        "grpo_input_file": "grpo_input.json",
        "grpo_eval_file": "grpo_eval.json",
    },
    Dataset.GEO_BENCH: {
        "hf_dataset_name": "GEO-Bench",
        "data_dir": "data/GEO-Bench",
        "train_dir": "data/GEO-Bench/train",
        "test_dir": "data/GEO-Bench/test",
        "rl_dir": "data/GEO-Bench/RL",
        "rule_candidate_file": "rule_candidate.json",
        "finetune_file": "finetune.json",
        "inference_file": "inference.json",
        "grpo_input_file": "grpo_input.json",
        "grpo_eval_file": "grpo_eval.json",
    },
}


def get_dataset_config(dataset_name: str) -> Dict[str, str]:
    """Get configuration for a specific dataset.
    
    Args:
        dataset_name: Name of the dataset (Researchy-GEO, E-commerce, or GEO-Bench)
        
    Returns:
        Dictionary containing dataset configuration
        
    Raises:
        ValueError: If dataset name is not supported
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            f"Supported datasets: {list(DATASET_CONFIGS.keys())}"
        )
    return DATASET_CONFIGS[dataset_name]


def get_full_path(dataset_name: str, path_key: str) -> str:
    """Get full path for a dataset configuration key.
    
    Args:
        dataset_name: Name of the dataset
        path_key: Configuration key (e.g., 'rule_candidate_file')
        
    Returns:
        Full path string
    """
    config = get_dataset_config(dataset_name)
    if path_key.endswith("_file"):
        return f"{config['rl_dir']}/{config[path_key]}"
    return config.get(path_key, "")


def get_rewrite_method_name(geo_method: str, dataset: str, engine_llm: str) -> str:
    """Generate standardized rewrite method name for storing results.
    
    Args:
        dataset: Name of the dataset (Researchy-GEO, E-commerce, or GEO-Bench)
        engine_llm: Generative engine LLM name (gemini, gpt, or claude)
        
    Returns:
        Standardized method name (e.g., 'autogeo_api_researchy_geo_gemini')
        
    Examples:
        >>> get_rewrite_method_name("autogeo_api", "Researchy-GEO", "gemini")
        'autogeo_api_researchy_geo_gemini'
        >>> get_rewrite_method_name("autogeo_api", "E-commerce", "gpt")
        'autogeo_api_e_commerce_gpt'
    """
    # Normalize dataset name: convert to lowercase and replace hyphens/spaces with underscores
    dataset_normalized = dataset.lower().replace("-", "_").replace(" ", "_")
    return f"{geo_method}_{dataset_normalized}_{engine_llm}"

