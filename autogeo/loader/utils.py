"""Utility functions for loader module."""
from ..config import Dataset


def parse_rewrite_method(rewrite_method_name: str) -> tuple:
    """Parse rewrite method name to extract dataset name and LLM name.
    
    Args:
        rewrite_method_name: Method name in format 'autogeo_{dataset}_{engine_llm}'
        
    Returns:
        Tuple of (dataset, engine_llm)
        
    Raises:
        ValueError: If dataset name cannot be parsed or is not supported
    """
    method_name = rewrite_method_name.lower().replace("autogeo_", "").replace("_mine", "")
    parts = method_name.split("_")
    
    if len(parts) < 2:
        raise ValueError(
            f"Invalid rewrite method name format: {rewrite_method_name}. "
            f"Expected format: 'autogeo_{{dataset}}_{{engine_llm}}'"
        )
    
    dataset_part = "_".join(parts[:-1])  # All parts except the last (LLM name)
    engine_llm = parts[-1]
    
    # Normalize dataset part: replace underscores with hyphens
    dataset_normalized = dataset_part.replace("_", "-")
    
    # Validate dataset name
    try:
        dataset_enum = Dataset(dataset_normalized)
        dataset = dataset_enum.value
    except ValueError:
        raise ValueError(
            f"Unsupported dataset in method name '{rewrite_method_name}': {dataset_normalized}. "
            f"Supported datasets: {[d.value for d in Dataset]}"
        )
    
    return dataset, engine_llm

