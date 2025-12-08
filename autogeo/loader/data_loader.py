import os
import json
from datasets import load_dataset, get_dataset_config_names
from ..config import Dataset, get_dataset_config, get_full_path

# Default configuration for Researchy-GEO dataset
def get_default_config() -> dict:
    """Get default configuration for Researchy-GEO dataset.
    
    Returns:
        Configuration dictionary
    """
    dataset_name = Dataset.RESEARCHY_GEO.value
    dataset_config = get_dataset_config(dataset_name)
    
    return {
        "hf_username": "yujiangw",
        "hf_dataset_name": dataset_config["hf_dataset_name"],
        "train_output_dir": dataset_config["train_dir"],
        "test_output_dir": dataset_config["test_dir"],
        "rule_candidate_output_file": get_full_path(dataset_name, 'rule_candidate_file'),
        "cold_start_output_file": get_full_path(dataset_name, 'finetune_file'),
        "inference_output_file": get_full_path(dataset_name, 'inference_file'),
        "grpo_input_output_file": get_full_path(dataset_name, 'grpo_input_file'),
        "grpo_eval_output_file": get_full_path(dataset_name, 'grpo_eval_file'),
    }

CONFIG = get_default_config()


def reconstruct_directory_split(
    dataset_split,
    output_dir: str,
    max_records_per_file: int = 100,
    skip_if_exists: bool = True
) -> bool:
    """Reconstruct dataset split into multiple JSON files in a directory.
    
    Args:
        dataset_split: Dataset split from Hugging Face datasets
        output_dir: Output directory path
        max_records_per_file: Maximum number of records per file (default: 100)
        skip_if_exists: Skip if directory exists and contains files (default: True)
        
    Returns:
        True if files were created, False if skipped
    """
    if not dataset_split or len(dataset_split) == 0:
        return False
    
    # Check if directory exists and has files
    if skip_if_exists and os.path.exists(output_dir):
        json_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
        if json_files:
            print(f"Skipping {output_dir} (already exists with {len(json_files)} file(s))")
            return False
    
    print(f"Reconstructing split into directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    record_chunks = [dataset_split[i : i + max_records_per_file] for i in range(0, len(dataset_split), max_records_per_file)]
    
    for i, chunk in enumerate(record_chunks):
        reformatted_chunk = {}
        num_records = len(chunk[list(chunk.keys())[0]])
        for idx in range(num_records):
            query_id = chunk["query_id"][idx]
            record_value = {key: chunk[key][idx] for key in chunk.keys() if key != 'query_id'}
            reformatted_chunk[query_id] = record_value
            
        output_filename = os.path.join(output_dir, f"datachunk_{i}.json")
        with open(output_filename, 'w', encoding='utf-8') as f: json.dump(reformatted_chunk, f, ensure_ascii=False, indent=4)
    print(f"-> Successfully created {len(record_chunks)} JSON file(s).")
    return True

def reconstruct_single_file(
    dataset_split,
    output_filename: str,
    original_format: str,
    extra_fields: dict = None,
    skip_if_exists: bool = True
) -> bool:
    """Reconstruct dataset split into a single JSON file.
    
    Args:
        dataset_split: Dataset split from Hugging Face datasets
        output_filename: Output file path
        original_format: Format type ("dict_of_dicts" or "list_of_dicts")
        extra_fields: Additional fields to add to each record (default: {})
        skip_if_exists: Skip if file already exists (default: True)
        
    Returns:
        True if file was created, False if skipped
    """
    if extra_fields is None:
        extra_fields = {}
    if not dataset_split or len(dataset_split) == 0:
        return False
    
    # Check if file already exists
    if skip_if_exists and os.path.exists(output_filename):
        print(f"Skipping {output_filename} (already exists)")
        return False
    
    print(f"Reconstructing split into single file: {output_filename}")
    
    reconstructed_data = None
    if original_format == "dict_of_dicts":
        reconstructed_data = {record.pop("query_id"): record for record in dataset_split}
    elif original_format == "list_of_dicts":
        reconstructed_data = [dict(record, **extra_fields) for record in dataset_split]

    if reconstructed_data is not None:
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        with open(output_filename, 'w', encoding='utf-8') as f: json.dump(reconstructed_data, f, ensure_ascii=False, indent=4)
        print(f"-> Successfully created file.")
        return True
    return False

def load_data(config: dict = None) -> None:
    """Load dataset from Hugging Face and reconstruct into local JSON files.
    
    Args:
        config: Configuration dictionary with dataset settings (default: CONFIG)
    """
    if config is None:
        config = CONFIG
    repo_id = f"{config['hf_username']}/{config['hf_dataset_name']}"

    try:
        print(f"Fetching available configurations from: https://huggingface.co/datasets/{repo_id}")
        config_names = get_dataset_config_names(repo_id)
        print("SUCCESS: Found configurations:", config_names)
    except Exception as e:
        print(f"FAILURE: Could not fetch configuration names. Is the dataset public? Error: {e}")
        return


    for config_name in config_names:
        print(f"\n--- Processing configuration: {config_name} ---")
        
        # Check if we need to download based on output files existence
        need_download = False
        if config_name == 'main':
            train_exists = False
            test_exists = False
            if os.path.exists(config['train_output_dir']) and os.path.isdir(config['train_output_dir']):
                train_exists = any(f.endswith('.json') for f in os.listdir(config['train_output_dir']))
            if os.path.exists(config['test_output_dir']) and os.path.isdir(config['test_output_dir']):
                test_exists = any(f.endswith('.json') for f in os.listdir(config['test_output_dir']))
            need_download = not (train_exists and test_exists)
        elif config_name == 'rule_candidate':
            need_download = not os.path.exists(config['rule_candidate_output_file'])
        elif config_name == 'grpo_eval':
            need_download = not os.path.exists(config['grpo_eval_output_file'])
        elif config_name == 'cold_start':
            need_download = not os.path.exists(config['cold_start_output_file'])
        elif config_name == 'grpo_input':
            need_download = not os.path.exists(config['grpo_input_output_file'])
        elif config_name == 'inference':
            need_download = not os.path.exists(config['inference_output_file'])
        
        if not need_download:
            print(f"  -> Skipping download for '{config_name}' (output files already exist)")
            continue
        
        try:
            dataset_dict = load_dataset(repo_id, name=config_name)
            print(f"  -> Successfully downloaded '{config_name}'.")
        except Exception as e:
            print(f"  -> FAILURE: Could not download data for config '{config_name}'. Error: {e}")
            continue 
        if config_name == 'main':
            if 'train' in dataset_dict: reconstruct_directory_split(dataset_dict['train'], config['train_output_dir'])
            if 'test' in dataset_dict: reconstruct_directory_split(dataset_dict['test'], config['test_output_dir'])
        elif config_name == 'rule_candidate': reconstruct_single_file(dataset_dict['train'], config['rule_candidate_output_file'], "dict_of_dicts")
        elif config_name == 'grpo_eval': reconstruct_single_file(dataset_dict['train'], config['grpo_eval_output_file'], "dict_of_dicts")
        elif config_name == 'cold_start': reconstruct_single_file(dataset_dict['train'], config['cold_start_output_file'], "list_of_dicts")
        elif config_name == 'grpo_input': reconstruct_single_file(dataset_dict['train'], config['grpo_input_output_file'], "list_of_dicts")
        elif config_name == 'inference': reconstruct_single_file(dataset_dict['train'], config['inference_output_file'], "list_of_dicts", extra_fields={"output": ""})
        else: print(f"  -> Warning: No reconstruction rule defined for config '{config_name}'.")


if __name__ == "__main__":
    load_data(config=CONFIG)