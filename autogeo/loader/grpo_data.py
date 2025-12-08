import json
import os
from ..rewriters import get_rewrite_prompt_template
from ..config import get_full_path
from .utils import parse_rewrite_method

def grpo_evaluation_construct(
    num_examples: int = 100,
    data_dir: str = "data/Researchy-GEO/test",
    engine_llm: str = "gemini",
    dataset: str = "Researchy-GEO",
    output_dir: str = None
) -> None:
    """Construct evaluation data for GRPO training.
    
    Args:
        num_examples: Number of examples to process
        data_dir: Directory containing data chunks
        engine_llm: Engine type used for evaluation (gemini, gpt, or claude)
        dataset: Name of the dataset (Researchy-GEO, E-commerce, or GEO-Bench)
        output_dir: Output file path for evaluation data (default: uses config)
    """
    if output_dir is None:
        output_dir = get_full_path(dataset, 'grpo_eval_file')
    evaluation_data = {}
    processed_count = 0
    chunk_idx = 0
    
    while processed_count < num_examples:
        filename = os.path.join(data_dir, f"datachunk_{chunk_idx}.json")
        if not os.path.exists(filename):
            chunk_idx += 1
            if chunk_idx > 1000:  # Safety limit
                break
            continue
        
        with open(filename, "r", encoding="utf-8") as f:
            chunk_data = json.load(f)
        
        record_ids = list(chunk_data.keys())
        for record_id in record_ids:
            if processed_count >= num_examples:
                break
            if "query" not in chunk_data[record_id] or "text_list" not in chunk_data[record_id] or "target_id" not in chunk_data[record_id] or engine_llm + "_geo_score" not in chunk_data[record_id] or "ori_keypoint_dict" not in chunk_data[record_id]:
                processed_count += 1
                continue
            eval_dict = {}
            eval_dict["query"] = chunk_data[record_id]["query"]
            eval_dict["text_list"] = chunk_data[record_id]["text_list"]
            eval_dict["target_id"] = chunk_data[record_id]["target_id"]
            eval_dict["ori_object_dict"] = chunk_data[record_id][engine_llm + "_geo_score"]
            eval_dict["ori_keypoint_dict"] = chunk_data[record_id]["ori_keypoint_dict"]
            evaluation_data[record_id] = eval_dict
            processed_count += 1
        
        if processed_count >= num_examples:
            break
        chunk_idx += 1
        print(f"Processed chunk {chunk_idx}")

    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    with open(output_dir, "w", encoding="utf-8") as f:
        json.dump(evaluation_data, f, ensure_ascii=False, indent=4)
    print(f"Total evaluation records: {len(evaluation_data)}")

def grpo_data_source_construct(
    num_examples: int = 100,
    data_dir: str = "data/Researchy-GEO/test",
    engine_llm: str = "gemini",
    rewrite_method_name: str = "autogeo_researchy_geo_gemini",
    dataset: str = "Researchy-GEO",
    output_dir: str = None
) -> None:
    """Construct training data source for GRPO.
    
    Args:
        num_examples: Number of examples to process
        data_dir: Directory containing data chunks
        engine_llm: Engine type used for evaluation (gemini, gpt, or claude)
        rewrite_method_name: Name of the rewrite method used
        dataset: Name of the dataset (Researchy-GEO, E-commerce, or GEO-Bench)
        output_dir: Output file path for training data (default: uses config)
    """
    parsed_dataset, parsed_engine_llm = parse_rewrite_method(rewrite_method_name)
    if dataset is None or dataset == "Researchy-GEO":
        dataset = parsed_dataset
    if output_dir is None:
        output_dir = get_full_path(dataset, 'grpo_input_file')
    prompt_template = get_rewrite_prompt_template(parsed_dataset, parsed_engine_llm)
    
    training_samples = []
    processed_count = 0
    chunk_idx = 0
    
    while processed_count < num_examples:
        filename = os.path.join(data_dir, f"datachunk_{chunk_idx}.json")
        if not os.path.exists(filename):
            chunk_idx += 1
            if chunk_idx > 1000:  # Safety limit
                break
            continue
        
        with open(filename, "r", encoding="utf-8") as f:
            chunk_data = json.load(f)
        
        record_ids = list(chunk_data.keys())
        for record_id in record_ids:
            if processed_count >= num_examples:
                break
            if "query" not in chunk_data[record_id] or "text_list" not in chunk_data[record_id] or "target_id" not in chunk_data[record_id] or engine_llm + "_geo_score" not in chunk_data[record_id] or "ori_keypoint_dict" not in chunk_data[record_id]:
                processed_count += 1
                continue
            original_text = chunk_data[record_id]["text_list"][chunk_data[record_id]["target_id"]]
            sample_dict = {} 
            sample_dict["problem"] = "task description: \n" + prompt_template + "\n source text: \n" + original_text
            sample_dict["solution"] = record_id
            training_samples.append(sample_dict)
            processed_count += 1
        
        if processed_count >= num_examples:
            break
        chunk_idx += 1

    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    with open(output_dir, "w", encoding="utf-8") as f:
        json.dump(training_samples, f, ensure_ascii=False, indent=4)
    print(f"Total samples: {len(training_samples)}")

if __name__ == "__main__":
    grpo_evaluation_construct(
        num_examples=100,
        data_dir="data/Researchy-GEO/test",
        engine_llm="gemini",
        dataset="Researchy-GEO"
    )
    grpo_data_source_construct(
        num_examples=100,
        data_dir="data/Researchy-GEO/test",
        engine_llm="gemini",
        rewrite_method_name="autogeo_researchy_geo_gemini",
        dataset="Researchy-GEO"
    )