import json
import os
from ..rewriters import get_rewrite_prompt_template
from ..config import get_full_path
from .utils import parse_rewrite_method

def inference_data_construct(
    num_examples: int = 100,
    data_dir: str = "data/Researchy-GEO/test",
    rewrite_method_name: str = "autogeo_researchy_geo_gemini",
    dataset: str = "Researchy-GEO",
    output_dir: str = None
) -> None:
    """Construct inference data for model evaluation.
    
    Args:
        num_examples: Number of examples to process
        data_dir: Directory containing data chunks
        rewrite_method_name: Name of the rewrite method used
        dataset: Name of the dataset (Researchy-GEO, E-commerce, or GEO-Bench)
        output_dir: Output file path for inference data (default: uses config)
    """
    parsed_dataset, engine_llm = parse_rewrite_method(rewrite_method_name)
    if dataset is None or dataset == "Researchy-GEO":
        dataset = parsed_dataset
    if output_dir is None:
        output_dir = get_full_path(dataset, 'inference_file')
    prompt_template = get_rewrite_prompt_template(parsed_dataset, engine_llm)
    
    inference_samples = []
    chunk_indices = list(range(ranges[0], ranges[1]))
    
    for chunk_idx in chunk_indices:
        filename = os.path.join(data_dir, f"datachunk_{chunk_idx}.json")
        if not os.path.exists(filename):
            # print(f"Warning: File not found: {filename}. Skipping.")
            continue
        with open(filename, "r", encoding="utf-8") as f:
            chunk_data = json.load(f)
        record_ids = list(chunk_data.keys())
        for record_id in record_ids:
            original_text = chunk_data[record_id]["text_list"][chunk_data[record_id]["target_id"]]
            sample_dict = {} 
            sample_dict["instruction"] = prompt_template
            sample_dict["input"] = "source: \n\n" + original_text
            sample_dict["output"] = ""
            inference_samples.append(sample_dict)

    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    with open(output_dir, "w", encoding="utf-8") as f:  
        json.dump(inference_samples, f, ensure_ascii=False, indent=4)
    print(f"Total samples: {len(inference_samples)}")

if __name__ == "__main__":
    ranges = (0, 10)  
    inference_data_construct(
        ranges=ranges,
        data_dir="data/Researchy-GEO/test",
        rewrite_method_name="autogeo_researchy_geo_gemini",
        dataset="Researchy-GEO"
    )