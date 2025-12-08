import json
import os
from ..rewriters import get_rewrite_prompt_template
from ..config import get_full_path
from ..utils import call_gemini
from .utils import parse_rewrite_method
from dotenv import load_dotenv

load_dotenv("keys.env")


CLEANING_SYSTEM_PROMPT = """
You are a highly specialized text formatting bot. Your ONLY job is to extract the core message from a given text and present it in a strict format.

Follow these rules STRICTLY:
1.  You will be given a text that might contain introductory phrases like "Here is the rewritten source:", "regenerate source:", or other conversational text.
2.  Your task is to IGNORE and REMOVE all such introductory phrases.
3.  You must extract ONLY the actual rewritten content that comes after the introduction.
4.  You MUST preserve the extracted content VERBATIM. Do not change any words, punctuation, or line breaks within the original content.
5.  Your entire output MUST start with the exact string "**Rewritten Source: **" (including the two asterisks, the space, and the colon).
6.  There should be NO text, NO explanation, and NO newlines before or after your formatted output. Your response must be ONLY the required output and nothing else.
"""


def filter_records(
    data: dict,
    rewrite_method_name: str = "autogeo_researchy_geo_gemini",
    engine_llm: str = "gemini",
    thresholds: float = 0.8
) -> list:
    """Filter records that meet quality thresholds.
    
    Args:
        data: Dictionary of records to filter
        rewrite_method_name: Name of the rewrite method used
        engine_llm: Engine type used for evaluation (gemini, gpt, or claude)
        thresholds: Minimum support rate threshold (default: 0.8)
        
    Returns:
        List of qualified record IDs
    """
    qualified_ids = []

    for record_id, record in data.items():
        original_obj = record.get(engine_llm + "_geo_score", {}) 
        new_obj = record.get(rewrite_method_name + "_dict", {})
        comparison_metrics = record.get(rewrite_method_name + "_coverage", {})
        if not (original_obj and new_obj and comparison_metrics):
            continue                   
        if not all(new_obj[k] > original_obj.get(k, -1) for k in original_obj):
            continue                     
        is_qualified = True

        if comparison_metrics.get("contradicted_count", 1) != 0:
            is_qualified = False
        def support_rate(m):
            denom = m.get("supported_count", 0) + m.get("omitted_count", 0) + m.get("contradicted_count", 0)
            return m.get("supported_count", 0) / denom if denom else 0
        if support_rate(comparison_metrics) < thresholds:
            is_qualified = False

        if not is_qualified:
            continue
        qualified_ids.append(record_id)
    return qualified_ids



def clean_rewritten_text_with_gemini(text_to_clean: str) -> str:
    """Clean rewritten text by removing introductory phrases.
    
    Args:
        text_to_clean: Text that may contain introductory phrases
        
    Returns:
        Cleaned text starting with "**Rewritten Source: **"
    """
    user_prompt = f"""
Here are examples of how to perform your task.

--- Example 1 ---
Input Text:
"Of course! Here is the rewritten source:

This is the first sentence.
This is the second sentence."

Your Output:
"**Rewritten Source: **This is the first sentence.
This is the second sentence."
---

--- Example 2 ---
Input Text:
"regenerate source: This version is more concise and impactful."

Your Output:
"**Rewritten Source: **This version is more concise and impactful."
---

--- Example 3 ---
Input Text:
"**Rewritten Source**

Here is the final text."

Your Output:
"**Rewritten Source: **Here is the final text."
---

Now, perform the task on the following text. Remember to follow all the rules exactly.

Input Text:
"{text_to_clean}"

Your Output:
"""

    print("--- Calling Gemini for cleaning task ---")
    cleaned_text = call_gemini(
        user_prompt=user_prompt.strip(),
        system_prompt=CLEANING_SYSTEM_PROMPT
    )
    return cleaned_text.strip()


def cold_start(
    num_examples: int = 100,
    data_dir: str = "data/Researchy-GEO/train",
    rewrite_method_name: str = "autogeo_researchy_geo_gemini",
    engine_llm: str = "gemini",
    thresholds: float = 0.8,
    dataset: str = "Researchy-GEO",
    output_dir: str = None
) -> None:
    """Generate cold start training data from filtered records.
    
    Args:
        num_examples: Number of examples to process
        data_dir: Directory containing data chunks
        rewrite_method_name: Name of the rewrite method used
        engine_llm: Engine type used for evaluation (gemini, gpt, or claude)
        thresholds: Minimum support rate threshold (default: 0.8)
        dataset: Name of the dataset (Researchy-GEO, E-commerce, or GEO-Bench)
        output_dir: Output file path for training data (default: uses config)
    """
    parsed_dataset, parsed_engine_llm = parse_rewrite_method(rewrite_method_name)
    if dataset is None or dataset == "Researchy-GEO":
        dataset = parsed_dataset
    if output_dir is None:
        output_dir = get_full_path(dataset, 'finetune_file')
    prompt_template = get_rewrite_prompt_template(parsed_dataset, parsed_engine_llm)
    
    qualified_ids = []
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
            processed_count += 1
        
        valid_ids = filter_records(chunk_data, rewrite_method_name=rewrite_method_name, engine_llm=engine_llm, thresholds=thresholds)
        qualified_ids.extend(valid_ids)
        
        if processed_count >= num_examples:
            break
        chunk_idx += 1

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
            if record_id in qualified_ids:
                original_text = chunk_data[record_id]["text_list"][chunk_data[record_id]["target_id"]]
                sample_dict = {} 
                sample_dict["instruction"] = prompt_template
                sample_dict["input"] = "source: \n\n" + original_text
                sample_dict["output"] = chunk_data[record_id][rewrite_method_name + "_text"]
                training_samples.append(sample_dict)
            processed_count += 1
        
        if processed_count >= num_examples:
            break
        chunk_idx += 1

    count = 0
    for sample in training_samples:
        cleaned_text = clean_rewritten_text_with_gemini(sample["output"])
        sample["output"] = cleaned_text
        count += 1
        print(f"Cleaned {count} samples so far...")

    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    with open(output_dir, "w", encoding="utf-8") as f:
        json.dump(training_samples, f, ensure_ascii=False, indent=4)

    print(f"Total samples: {len(training_samples)}")

