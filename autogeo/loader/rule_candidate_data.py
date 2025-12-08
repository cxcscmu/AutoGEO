import json
import os
from ..evaluation.metrics import impression_wordpos_count_simple, impression_word_count_simple, impression_pos_count_simple, extract_citations_new
from ..config import Dataset, get_full_path

def rule_candidate_data(
    num_examples: int = 100,
    data_dir: str = "data/Researchy-GEO/test",
    engine_llm: str = "gemini",
    dataset: str = "Researchy-GEO"
) -> None:
    """Construct rule candidate data from evaluation results.
    
    Args:
        num_examples: Number of examples to process
        data_dir: Directory containing data chunks
        engine_llm: LLM name used for evaluation (gemini, gpt, or claude)
        dataset: Name of the dataset (Researchy-GEO, E-commerce, or GEO-Bench)
    """
    auto_rule_dict = {}
    processed_count = 0
    chunk_idx = 0
    
    while processed_count < num_examples:
        filename = os.path.join(data_dir, f"datachunk_{chunk_idx}.json")
        if not os.path.exists(filename):
            chunk_idx += 1
            if chunk_idx > 1000:  # Safety limit
                break
            continue
        
        with open(filename, 'r', encoding="utf-8") as f:
            data = json.load(f)
        
        question_id_list = list(data.keys())
        for question_id in question_id_list:
            if processed_count >= num_examples:
                break
            print(f"Evaluating question {question_id}")
            try:
                auto_rule_dict[question_id] = {}
                text_list = data[question_id]["text_list"]
                auto_rule_dict[question_id]['query'] = data[question_id]["query"]
                original_response = data[question_id][engine_llm + "_response"]  
                score_list = []
                for idx in range(len(text_list)):
                    citations = extract_citations_new(original_response)
                    if citations and idx < len(citations):
                        score_list.append(
                            impression_wordpos_count_simple(citations)[idx] +
                            impression_word_count_simple(citations)[idx] +
                            impression_pos_count_simple(citations)[idx]
                        )
                    else:
                        score_list.append(0)
                
                if not score_list:
                    print(f"Skipping {question_id}: No valid scores generated.")
                    if question_id in auto_rule_dict:
                        del auto_rule_dict[question_id]
                    continue
                    
                min_score_index = score_list.index(min(score_list))
                max_score_index = score_list.index(max(score_list))
                auto_rule_dict[question_id]['bad_document'] = text_list[min_score_index]
                auto_rule_dict[question_id]['good_document'] = text_list[max_score_index]
                auto_rule_dict[question_id]['winner'] = "doc_a" if max_score_index == data[question_id].get('target_id', -1) else "doc_b"
                auto_rule_dict[question_id]['document_a'] = text_list[max_score_index]
                auto_rule_dict[question_id]['document_b'] = text_list[min_score_index]
                auto_rule_dict[question_id]['good_document_content'] = text_list[max_score_index]
                auto_rule_dict[question_id]['bad_document_content'] = text_list[min_score_index]
                print(f"Finished evaluating question {question_id}")
                processed_count += 1

            except Exception as e:
                print(f'Error evaluation: {e}')
                if question_id in auto_rule_dict:
                    del auto_rule_dict[question_id]
                processed_count += 1
        
        if processed_count >= num_examples:
            break
        chunk_idx += 1

    # Use config to get output path
    try:
        dataset_enum = Dataset(dataset)
        output_path = get_full_path(dataset_enum, "rule_candidate_file")
    except (ValueError, KeyError):
        # Fallback to old path format
        dataset_short = dataset.replace("-", "_").replace(" ", "_").lower()
        output_path = os.path.join(data_dir.rsplit('/', 1)[0], "RL", f"{dataset}_rule_candidate.json")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(auto_rule_dict, f, ensure_ascii=False, indent=4)
    print(f"Total rule candidate samples: {len(auto_rule_dict)}")

if __name__ == "__main__":
    rule_candidate_data(
        num_examples=100,
        data_dir="data/Researchy-GEO/test",
        engine_llm="gemini",
        dataset="Researchy-GEO"
    )