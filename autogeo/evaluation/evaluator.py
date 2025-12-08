import json
import openai
import os
import sys
import glob
import nltk
import threading
import pprint
from pathlib import Path
from typing import Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from .metrics import *
from .aggregate_results import aggregate_json_files
from .generative_engine import generate_answer_gemini, generate_answer_gpt, generate_answer_claude
from ..utils.logger import get_logger

nltk.download('punkt_tab', quiet=True)
load_dotenv("keys.env")
openai.api_key = os.getenv("OPENAI_API_KEY")

def process_prediction_text(text: str, length: int = 3000) -> str:
    """Process prediction text to extract rewritten content.
    
    Args:
        text: Raw prediction text that may contain introductory phrases
        length: Minimum length threshold for valid text
        
    Returns:
        Extracted text content
    """
    # keywords = ["**Rewritten Source", "**Regenerated Source"]
    # last_pos = -1
    # found_keyword = None
    # for kw in keywords:
    #     pos = text.rfind(kw)  
    #     if pos > last_pos:
    #         last_pos = pos
    #         found_keyword = kw

    # if last_pos == -1:
    #     return None
    # remaining_text = text[last_pos + len(found_keyword):]
    # first_star_pos = remaining_text.find('*')
    # if first_star_pos == -1:
    #     return None

    # current_pos = first_star_pos
    # while current_pos < len(remaining_text) and remaining_text[current_pos] == '*':
    #     current_pos += 1
    # while current_pos < len(remaining_text) and remaining_text[current_pos] == '\n':
    #     current_pos += 1
    # final_text = remaining_text[current_pos:]
    # if len(final_text) < length:
    #     return None
    # return final_text

    return text

def _process_single_question_autogeo_api(
    question_id: str,
    data: dict,
    rewrite_method_name: str,
    engine_llm: str,
    rewrite_key: str,
    logger: Any,
    file_lock: threading.Lock,
    filename: str
) -> tuple[bool, bool]:
    """Process a single question for AutoGEO API evaluation.
    
    Returns:
        (success, already_exists) tuple
        - success: True if successfully processed, False otherwise
        - already_exists: True if response and score already existed
    """
    try:
        response_key = rewrite_method_name + '_response'
        score_key = rewrite_method_name + '_geo_score'
        
        if (response_key in data[question_id] and data[question_id][response_key] and
            score_key in data[question_id] and data[question_id][score_key]):
            return True, True
        
        query = data[question_id]["query"]
        idx = data[question_id]['target_id']
        rewritten_text_list = data[question_id]["text_list"].copy()
        rewritten_text = data[question_id][rewrite_key]
        
        if rewritten_text is None:
            logger.warning(f"Messy format for question {question_id}, skipping")
            return False, False
        
        rewritten_text_list[idx] = rewritten_text
        
        if response_key in data[question_id] and data[question_id][response_key]:
            rewritten_response = data[question_id][response_key]
        else:
            max_retries = 3
            try_count = 0
            rewritten_response = None
            
            while try_count < max_retries:
                try:
                    if "gpt" in engine_llm:
                        rewritten_response = generate_answer_gpt(query, rewritten_text_list, model_name=engine_llm)
                    elif "claude" in engine_llm:
                        rewritten_response = generate_answer_claude(query, rewritten_text_list, model_name=engine_llm)
                    elif "gemini" in engine_llm:
                        rewritten_response = generate_answer_gemini(query, rewritten_text_list, model_name=engine_llm)
                    else:
                        logger.error(f"Unknown engine_llm: {engine_llm}")
                        return False, False
                    break
                except Exception as e:
                    try_count += 1
                    if try_count < max_retries:
                        logger.warning(f'Error calling {engine_llm} API for answer generation (attempt {try_count}/{max_retries}): {e}')
                    else:
                        logger.error(f"Failed to generate answer for question {question_id} after {max_retries} attempts: {e}")
            
            if rewritten_response is None:
                logger.warning(f"Failed to generate response for question {question_id}, skipping...")
                return False, False
        
        if score_key in data[question_id] and data[question_id][score_key]:
            score_data = data[question_id][score_key]
        else:
            citations = extract_citations_new(rewritten_response)
            score_data = {
                'wordpos': impression_wordpos_count_simple(citations)[idx],
                'word': impression_word_count_simple(citations)[idx],
                'pos': impression_pos_count_simple(citations)[idx]
            }
        
        with file_lock:
            if response_key not in data[question_id] or not data[question_id][response_key]:
                data[question_id][response_key] = rewritten_response
            if score_key not in data[question_id] or not data[question_id][score_key]:
                data[question_id][score_key] = score_data
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        
        return True, False
        
    except Exception as e:
        logger.error(f'Error evaluating question {question_id}: {e}')
        return False, False


def autogeo_evaluation(
    num_examples: Optional[int] = 100,
    data_dir: str = "data/Researchy-GEO/test",
    engine_llm: str = "gemini",
    rewrite_method_name: str = "autogeo_api_researchy_geo_gemini",
    need_geu_score: bool = True,
    log_dir: str = "logs",
    logger: Optional[Any] = None,
    max_workers: int = 64
) -> None:
    """Evaluate AutoGEO API rewritten documents.
    
    Args:
        num_examples: Number of examples to process
        data_dir: Directory containing data chunks
        engine_llm: LLM name used for answer generation (gemini, gpt, or claude)
        rewrite_method_name: Name of the rewrite method used
        need_geu_score: Whether to compute geu score (default: True)
        log_dir: Directory to save log files
        logger: Optional logger instance to use (if None, creates a new one)
        max_workers: Maximum number of parallel workers (default: 64)
    """
    if logger is None:
        task_name = f"eval_autogeo_{rewrite_method_name}_{engine_llm}"
        logger = get_logger(log_dir=log_dir, task_name=task_name)
    logger.info(f"Starting evaluation: {rewrite_method_name}")
    if num_examples is None:
        logger.info(f"Processing all examples, Data dir: {data_dir}")
    else:
        logger.info(f"Number of examples: {num_examples}, Data dir: {data_dir}")
    
    logger.info(f"Target number of examples to evaluate: {num_examples if num_examples is not None else 'all'}")
    logger.info(f"Using {max_workers} parallel workers")
    count = 0
    
    total_questions = num_examples
    rewrite_key = rewrite_method_name + '_text'
    if num_examples is None:
        total_questions = 0
        chunk_files = sorted(glob.glob(f"{data_dir}/datachunk_*.json"))
        for filename in chunk_files:
            try:
                with open(filename, 'r', encoding="utf-8") as f:
                    data = json.load(f)
                for qid in data.keys():
                    if rewrite_key in data[qid] and data[qid][rewrite_key]:
                        total_questions += 1
            except:
                pass
    
    pbar = tqdm(total=total_questions, desc="Step 3: AutoGEO Evaluation", unit="question", dynamic_ncols=True) if total_questions is not None else None
    
    chunk_idx = 0
    
    while num_examples is None or count < num_examples:
        filename = f"{data_dir}/datachunk_{chunk_idx}.json"
        if not os.path.exists(filename):
            # logger.warning(f"File not found: {filename}. No more chunks available.")
            break
            
        with open(filename, 'r', encoding="utf-8") as f:
            data = json.load(f)
        all_question_ids = sorted(list(data.keys()))
        
        if num_examples is None:
            question_id_list = all_question_ids
        else:
            remaining = num_examples - count
            if remaining <= 0:
                break
            question_id_list = all_question_ids[:remaining]
        
        questions_with_rewrite = [qid for qid in question_id_list 
                                  if rewrite_key in data[qid] and data[qid][rewrite_key]]
        
        if not questions_with_rewrite:
            chunk_idx += 1
            continue
        
        file_lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='AutoGEOEval') as executor:
            future_to_qid = {
                executor.submit(
                    _process_single_question_autogeo_api,
                    question_id,
                    data,
                    rewrite_method_name,
                    engine_llm,
                    rewrite_key,
                    logger,
                    file_lock,
                    filename
                ): question_id
                for question_id in questions_with_rewrite
            }
            
            for future in as_completed(future_to_qid):
                question_id = future_to_qid[future]
                try:
                    success, already_exists = future.result()
                    if success:
                        count += 1
                        if pbar is not None:
                            pbar.update(1)
                except Exception as e:
                    logger.error(f'Error processing question {question_id}: {e}')
        
        chunk_idx += 1
    
    if pbar is not None:
        pbar.close()
    
    if num_examples is not None:
        logger.log_progress(count, num_examples, "questions")
        logger.info(f"Geo score evaluation completed. Processed {count}/{num_examples} questions.")
    else:
        logger.info(f"Geo score evaluation completed. Processed {count} questions in total.")
    
    geo_results = aggregate_json_files(data_dir, rewrite_method_name + "_geo_score")
    logger.info(f"\nGeo Score Results:")
    logger.info(pprint.pformat(geo_results))

    if need_geu_score:
        geu_score(
            num_examples=num_examples,
            engine_llm=rewrite_method_name,
            rewrite_method_name=rewrite_method_name,
            metrics_to_evaluate=["citation_quality", "quality_dimensions", "keypoint_coverage"] if 'Researchy-GEO' in data_dir else ["citation_quality", "quality_dimensions"], # only Researchy-GEO has keypoint annotations
            max_workers=max_workers,
            data_dir=data_dir,
            text_list_name="text_list",
            logger=logger
        )
        geu_results = aggregate_json_files(data_dir, rewrite_method_name + "_geu_score")
        logger.info(f"\nGeu Score Results:")
        logger.info(pprint.pformat(geu_results))


