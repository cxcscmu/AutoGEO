"""
AutoGEO API module for document rewriting.
"""
import json
import os
import glob
import threading
from typing import Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from .core import rewrite_document
from ..config import Dataset, get_rewrite_method_name
from ..utils.logger import get_logger


def _process_single_question_rewrite(
    question_id: str,
    data: dict,
    rewrite_method_name: str,
    dataset_enum: Dataset,
    engine_llm: str,
    max_retries: int,
    logger: Any,
    file_lock: threading.Lock,
    filename: str
) -> tuple[bool, bool]:
    """Process a single question for document rewriting.
    
    Returns:
        (success, already_exists) tuple
        - success: True if successfully processed, False otherwise
        - already_exists: True if rewrite result already existed
    """
    rewrite_key = rewrite_method_name + '_text'
    
    if rewrite_key in data[question_id] and data[question_id][rewrite_key]:
        # logger.info(f"Question {question_id} already has rewrite result, loading from saved data...")
        return True, True
    
    try_count = 0
    success = False
    
    while try_count < max_retries:
        try:
            idx = data[question_id]['target_id']
            original_text = data[question_id]["text_list"][idx]
            
            logger.debug(f"Rewriting document for question {question_id}...")
            rewritten_text = rewrite_document(
                document=original_text,
                dataset=dataset_enum.value,
                engine_llm=engine_llm,
            )
            
            with file_lock:
                data[question_id][rewrite_key] = rewritten_text
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
            
            success = True
            break
            
        except Exception as e:
            try_count += 1
            logger.warning(f'Error rewriting question {question_id} (attempt {try_count}/{max_retries}): {e}')
            if try_count >= max_retries:
                logger.error(f"Failed to rewrite question {question_id} after {max_retries} attempts.")
    
    return success, False


def api_rewrite_documents(
    num_examples: Optional[int],
    data_dir: str,
    dataset: str,
    engine_llm: str,
    max_retries: int = 5,
    log_dir: str = "logs",
    logger: Optional[Any] = None,
    max_workers: int = 64,
    progress_bar: Optional[Any] = None
) -> None:
    """Rewrite documents using AutoGEO API.
    
    Args:
        num_examples: Number of examples to process
        data_dir: Directory containing data chunks
        dataset: Name of the dataset (Researchy-GEO, E-commerce, or GEO-Bench)
        engine_llm: External generative engine LLM (gemini, gpt, or claude)
        max_retries: Maximum number of retries for each document
        log_dir: Directory to save log files
        logger: Optional logger instance to use (if None, creates a new one)
        max_workers: Maximum number of parallel workers (default: 64)
        progress_bar: Optional tqdm progress bar to update
    """
    rewrite_method_name = get_rewrite_method_name("autogeo_api", dataset, engine_llm)
    
    if logger is None:
        task_name = f"rewrite_{dataset}_{engine_llm}"
        logger = get_logger(log_dir=log_dir, task_name=task_name)
    logger.info(f"Starting document rewriting: {dataset} with {engine_llm}")
    if num_examples is None:
        logger.info(f"Processing all examples, Method: {rewrite_method_name}")
    else:
        logger.info(f"Number of examples: {num_examples}, Method: {rewrite_method_name}")
    
    try:
        dataset_enum = Dataset(dataset)
    except ValueError:
        logger.error(f"Unsupported dataset: {dataset}")
        raise ValueError(
            f"Unsupported dataset: {dataset}. "
            f"Supported: {[d.value for d in Dataset]}"
        )
    
    from .core import _load_rules_from_file
    _, rule_file_path = _load_rules_from_file(dataset, engine_llm, None)
    if rule_file_path:
        logger.info(f"Using rules from: {rule_file_path}")
    else:
        logger.info(f"Using default rules for {dataset} with {engine_llm}")
    
    processed_questions = 0
    
    logger.info(f"Target number of examples to process: {num_examples if num_examples is not None else 'all'}")
    logger.info(f"Using {max_workers} parallel workers")
    
    total_questions = num_examples
    if num_examples is None:
        total_questions = 0
        chunk_files = sorted(glob.glob(f"{data_dir}/datachunk_*.json"))
        for filename in chunk_files:
            try:
                with open(filename, 'r', encoding="utf-8") as f:
                    data = json.load(f)
                total_questions += len(data)
            except:
                pass
    
    pbar = tqdm(total=total_questions, desc="Step 2: Document Rewriting", unit="question", dynamic_ncols=True) if total_questions is not None else None
    
    chunk_idx = 0
    while num_examples is None or processed_questions < num_examples:
        filename = f"{data_dir}/datachunk_{chunk_idx}.json"
        if not os.path.exists(filename):
            # logger.warning(f"File not found: {filename}. No more chunks available.")
            break
        
        try:
            with open(filename, 'r', encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Error reading {filename}: {e}. Skipping.")
            chunk_idx += 1
            continue
        
        all_question_ids = sorted(list(data.keys()))
        
        if num_examples is None:
            question_id_list = all_question_ids
        else:
            remaining = num_examples - processed_questions
            if remaining <= 0:
                break
            question_id_list = all_question_ids[:remaining]
        
        file_lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='RewriteWorker') as executor:
            future_to_qid = {
                executor.submit(
                    _process_single_question_rewrite,
                    question_id,
                    data,
                    rewrite_method_name,
                    dataset_enum,
                    engine_llm,
                    max_retries,
                    logger,
                    file_lock,
                    filename
                ): question_id
                for question_id in question_id_list
            }
            
            for future in as_completed(future_to_qid):
                question_id = future_to_qid[future]
                try:
                    success, already_exists = future.result()
                    processed_questions += 1
                    
                    if pbar is not None:
                        pbar.update(1)
                    
                    if not success and not already_exists:
                        logger.error(f"Failed to process question {question_id}")
                    
                    if num_examples is not None and processed_questions >= num_examples:
                        for f in future_to_qid:
                            f.cancel()
                        break
                        
                except Exception as e:
                    logger.error(f'Error processing question {question_id}: {e}')
        
        if num_examples is not None and processed_questions >= num_examples:
            break
        
        chunk_idx += 1
    
    if pbar is not None:
        pbar.close()
    
    if num_examples is not None:
        logger.log_progress(processed_questions, num_examples, "questions")
        logger.info(f"Document rewriting completed. Processed {processed_questions}/{num_examples} questions.")
    else:
        logger.info(f"Document rewriting completed. Processed {processed_questions} questions in total.")
    # Don't close logger if it was provided externally
