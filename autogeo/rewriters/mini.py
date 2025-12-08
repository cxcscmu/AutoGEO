"""
AutoGEO Mini module for document rewriting using local models.
"""
import json
import os
import glob
from typing import Optional, Any
from tqdm import tqdm
from transformers import AutoTokenizer
from ..evaluation.evaluator import process_prediction_text

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


def generate_inference_with_vllm(
    model_path: str,
    data_dir: str,
    rewrite_method_name: str,
    rules_string: str,
    num_examples: int = None,
    batch_size: int = 1024,
    temperature: float = 0.95,
    top_p: float = 0.7,
    top_k: int = 50,
    max_new_tokens: int = 2048,
    repetition_penalty: float = 1.0,
    logger=None
):
    """Generate inference results using vLLM engine (optimized for speed).
    
    Args:
        model_path: Path to the model checkpoint
        data_dir: Directory containing data chunks
        rewrite_method_name: Name of the rewrite method
        num_examples: Number of examples to process (None for all)
        batch_size: Batch size for inference
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        max_new_tokens: Maximum number of new tokens to generate
        repetition_penalty: Repetition penalty
        logger: Logger instance
    """
    if not VLLM_AVAILABLE:
        raise ImportError("vllm is required for optimized inference. Please install it with: pip install vllm")
    
    if logger:
        logger.info(f"Loading model with vLLM from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=8192 + max_new_tokens,
        tensor_parallel_size=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        disable_log_stats=True,
    )
    
    if logger:
        logger.info("vLLM model loaded successfully")
    
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        skip_special_tokens=True,
    )
    
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
    
    pbar = tqdm(total=total_questions, desc="Generating predictions (vLLM)", unit="question", dynamic_ncols=True)
    
    processed_count = 0
    chunk_idx = 0
    
    while num_examples is None or processed_count < num_examples:
        filename = f"{data_dir}/datachunk_{chunk_idx}.json"
        
        if not os.path.exists(filename):
            break
        
        try:
            with open(filename, 'r', encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            if logger:
                logger.warning(f"Error reading {filename}: {e}. Skipping.")
            chunk_idx += 1
            continue
        
        # Sort question IDs for consistent order
        all_question_ids = sorted(list(data.keys()))
        
        # Limit questions based on remaining count
        if num_examples is None:
            question_id_list = all_question_ids
        else:
            remaining = num_examples - processed_count
            if remaining <= 0:
                break
            question_id_list = all_question_ids[:remaining]
        
        # Process questions in batches
        text_key = f"{rewrite_method_name}_text"
        
        for batch_start in range(0, len(question_id_list), batch_size):
            batch_end = min(batch_start + batch_size, len(question_id_list))
            batch_question_ids = question_id_list[batch_start:batch_end]
            
            # Collect batch data
            batch_prompts = []
            batch_metadata = []
            skipped_count = 0
            
            for question_id in batch_question_ids:
                question_data = data[question_id]
                
                # Check if rewrite result already exists
                if text_key in question_data and question_data[text_key]:
                    skipped_count += 1
                    continue
                
                text_list = question_data.get("text_list", [])
                target_id = question_data.get("target_id", 0)
                
                if target_id >= len(text_list):
                    if logger:
                        logger.warning(f"Invalid target_id {target_id} for question {question_id}")
                    skipped_count += 1
                    continue
                
                document = text_list[target_id]
                
                # Construct prompt (similar to training data format)
                prompt = f"""Here is the source:
{document}

You are given a website document as a source. This source, along with other sources, will be used by a language model (LLM) to generate answers to user questions, with each line in the generated answer being cited with its original source. Your task, as the owner of the source, is to **rewrite your document in a way that maximizes its visibility and impact in the LLM's final answer, ensuring your source is more likely to be quoted and cited**.

You can regenerate the provided source so that it strictly adheres to the "Quality Guidelines", and you can also apply any other methods or techniques, as long as they help your rewritten source text rank higher in terms of relevance, authority, and impact in the LLM's generated answers.

## Quality Guidelines to Follow:

{rules_string}
""".strip()
                batch_prompts.append(prompt)
                batch_metadata.append((question_id, question_data))
            
            # Update progress for skipped questions
            if skipped_count > 0:
                pbar.update(skipped_count)
                processed_count += skipped_count
            
            if not batch_prompts:
                continue
            
            # Generate with vLLM (disable vLLM's internal progress bar)
            # vLLM uses tqdm which writes to stderr, so we suppress it
            from contextlib import redirect_stderr
            from io import StringIO
            
            with redirect_stderr(StringIO()):
                outputs = llm.generate(batch_prompts, sampling_params)
            
            for i, (question_id, question_data) in enumerate(batch_metadata):
                if question_data is None:
                    continue
                
                if i < len(outputs):
                    generated_text = outputs[i].outputs[0].text
                    
                    processed_text = process_prediction_text(generated_text)
                    
                    if processed_text:
                        question_data[text_key] = processed_text
            
            pbar.update(len(batch_prompts))
            processed_count += len(batch_prompts)
        
        # Save modified chunk
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        chunk_idx += 1
    
    pbar.close()
    
    if logger:
        logger.info(f"Completed inference for {processed_count} questions using vLLM")


def generate_inference_with_transformers(
    model_path: str,
    data_dir: str,
    rewrite_method_name: str,
    rules_string: str,
    num_examples: int = None,
    batch_size: int = 32,
    logger=None
):
    """Generate inference results and directly write to data chunks (like api_rewrite_documents).
    
    Args:
        model_path: Path to the model checkpoint
        data_dir: Directory containing data chunks
        rewrite_method_name: Name of the rewrite method
        num_examples: Number of examples to process (None for all)
        batch_size: Batch size for inference
        logger: Logger instance
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers and torch are required for autogeo_mini. Please install them.")
    
    if logger:
        logger.info(f"Loading model from {model_path}...")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    if hasattr(model, 'hf_device_map'):
        device_info = f"Model device map: {model.hf_device_map}"
    else:
        device_info = f"Model device: {next(model.parameters()).device}"
    
    if logger:
        logger.info("Model loaded successfully")
        logger.info(device_info)
    else:
        print(device_info)
    
    from ..evaluation.evaluator import process_prediction_text
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
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
    
    pbar = tqdm(total=total_questions, desc="Generating predictions", unit="question", dynamic_ncols=True)
    
    processed_count = 0
    chunk_idx = 0
    
    while num_examples is None or processed_count < num_examples:
        filename = f"{data_dir}/datachunk_{chunk_idx}.json"
        
        if not os.path.exists(filename):
            break
        
        try:
            with open(filename, 'r', encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            if logger:
                logger.warning(f"Error reading {filename}: {e}. Skipping.")
            chunk_idx += 1
            continue
        
        all_question_ids = sorted(list(data.keys()))
        
        if num_examples is None:
            question_id_list = all_question_ids
        else:
            remaining = num_examples - processed_count
            if remaining <= 0:
                break
            question_id_list = all_question_ids[:remaining]
        
        text_key = f"{rewrite_method_name}_text"
        
        for batch_start in range(0, len(question_id_list), batch_size):
            batch_end = min(batch_start + batch_size, len(question_id_list))
            batch_question_ids = question_id_list[batch_start:batch_end]
            
            batch_prompts = []
            batch_metadata = []
            skipped_count = 0
            
            for question_id in batch_question_ids:
                question_data = data[question_id]
                
                if text_key in question_data and question_data[text_key]:
                    skipped_count += 1
                    continue
                
                query = question_data.get("query", "")
                text_list = question_data.get("text_list", [])
                target_id = question_data.get("target_id", 0)
                
                if target_id >= len(text_list):
                    if logger:
                        logger.warning(f"Invalid target_id {target_id} for question {question_id}")
                    skipped_count += 1
                    continue
                
                document = text_list[target_id]
                
                prompt = f"""Here is the source:
{document}

You are given a website document as a source. This source, along with other sources, will be used by a language model (LLM) to generate answers to user questions, with each line in the generated answer being cited with its original source. Your task, as the owner of the source, is to **rewrite your document in a way that maximizes its visibility and impact in the LLM's final answer, ensuring your source is more likely to be quoted and cited**.

You can regenerate the provided source so that it strictly adheres to the "Quality Guidelines", and you can also apply any other methods or techniques, as long as they help your rewritten source text rank higher in terms of relevance, authority, and impact in the LLM's generated answers.

## Quality Guidelines to Follow:

{rules_string}
""".strip()
                batch_prompts.append(prompt)
                batch_metadata.append((question_id, question_data))
            
            if skipped_count > 0:
                pbar.update(skipped_count)
                processed_count += skipped_count
            
            if not batch_prompts or all(p == "" for p in batch_prompts):
                continue
            
            with torch.no_grad():
                inputs = tokenizer(
                    batch_prompts, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=3000,
                    padding=True
                ).to(model.device)
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    temperature=0.95,
                    top_p=0.7,
                    top_k=50,
                    repetition_penalty=1.0,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                
                for i, (question_id, question_data) in enumerate(batch_metadata):
                    if question_data is None:
                        continue
                    
                    input_length = len(inputs.input_ids[i])
                    generated_text = tokenizer.decode(outputs[i][input_length:], skip_special_tokens=True)
                    
                    processed_text = process_prediction_text(generated_text)
                    
                    if processed_text:
                        question_data[text_key] = processed_text
            
            pbar.update(len(batch_prompts))
            processed_count += len(batch_prompts)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        chunk_idx += 1
    
    pbar.close()
    
    if logger:
        logger.info(f"Completed inference for {processed_count} questions")


def mini_rewrite_documents(
    model_path: str,
    data_dir: str,
    rewrite_method_name: str,
    dataset: str,
    engine_llm: str,
    num_examples: Optional[int] = None,
    batch_size: int = 1024,
    logger: Optional[Any] = None
) -> None:
    """Rewrite documents using AutoGEO Mini (local model).
    
    Args:
        model_path: Path to the model checkpoint
        data_dir: Directory containing data chunks
        rewrite_method_name: Name of the rewrite method
        num_examples: Number of examples to process (None for all)
        batch_size: Batch size for inference
        logger: Optional logger instance to use
    """
    if logger:
        logger.info(f"Starting document rewriting with AutoGEO Mini")
        logger.info(f"Model Path: {model_path}")
    
    from .core import _load_rules_from_file
    rule_list, rule_file_path = _load_rules_from_file(dataset, engine_llm, None)
    rules_string = "\n".join(rule_list)
    
    if rule_file_path:
        logger.info(f"Using rules from: {rule_file_path}")
    else:
        logger.info(f"Using default rules for {dataset} with {engine_llm}")
    
    try:
        if VLLM_AVAILABLE:
            if logger:
                logger.info("Using vLLM engine for optimized inference")
            generate_inference_with_vllm(
                model_path=model_path,
                data_dir=data_dir,
                rewrite_method_name=rewrite_method_name,
                rules_string=rules_string,
                num_examples=num_examples,
                batch_size=batch_size,
                logger=logger
            )
        else:
            if logger:
                logger.warning("vLLM not available, falling back to transformers (slower)")
            generate_inference_with_transformers(
                model_path=model_path,
                data_dir=data_dir,
                rewrite_method_name=rewrite_method_name,
                rules_string=rules_string,
                num_examples=num_examples,
                batch_size=batch_size,
                logger=logger
            )
    except Exception as e:
        if logger:
            logger.error(f"Inference failed with error: {e}")
            import traceback
            logger.error(traceback.format_exc())
        raise

