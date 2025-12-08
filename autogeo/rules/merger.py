import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Union
from .llm_client import get_llm_response 

def get_rule_merging_prompt(rules_list: List[str]) -> str:
    """Generate prompt for merging rules.
    
    Args:
        rules_list: List of rules to merge
        
    Returns:
        Formatted prompt string for rule merging
    """
    rules_text = "\n".join(f"- {rule}" for rule in rules_list)
    return f"""
[Persona]
You are an expert in Information Retrieval and Knowledge Management, specializing in defining principles for high-quality RAG source documents.

[Task]
Consolidate the given list of rules into a set of core principles. Merge semantically similar rules, eliminate duplicates, and rephrase for clarity.

[Criteria for a Good Merged Rule]
1.  **Atomic**: Expresses a single, distinct idea.
2.  **Actionable**: Provides a clear, evaluatable instruction.
3.  **Unambiguous**: Uses simple, direct language.

[Example of what to do]
- Original Rules: ["The document must be short.", "Keep text concise."]
- Good Merged Rule: ["The document should be concise, preferring shorter sentences and paragraphs."]

[Example of what to avoid (Over-merging)]
- Original Rules: ["The text needs to be factual.", "The text should provide multiple viewpoints."]
- Bad Merged Rule: ["The text must be factual and provide multiple viewpoints."] (These are two distinct ideas and should be separate rules).

[Instruction on Output Format]
Return the merged list as a single, valid JSON array of strings. Do not use ```json``` or add explanations.

[Original Rules]
{rules_text}

[Merged Rules JSON]
"""


def get_rule_filter_prompt(rule: str) -> str:
    """Generate prompt to filter rules, removing query-specific dependencies.
    
    Args:
        rule: Rule text that may contain query-specific references
        
    Returns:
        Formatted prompt string for rule filtering
    """
    return f"""
[Persona]
You are a technical writer specializing in creating context-independent documentation.

[Task]
Analyze the following rule. Your goal is to remove any part of the rule that makes it dependent on a specific user "query", "question", or "input". The rewritten rule should state a general principle.

- If the rule contains a general principle AND a reference to a query, remove only the query reference.
- If the entire rule is ONLY about how to handle a query (e.g., "The document should directly answer the query."), the principle is not general. In this case, you should return an empty string.

[Examples]
- Input Rule: "The document should provide specific facts and data relevant to the user's query."
- Output JSON: {{"modified_rule": "The document should provide specific facts and data."}}

- Input Rule: "The source must be recent and directly answer the question."
- Output JSON: {{"modified_rule": "The source must be recent."}}

- Input Rule: "The text must be authoritative."
- Output JSON: {{"modified_rule": "The text must be authoritative."}}

- Input Rule: "Directly answer the user's question."
- Output JSON: {{"modified_rule": ""}}

[Instruction on Output Format]
Return a single, valid JSON object with one key: "modified_rule". The value should be the modified string.

[Input Rule]
"{rule}"

[Output JSON]
"""

def estimate_token_count(rules_list: List[str]) -> int:
    """Estimate token count for a list of rules.
    
    Args:
        rules_list: List of rule strings
        
    Returns:
        Estimated token count (characters / 4)
    """
    total_chars = sum(len(rule) for rule in rules_list)
    return total_chars // 4


def _call_llm_with_retry(
    prompt: str,
    llm_args: dict,
    max_output_tokens: int,
    retries: int,
    retry_delay: int
) -> Optional[Union[List, Dict]]:
    """Call LLM with retry logic and JSON parsing.
    
    Args:
        prompt: Prompt text for LLM
        llm_args: Dictionary of LLM arguments
        max_output_tokens: Maximum output tokens (unused, kept for compatibility)
        retries: Number of retry attempts
        retry_delay: Delay in seconds between retries
        
    Returns:
        Parsed JSON response (list or dict), or None if all retries fail
    """
    for attempt in range(retries):
        try:
            response_text = get_llm_response(prompt=prompt, **llm_args)
            first_brace = response_text.find('{')
            first_bracket = response_text.find('[')

            if first_brace == -1 and first_bracket == -1:
                raise json.JSONDecodeError("No JSON object or array found in response", response_text, 0)
            if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
                start_pos = first_brace
                end_char = '}'
            else:
                start_pos = first_bracket
                end_char = ']'
            
            end_pos = response_text.rfind(end_char)
            if end_pos == -1 or end_pos < start_pos:
                raise json.JSONDecodeError(f"Mismatched JSON delimiters for '{end_char}'", response_text, 0)
            json_str = response_text[start_pos : end_pos + 1]
            return json.loads(json_str)

        except (json.JSONDecodeError, TypeError, AttributeError) as e:
            print(f"    Attempt {attempt + 1}/{retries} failed: {e}")
            if attempt < retries - 1:
                print(f"    Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
    
    print(f"  [CRITICAL] All {retries} retries failed for a prompt.")
    return None


def _hierarchical_merge(
    rules_list: List[str], 
    llm_args: dict, 
    max_tokens_per_chunk: int,
    max_output_tokens: int,
    retries: int, 
    retry_delay: int,
    parallel_workers: int,
) -> List[str]:
    """Hierarchically merge rules by splitting into chunks if needed.
    
    Args:
        rules_list: List of rules to merge
        llm_args: Dictionary of LLM arguments
        max_tokens_per_chunk: Maximum tokens per chunk for merging
        max_output_tokens: Maximum output tokens for LLM
        retries: Number of retry attempts
        retry_delay: Delay in seconds between retries
        parallel_workers: Maximum number of concurrent chunk merges
        
    Returns:
        List of merged rules

    Notes:
        The chunk merging phase supports lightweight parallelism controlled by
        ``parallel_workers``. Each worker performs independent LLM calls, so
        choose a value that matches API rate limits.
    """
    if not rules_list:
        return []
    print(f"\n--- STAGE 1: HIERARCHICALLY MERGING {len(rules_list)} rules ---")
    current_rules = rules_list
    
    while estimate_token_count(current_rules) > max_tokens_per_chunk:
        print(f"  Token count ({estimate_token_count(current_rules)}) exceeds limit. Starting new merge level.")
        chunks, current_chunk, current_chunk_tokens = [], [], 0
        for rule in current_rules:
            rule_tokens = estimate_token_count([rule])
            if current_chunk_tokens + rule_tokens > max_tokens_per_chunk and current_chunk:
                chunks.append(current_chunk)
                current_chunk, current_chunk_tokens = [rule], rule_tokens
            else:
                current_chunk.append(rule)
                current_chunk_tokens += rule_tokens
        if current_chunk: chunks.append(current_chunk)
        print(f"  Split {len(current_rules)} rules into {len(chunks)} chunks for this level.")

        next_level_rules = []

        def merge_single_chunk(chunk_index: int, chunk_rules: List[str]) -> List[str]:
            print(f"    Merging chunk {chunk_index + 1}/{len(chunks)} ({len(chunk_rules)} rules)...")
            prompt = get_rule_merging_prompt(chunk_rules)
            merged_chunk = _call_llm_with_retry(prompt, llm_args, max_output_tokens, retries, retry_delay)
            if merged_chunk and isinstance(merged_chunk, list):
                return merged_chunk
            print(f"    [WARNING] Failed to merge chunk {chunk_index + 1}. Skipping.")
            return []

        worker_count = max(1, min(parallel_workers, len(chunks)))
        if worker_count == 1:
            for idx, chunk in enumerate(chunks):
                next_level_rules.extend(merge_single_chunk(idx, chunk))
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_to_index = {
                    executor.submit(merge_single_chunk, idx, chunk): idx
                    for idx, chunk in enumerate(chunks)
                }
                for future in as_completed(future_to_index):
                    try:
                        next_level_rules.extend(future.result())
                    except Exception as exc:
                        idx = future_to_index[future]
                        print(f"    [WARNING] Chunk {idx + 1} raised an exception: {exc}")
        current_rules = sorted(list(set(next_level_rules)))
        print(f"  Merge level complete. Reduced to {len(current_rules)} rules.")

    print("  Performing final consolidation merge...")
    final_prompt = get_rule_merging_prompt(current_rules)
    final_merged_rules = _call_llm_with_retry(final_prompt, llm_args, max_output_tokens, retries, retry_delay)
    if not isinstance(final_merged_rules, list):
        print("  [WARNING] Final merge failed. Using pre-merge rules.")
        return current_rules
    print(f"--- HIERARCHICAL MERGE COMPLETE: {len(final_merged_rules)} consolidated rules ---")
    return sorted(list(set(final_merged_rules)))


def _filter_rules(
    rules_list: List[str],
    llm_args: dict,
    max_output_tokens: int,
    retries: int,
    retry_delay: int
) -> List[str]:
    """Filter rules to remove query-specific dependencies and keep only general principles.
    
    Args:
        rules_list: List of rules to filter
        llm_args: Dictionary of LLM arguments
        max_output_tokens: Maximum output tokens for LLM
        retries: Number of retry attempts
        retry_delay: Delay in seconds between retries
        
    Returns:
        List of filtered, query-agnostic rules
    """
    if not rules_list:
        return []
    print(f"\n--- STAGE 2: FILTERING RULES from {len(rules_list)} rules ---")
    filtered_rules = []
    for i, rule in enumerate(rules_list):
        print(f"  Filtering rule {i+1}/{len(rules_list)}...")
        prompt = get_rule_filter_prompt(rule)
        response_json = _call_llm_with_retry(prompt, llm_args, max_output_tokens, retries, retry_delay)
        
        if isinstance(response_json, dict) and "modified_rule" in response_json:
            modified_rule = response_json["modified_rule"]
            if modified_rule and isinstance(modified_rule, str) and modified_rule.strip():
                filtered_rules.append(modified_rule.strip())
        else:
             print(f"  [WARNING] Failed to process or got empty result for rule: '{rule}'")

    cleaned_rules = sorted(list(set(filtered_rules)))
    print(f"--- RULE FILTERING COMPLETE: {len(cleaned_rules)} filtered rules remain ---")
    return cleaned_rules


def get_merged_rules(
    rules_list: List[str], 
    *, 
    llm_args: dict,
    max_tokens_per_chunk: int = 20000,
    max_output_tokens: int = 8192,
    retries: int = 3, 
    retry_delay: int = 5,
    parallel_workers: int = 16,
    logger=None
) -> Dict[str, Union[List[str], List[List[str]]]]:
    """Merge and process rules through hierarchical merging, filtering, and contradiction detection.
    
    Args:
        rules_list: List of rules to merge
        llm_args: Dictionary of LLM arguments (model, api keys, etc.)
        max_tokens_per_chunk: Maximum tokens per chunk for merging (default: 20000)
        max_output_tokens: Maximum output tokens for LLM (default: 8192)
        retries: Number of retry attempts (default: 3)
        retry_delay: Delay in seconds between retries (default: 5)
        parallel_workers: Max number of concurrent merges per level (default: 16)
        logger: Optional logger instance for logging
        
    Returns:
        Dictionary with keys:
            - merged_rules: List of rules after hierarchical merging
            - filtered_rules: List of rules after filtering

    Notes:
        The ``parallel_workers`` arg can also be overridden by passing
        ``merge_workers`` inside ``llm_args`` for finer runtime control.
    """
    llm_args["model"] = llm_args.get("model", "gemini-2.5-pro")
    if not rules_list:
        if logger:
            logger.warning("No rules provided for merging")
        return {"merged_rules": [], "filtered_rules": []}
    
    if logger:
        logger.info("\n--- STARTING RULE PROCESSING PIPELINE ---")
        logger.info(f"Total rules to process: {len(rules_list)}")
    else:
        print("\n--- STARTING RULE PROCESSING PIPELINE ---")
    
    if logger:
        logger.info("Stage 1: Hierarchical merging...")
    worker_override = llm_args.get("merge_workers")
    if isinstance(worker_override, int) and worker_override > 0:
        parallel_workers = worker_override

    fully_merged_rules = _hierarchical_merge(
        rules_list,
        llm_args,
        max_tokens_per_chunk,
        max_output_tokens,
        retries,
        retry_delay,
        parallel_workers,
    )
    if logger:
        logger.info(f"Hierarchical merging completed: {len(fully_merged_rules)} merged rules")
    
    if logger:
        logger.info("Stage 2: Filtering rules...")
    filtered_rules = _filter_rules(
        fully_merged_rules, llm_args, 512, retries, retry_delay
    )
    if logger:
        logger.info(f"Filtering completed: {len(filtered_rules)} filtered rules")

    if logger:
        logger.info(f"Stage 3: Identifying contradictions among {len(filtered_rules)} rules...")
    else:
        print(f"\n--- STAGE 3: IDENTIFYING CONTRADICTIONS among {len(filtered_rules)} final rules ---")
    
    result = {
        "merged_rules": fully_merged_rules,
        "filtered_rules": filtered_rules
    }
    
    if logger:
        logger.info("--- PIPELINE COMPLETE ---")
        logger.info(f"Final filtered rules: {len(result['filtered_rules'])}")
    else:
        print("\n--- PIPELINE COMPLETE ---")
        print(f"Final filtered rules: {len(result['filtered_rules'])}")
    
    return result
