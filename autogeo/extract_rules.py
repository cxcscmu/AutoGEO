"""
Main entry point for rule extraction.
"""
import os
import sys
import json
import argparse
import threading
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

# Add parent directory to path to import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from autogeo.config import Dataset, get_dataset_config, get_full_path
from autogeo.utils.logger import get_logger
from autogeo.loader.data_loader import load_data
from autogeo.rules import (
    load_engine_preference_dataset,
    prepare_example,
    get_explanation_response,
    get_extracted_rules,
    get_merged_rules,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv("keys.env")


def evaluate_example(example: dict, llm_args: dict, logger=None) -> dict:
    """Evaluate a single example to extract rules.
    
    Args:
        example: Example dictionary with query, documents, and winner
        llm_args: Dictionary of LLM arguments
        logger: Optional logger instance for logging
        
    Returns:
        Dictionary with extracted information
    """
    example_id = example.get("id", "unknown")
    if logger:
        logger.debug(f"Processing example {example_id}...")
    
    explanation_text, _ = get_explanation_response(
        query=example["query"],
        document_a=example["document_a"],
        document_b=example["document_b"],
        winner=example["winner"],
        llm_args=llm_args,
    )
    
    if logger:
        logger.debug(f"Explanation generated for example {example_id} ({len(explanation_text)} chars)")
    
    extracted_rules = get_extracted_rules(
        explanation_text, winner=example["winner"], llm_args=llm_args
    )
    
    if logger:
        num_rules = len(extracted_rules) if isinstance(extracted_rules, list) else 0
        logger.debug(f"Extracted {num_rules} rules from example {example_id}")
    
    return {
        "id": example["id"],
        "query": example["query"],
        "winner": example["winner"],
        "explanation": explanation_text,
        "extracted_rules": extracted_rules,
        "good_document": example["good_document_content"],
        "bad_document": example["bad_document_content"],
    }


def main():
    parser = argparse.ArgumentParser(description="Extract rule extraction samples from a dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="Researchy-GEO",
        choices=[d.value for d in Dataset],
        help="Name of the dataset: Researchy-GEO, E-commerce, or GEO-Bench"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to the rule extraction samples JSON dataset (auto-detected from dataset if not provided)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save the results (auto-generated if not provided)"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=None,
        help="Number of examples to process from the dataset (optional, processes all by default)"
    )
    parser.add_argument(
        "--engine_llm",
        type=str,
        required=True,
        help="LLM that generative engine uses (e.g., 'gemini-2.5-flash-lite')"
    )
    parser.add_argument(
        "--google_api_key",
        type=str,
        default=os.getenv("GOOGLE_API_KEY"),
        help="Google API key for Gemini"
    )
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default=os.getenv("OPENAI_API_KEY"),
        help="OpenAI API key"
    )
    parser.add_argument(
        "--anthropic_api_key",
        type=str,
        default=os.getenv("ANTHROPIC_API_KEY"),
        help="Anthropic API key"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=64,
        help="Number of processes for parallel evaluation (default: 64)"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory to save log files (default: logs)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available"
    )
    args = parser.parse_args()
    
    # Auto-detect dataset_path if not provided
    if args.dataset_path is None:
        args.dataset_path = get_full_path(args.dataset, 'rule_candidate_file')
        if not os.path.exists(args.dataset_path):
            print(f"Rule candidate file not found: {args.dataset_path}")
            print("Attempting to download data automatically...")
            
            # Get dataset configuration
            dataset_config = get_dataset_config(args.dataset)
            
            # Build configuration for dataloader
            config = {
                "hf_username": "yujiangw",
                "hf_dataset_name": dataset_config["hf_dataset_name"],
                "train_output_dir": dataset_config["train_dir"],
                "test_output_dir": dataset_config["test_dir"],
                "rule_candidate_output_file": get_full_path(args.dataset, 'rule_candidate_file'),
                "cold_start_output_file": get_full_path(args.dataset, 'finetune_file'),
                "inference_output_file": get_full_path(args.dataset, 'inference_file'),
                "grpo_input_output_file": get_full_path(args.dataset, 'grpo_input_file'),
                "grpo_eval_output_file": get_full_path(args.dataset, 'grpo_eval_file'),
            }
            
            # Download data
            try:
                load_data(config=config)
                print("Data download completed!")
            except Exception as e:
                raise FileNotFoundError(
                    f"Failed to automatically download data: {e}\n"
                    f"Rule candidate file not found: {args.dataset_path}. "
                    f"Please check your internet connection or specify --dataset_path manually"
                ) from e
            
            # Verify file exists after download
            if not os.path.exists(args.dataset_path):
                raise FileNotFoundError(
                    f"Rule candidate file still not found after download: {args.dataset_path}. "
                    f"Please check the dataset configuration or specify --dataset_path manually"
                )
    
    # Initialize logger
    task_name = f"rule_extraction_{args.dataset}_{args.engine_llm.replace('-', '_')}"
    logger = get_logger(log_dir=args.log_dir, task_name=task_name)
    logger.info("="*80)
    logger.info("Rule Extraction Pipeline Started")
    logger.info("="*80)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"LLM Model: {args.engine_llm}")
    logger.info(f"Number of processes: {args.max_workers}")
    
    # Auto-generate output_dir if not provided
    if args.output_dir is None:
        # Use full engine_llm for folder name (e.g., gemini-2.5-flash-lite)
        results_folder = f"data/{args.dataset}/rule_sets/{args.engine_llm}"
    else:
        results_folder = args.output_dir
    
    os.makedirs(results_folder, exist_ok=True)
    logger.info(f"Results will be saved to: {results_folder}")
    
    # Checkpoint and result file paths
    checkpoint_file = os.path.join(results_folder, "checkpoint.json")
    explanation_json = os.path.join(results_folder, "explanation_results.json")
    individual_results_dir = os.path.join(results_folder, "individual_results")
    
    llm_args = {
        "model": args.engine_llm,
        "google_api_key": args.google_api_key,
        "openai_api_key": args.openai_api_key,
        "anthropic_api_key": args.anthropic_api_key,
    }
    
    # Load and process dataset
    logger.info(f"Loading dataset from: {args.dataset_path}")
    full_dataset = load_engine_preference_dataset(args.dataset_path)
    if args.num_examples:
        dataset_to_process = full_dataset.select(range(min(args.num_examples, len(full_dataset))))
    else:
        dataset_to_process = full_dataset
    
    total_examples = len(dataset_to_process)
    logger.info(f"Total examples to process: {total_examples}")
    
    # Check for checkpoint/resume - always check, not just when --resume is set
    processed_indices = set()
    
    # Check main JSON file
    if os.path.exists(explanation_json):
        logger.info(f"Found existing results file: {explanation_json}")
        try:
            with open(explanation_json, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            main_ids = {item.get("id") for item in existing_results if "id" in item}
            processed_indices.update(main_ids)
            logger.info(f"Found {len(main_ids)} already processed examples in main file")
        except Exception as e:
            logger.warning(f"Failed to load main checkpoint: {e}. Checking individual results...")
    
    # Also check individual results directory
    if os.path.exists(individual_results_dir):
        try:
            individual_files = [f for f in os.listdir(individual_results_dir) if f.endswith('.json')]
            individual_ids = {f.replace('.json', '') for f in individual_files}
            if individual_ids:
                processed_indices.update(individual_ids)
                logger.info(f"Found {len(individual_ids)} individual result files")
        except Exception as e:
            logger.warning(f"Failed to read individual results directory: {e}")
    
    # Check checkpoint file for additional info
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            checkpoint_ids = set(checkpoint_data.get("processed_ids", []))
            if checkpoint_ids:
                processed_indices.update(checkpoint_ids)
                logger.info(f"Found {len(checkpoint_ids)} processed IDs in checkpoint file")
                logger.info(f"Checkpoint shows {checkpoint_data.get('processed_count', 0)}/{checkpoint_data.get('total_count', 0)} examples processed")
        except Exception as e:
            logger.debug(f"Could not read checkpoint file: {e}")
    
    if processed_indices:
        logger.info(f"Total {len(processed_indices)} examples already processed, will skip them")
        if not args.resume:
            logger.info("Note: Use --resume flag to explicitly enable resume mode (auto-detection is enabled by default)")
    else:
        logger.info("No existing results found, starting from scratch")
    
    # Prepare dataset
    logger.info("Preparing samples for rule extraction...")
    try:
        prepared_dataset = dataset_to_process.map(
            prepare_example, 
            num_proc=args.max_workers,
            desc="Preparing samples for rule extraction"
        )
    except RuntimeError as e:
        if "subprocess" in str(e).lower() or "multiprocessing" in str(e).lower():
            logger.warning(f"Multiprocessing failed: {e}")
            logger.info("Retrying with single process...")
            prepared_dataset = dataset_to_process.map(
                prepare_example, 
                num_proc=1,
                desc="Preparing samples for rule extraction (single process)"
            )
        else:
            raise
    
    # Process examples with checkpoint support
    # Always skip already processed examples if found (auto-resume)
    if processed_indices:
        logger.info(f"Skipping {len(processed_indices)} already processed examples (auto-resume enabled)")
        # Filter out already processed examples
        dataset_to_eval = prepared_dataset.filter(
            lambda x: x.get("id") not in processed_indices
        )
        remaining = len(dataset_to_eval)
        logger.info(f"Remaining examples to process: {remaining}")
    else:
        dataset_to_eval = prepared_dataset
        remaining = len(dataset_to_eval)
    
    if remaining > 0:
        logger.info(f"Processing {remaining} examples...")
        
        # Load existing results if any were found (auto-resume)
        all_results = []
        if processed_indices and os.path.exists(explanation_json):
            try:
                with open(explanation_json, 'r', encoding='utf-8') as f:
                    all_results = json.load(f)
                logger.info(f"Loaded {len(all_results)} existing results from previous run")
            except Exception as e:
                logger.warning(f"Failed to load existing results: {e}. Will start fresh but skip processed examples.")
                all_results = []
        
        # Create individual results directory for tracking
        individual_results_dir = os.path.join(results_folder, "individual_results")
        os.makedirs(individual_results_dir, exist_ok=True)
        logger.info(f"Individual results will be saved to: {individual_results_dir}")
        
        # File lock for thread-safe writing
        save_lock = threading.Lock()
        save_counter = [0]  # Use list for mutable counter in nested function
        
        def save_result(result: dict):
            """Save individual result and update main file."""
            with save_lock:
                # Save individual result file
                example_id = result.get("id", "unknown")
                individual_file = os.path.join(individual_results_dir, f"{example_id}.json")
                with open(individual_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                # Add to all_results
                all_results.append(result)
                save_counter[0] += 1
                
                # Save main file every 10 examples or at the end
                if save_counter[0] % 10 == 0 or save_counter[0] >= remaining:
                    df_temp = pd.DataFrame(all_results)
                    df_temp.to_json(explanation_json, orient="records", indent=4)
                    
                    # Save checkpoint metadata
                    checkpoint_data = {
                        "processed_count": len(all_results),
                        "total_count": total_examples,
                        "processed_ids": [item.get("id") for item in all_results if "id" in item],
                        "timestamp": datetime.now().isoformat(),
                        "last_saved_example": example_id
                    }
                    with open(checkpoint_file, 'w', encoding='utf-8') as f:
                        json.dump(checkpoint_data, f, indent=2)
        
        # Modified evaluate function that saves after each example
        def evaluate_and_save_example(example: dict) -> dict:
            """Evaluate example and save result immediately."""
            example_id = example.get("id", "unknown")
            # Skip if already processed (shouldn't happen due to filter, but double-check)
            if example_id in processed_indices:
                logger.debug(f"Skipping already processed example: {example_id}")
                # Load existing result if available
                individual_file = os.path.join(individual_results_dir, f"{example_id}.json")
                if os.path.exists(individual_file):
                    try:
                        with open(individual_file, 'r', encoding='utf-8') as f:
                            return json.load(f)
                    except Exception as e:
                        logger.warning(f"Failed to load existing result for {example_id}: {e}")
                return {}
            
            result = evaluate_example(example, llm_args, logger=logger)
            save_result(result)
            return result
        
        # Process all remaining examples
        # Each example is saved immediately after processing, so no need to batch
        logger.info(f"Starting processing of {remaining} examples with {args.max_workers} processes...")
        logger.info("Each example will be saved immediately after processing.")
        
        try:
            # Process all examples - each one saves immediately via save_result
            all_processed_results = dataset_to_eval.map(
                evaluate_and_save_example, 
                num_proc=args.max_workers 
            )
            # Convert to list to ensure all processing completes
            all_processed_results.to_list()
            
            # Final save to ensure everything is up to date
            with save_lock:
                df_temp = pd.DataFrame(all_results)
                df_temp.to_json(explanation_json, orient="records", indent=4)
                
                checkpoint_data = {
                    "processed_count": len(all_results),
                    "total_count": total_examples,
                    "processed_ids": [item.get("id") for item in all_results if "id" in item],
                    "timestamp": datetime.now().isoformat(),
                    "status": "completed"
                }
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint_data, f, indent=2)
                
                logger.info(f"Processing completed: {len(all_results)}/{total_examples} examples saved")
                    
        except Exception as e:
            logger.error(f"Error during processing: {e}")
            # Save current progress before raising
            with save_lock:
                df_temp = pd.DataFrame(all_results)
                df_temp.to_json(explanation_json, orient="records", indent=4)
                checkpoint_data = {
                    "processed_count": len(all_results),
                    "total_count": total_examples,
                    "processed_ids": [item.get("id") for item in all_results if "id" in item],
                    "timestamp": datetime.now().isoformat(),
                    "status": "error",
                    "error": str(e)
                }
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint_data, f, indent=2)
                logger.warning(f"Progress saved before error: {len(all_results)}/{total_examples} examples")
            raise
        
        # After processing, re-aggregate all results from individual files
        # This is necessary because in multi-process mode, each process has its own all_results copy
        logger.info("Re-aggregating results from individual files...")
        all_results = []
        if os.path.exists(individual_results_dir):
            individual_files = [f for f in os.listdir(individual_results_dir) if f.endswith('.json')]
            logger.info(f"Found {len(individual_files)} individual result files, aggregating...")
            for filename in individual_files:
                individual_file = os.path.join(individual_results_dir, filename)
                try:
                    with open(individual_file, 'r', encoding='utf-8') as f:
                        result = json.load(f)
                        if result:  # Only add non-empty results
                            all_results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to load {filename}: {e}")
            logger.info(f"Aggregated {len(all_results)} results from individual files")
            
            # Update main results file with aggregated data
            if all_results:
                df_temp = pd.DataFrame(all_results)
                df_temp.to_json(explanation_json, orient="records", indent=4)
                logger.info(f"Updated main results file with {len(all_results)} results")
        else:
            # Fallback to loading from main file if individual directory doesn't exist
            if os.path.exists(explanation_json):
                with open(explanation_json, 'r', encoding='utf-8') as f:
                    all_results = json.load(f)
                logger.info(f"Loaded {len(all_results)} results from main file")
        
        df_explanation = pd.DataFrame(all_results)
    else:
        logger.info("All examples already processed. Aggregating results from individual files...")
        # Re-aggregate all results from individual files to ensure completeness
        all_results = []
        if os.path.exists(individual_results_dir):
            individual_files = [f for f in os.listdir(individual_results_dir) if f.endswith('.json')]
            logger.info(f"Found {len(individual_files)} individual result files, aggregating...")
            for filename in individual_files:
                individual_file = os.path.join(individual_results_dir, filename)
                try:
                    with open(individual_file, 'r', encoding='utf-8') as f:
                        result = json.load(f)
                        if result:  # Only add non-empty results
                            all_results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to load {filename}: {e}")
            logger.info(f"Aggregated {len(all_results)} results from individual files")
        else:
            # Fallback to loading from main file if individual directory doesn't exist
            if os.path.exists(explanation_json):
                with open(explanation_json, 'r', encoding='utf-8') as f:
                    all_results = json.load(f)
                logger.info(f"Loaded {len(all_results)} results from main file")
        df_explanation = pd.DataFrame(all_results)
    
    # Save final results
    explanation_tsv = os.path.join(results_folder, "explanation_results.tsv")
    df_explanation.to_csv(explanation_tsv, sep="\t", index=False)
    df_explanation.to_json(explanation_json, orient="records", indent=4)
    logger.info(f"Final results saved:")
    logger.info(f"  - Main results: {explanation_json}")
    logger.info(f"  - TSV format: {explanation_tsv}")
    logger.info(f"  - Individual results: {individual_results_dir} ({len(os.listdir(individual_results_dir)) if os.path.exists(individual_results_dir) else 0} files)")
    logger.info(f"Total examples processed: {len(df_explanation)}")
    
    # Extract and merge rules
    logger.info("\n" + "="*80)
    logger.info("Step 2: Extracting and Merging Rules")
    logger.info("="*80)
    
    merged_rules_filename = os.path.join(results_folder, "merged_rules.json")
    
    # Check if merged rules already exist and are valid
    merged_rules = None
    if os.path.exists(merged_rules_filename):
        try:
            with open(merged_rules_filename, 'r', encoding='utf-8') as f:
                existing_rules = json.load(f)
            # Validate: check if it has filtered_rules and it's not empty
            if isinstance(existing_rules, dict) and 'filtered_rules' in existing_rules:
                if isinstance(existing_rules['filtered_rules'], list) and len(existing_rules['filtered_rules']) > 0:
                    merged_rules = existing_rules
                    logger.info(f"Found existing merged rules file: {merged_rules_filename}")
                    logger.info(f"Final rules count: {len(merged_rules.get('filtered_rules', []))}")
                    logger.info("Skipping rule merging process as merged rules already exist and are valid.")
                else:
                    logger.warning(f"Existing merged_rules.json found but 'filtered_rules' is empty or invalid. Will re-merge.")
            else:
                logger.warning(f"Existing merged_rules.json found but missing 'filtered_rules' field. Will re-merge.")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load existing merged_rules.json: {e}. Will re-merge.")
    
    # Only merge if we don't have valid merged rules
    if merged_rules is None:
        all_rules = [rule for rules in df_explanation["extracted_rules"] if isinstance(rules, list) for rule in rules]
        unique_rules = list(set(all_rules))
        logger.info(f"Found {len(all_rules)} rules in total, with {len(unique_rules)} unique rules before merging.")
        
        # Use stronger model for merging
        merge_llm_args = {
            "model": "gemini-2.5-pro",
            "google_api_key": args.google_api_key,
            "openai_api_key": args.openai_api_key,
            "anthropic_api_key": args.anthropic_api_key,
        }
        
        if unique_rules:
            logger.info(f"Starting rule merging process with {len(unique_rules)} unique rules...")
            logger.info("This may take a while as it involves multiple stages: merging, filtering, and contradiction detection.")
            
            merged_rules = get_merged_rules(unique_rules, llm_args=merge_llm_args, logger=logger)
            
            with open(merged_rules_filename, "w", encoding="utf-8") as f:
                json.dump(merged_rules, f, indent=4, ensure_ascii=False)
            
            logger.info(f"Merged rules saved to: {merged_rules_filename}")
            logger.info(f"Final rules count: {len(merged_rules.get('filtered_rules', []))}")
        else:
            logger.warning("No rules were extracted to merge.")
    
    logger.info("\n" + "="*80)
    logger.info("Rule Extraction Pipeline Completed")
    logger.info("="*80)
    logger.close()


if __name__ == "__main__":
    main()

