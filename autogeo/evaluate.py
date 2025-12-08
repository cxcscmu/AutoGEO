"""
Main entry point for AutoGEO evaluation pipeline.
"""
import argparse
import os
import sys
import pprint
import requests
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from autogeo.rewriters import api_rewrite_documents, mini_rewrite_documents
from autogeo.loader import load_data, download_github_folder
from autogeo.evaluation import aggregate_json_files, autogeo_evaluation
from autogeo.config import Dataset, get_dataset_config, get_full_path, get_rewrite_method_name
from autogeo.utils.logger import get_logger


def main():
    """Main entry point for AutoGEO evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="AutoGEO evaluation pipeline for document rewriting and evaluation."
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['vanilla', 'autogeo_api', 'autogeo_mini'],
        help='Model to evaluate: vanilla, autogeo_api, or autogeo_mini'
    )
    parser.add_argument(
        '--engine_llm',
        type=str,
        required=True,
        help='Generative engine LLM: gemini-2.5-flash-lite, gpt-4o-mini, or claude-3-haiku'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=[d.value for d in Dataset],
        help='Dataset name: Researchy-GEO, E-commerce, or GEO-Bench'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Path to model checkpoint (required for autogeo_mini)'
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help='Test data directory (auto-detected from dataset if not provided)'
    )
    parser.add_argument(
        '--hf_username',
        type=str,
        default='yujiangw',
        help='Hugging Face username'
    )
    parser.add_argument(
        '--hf_dataset_name',
        type=str,
        default=None,
        help='Hugging Face dataset name (auto-detected from dataset if not provided)'
    )
    parser.add_argument(
        '--keypoint_data_dir',
        type=str,
        default=None,
        help='Directory for keypoint data (auto-detected if not provided)'
    )
    
    parser.add_argument(
        '--num_examples',
        type=lambda x: None if x.lower() == 'none' else int(x),
        default=None,
        help='Number of examples to process (default: None for all examples)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1024,
        help='Batch size for inference (default: 1024 for vLLM, 32 for transformers)'
    )
    parser.add_argument(
        '--need_geu_score',
        action='store_true',
        default=False,
        help='Enable geu score evaluation'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='logs',
        help='Directory to save log files (default: logs)'
    )
    parser.add_argument(
        '--max_workers',
        type=int,
        default=64,
        help='Maximum number of parallel workers (default: 64)'
    )
    
    parser.add_argument(
        '--repo_owner',
        type=str,
        default='cxcscmu',
        help='Owner of the GitHub repository'
    )
    parser.add_argument(
        '--repo_name',
        type=str,
        default='deepresearch_benchmarking',
        help='Name of the GitHub repository'
    )
    parser.add_argument(
        '--folder_path',
        type=str,
        default='data/Researchy-GEO/key_point',
        help='Folder path within the GitHub repository'
    )
    parser.add_argument(
        '--branch',
        type=str,
        default='main',
        help='Branch of the GitHub repository'
    )
    
    args = parser.parse_args()
    
    if args.model == 'autogeo_mini' and args.model_path is None:
        print("Error: --model_path is required when using --model autogeo_mini")
        sys.exit(1)
    
    main_logger = get_logger(
        log_dir=args.log_dir,
        task_name=f"{args.model}_{args.dataset}_{args.engine_llm}"
    )
    main_logger.info("="*80)
    main_logger.info(f"{args.model.upper()} Evaluation Started")
    main_logger.info("="*80)
    main_logger.info(f"Model: {args.model}")
    main_logger.info(f"Dataset: {args.dataset}")
    main_logger.info(f"Engine LLM: {args.engine_llm}")
    main_logger.info(f"Number of examples: {args.num_examples}")
    main_logger.info(f"Log Directory: {args.log_dir}")
    if args.model == 'autogeo_mini':
        main_logger.info(f"Model Path: {args.model_path}")
    
    dataset_config = get_dataset_config(args.dataset)
    
    if args.hf_dataset_name is None:
        args.hf_dataset_name = dataset_config['hf_dataset_name']
    if args.data_dir is None:
        args.data_dir = dataset_config['test_dir']
    if args.keypoint_data_dir is None:
        args.keypoint_data_dir = dataset_config['rl_dir']
    
    if args.model == 'vanilla':
        method_name = args.engine_llm
    else:
        method_name = get_rewrite_method_name(args.model, args.dataset, args.engine_llm)
    
    main_logger.info(f"Method Name: {method_name}")
    
    config = {
        "hf_username": args.hf_username,
        "hf_dataset_name": args.hf_dataset_name,
        "train_output_dir": dataset_config['train_dir'],
        "test_output_dir": dataset_config['test_dir'],
        "rule_candidate_output_file": get_full_path(args.dataset, 'rule_candidate_file'),
        "cold_start_output_file": get_full_path(args.dataset, 'finetune_file'),
        "inference_output_file": get_full_path(args.dataset, 'inference_file'),
        "grpo_input_output_file": get_full_path(args.dataset, 'grpo_input_file'),
        "grpo_eval_output_file": get_full_path(args.dataset, 'grpo_eval_file'),
    }
    
    try:
        main_logger.info(f"Downloading keypoint data from GitHub repository: {args.repo_owner}/{args.repo_name} ({args.branch})")
        download_github_folder(
            args.repo_owner,
            args.repo_name,
            args.folder_path,
            args.branch
        )
        main_logger.info("Keypoint data downloaded successfully!")
    except requests.exceptions.RequestException as e:
        main_logger.error(f"Error occurred during keypoint data download: {e}")
    except Exception as e:
        main_logger.error(f"An unknown error occurred during keypoint data download: {e}")
    
    main_logger.info(f"Loading data for dataset: {args.dataset}")
    load_data(config=config)
    results = {}
    
    # ========== MODEL EVALUATION ==========
    if args.model == 'vanilla':
        main_logger.info("\n" + "="*80)
        main_logger.info("Preparing Vanilla Model evaluation")
        main_logger.info("="*80)
        
        chunk_idx = 0
        processed = 0
        max_to_process = args.num_examples if args.num_examples is not None else float('inf')
        
        while processed < max_to_process:
            filename = f"{args.data_dir}/datachunk_{chunk_idx}.json"
            if not os.path.exists(filename):
                break
            
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                question_ids = sorted(list(data.keys()))
                remaining = max_to_process - processed
                if args.num_examples is not None:
                    question_ids = question_ids[:int(remaining)]
                
                modified = False
                for qid in question_ids:
                    question_data = data[qid]
                    target_id = question_data.get('target_id', 0)
                    text_list = question_data.get('text_list', [])
                    
                    if target_id < len(text_list):
                        question_data[f'{method_name}_text'] = text_list[target_id]
                        modified = True
                        processed += 1
                
                if modified:
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=4)
                    main_logger.info(f"Prepared {filename}")
                
                chunk_idx += 1
            except Exception as e:
                main_logger.warning(f"Error processing {filename}: {e}")
                chunk_idx += 1
                continue
        
        main_logger.info(f"Prepared {processed} questions for vanilla evaluation")
        
        main_logger.info("\n" + "="*80)
        main_logger.info("Evaluating Vanilla Model")
        main_logger.info("="*80)
        autogeo_evaluation(
            num_examples=args.num_examples,
            data_dir=args.data_dir,
            engine_llm=args.engine_llm,
            rewrite_method_name=method_name,
            need_geu_score=args.need_geu_score,
            logger=main_logger,
            max_workers=args.max_workers
        )
        
        results[f"Vanilla_{args.engine_llm}_geo"] = aggregate_json_files(args.data_dir, method_name + "_geo_score")
        if args.need_geu_score:
            results[f"Vanilla_{args.engine_llm}_geu"] = aggregate_json_files(args.data_dir, method_name + "_geu_score")
    
    elif args.model == 'autogeo_api':
        main_logger.info("\n" + "="*80)
        main_logger.info(f"Step 1: Rewriting documents using AutoGEO API")
        main_logger.info("="*80)
        api_rewrite_documents(
            num_examples=args.num_examples,
            data_dir=args.data_dir,
            dataset=args.dataset,
            engine_llm=args.engine_llm,
            logger=main_logger,
            max_workers=args.max_workers
        )
        
        main_logger.info("\n" + "="*80)
        main_logger.info(f"Step 2: Evaluating rewritten documents")
        main_logger.info("="*80)
        autogeo_evaluation(
            num_examples=args.num_examples,
            data_dir=args.data_dir,
            engine_llm=args.engine_llm,
            rewrite_method_name=method_name,
            need_geu_score=args.need_geu_score,
            logger=main_logger,
            max_workers=args.max_workers
        )
        
        results[f"AutoGEO_API_{method_name}_geo"] = aggregate_json_files(args.data_dir, method_name + "_geo_score")
        if args.need_geu_score:
            results[f"AutoGEO_API_{method_name}_geu"] = aggregate_json_files(args.data_dir, method_name + "_geu_score")
    
    elif args.model == 'autogeo_mini':
        main_logger.info("\n" + "="*80)
        main_logger.info(f"Step 1: Generating and rewriting documents using checkpoint")
        main_logger.info("="*80)
        
        mini_rewrite_documents(
            model_path=args.model_path,
            data_dir=args.data_dir,
            rewrite_method_name=method_name,
            dataset=args.dataset,
            engine_llm=args.engine_llm,
            num_examples=args.num_examples,
            batch_size=args.batch_size,
            logger=main_logger
        )
        
        main_logger.info("\n" + "="*80)
        main_logger.info(f"Step 2: Evaluating rewritten documents")
        main_logger.info("="*80)
        autogeo_evaluation(
            num_examples=args.num_examples,
            data_dir=args.data_dir,
            engine_llm=args.engine_llm,
            rewrite_method_name=method_name,
            need_geu_score=args.need_geu_score,
            logger=main_logger,
            max_workers=args.max_workers
        )
        
        results[f"AutoGEO_MINI_{method_name}_geo"] = aggregate_json_files(args.data_dir, method_name + "_geo_score")
        if args.need_geu_score:
            results[f"AutoGEO_MINI_{method_name}_geu"] = aggregate_json_files(args.data_dir, method_name + "_geu_score")
    
    # ========== PIPELINE COMPLETE ==========
    main_logger.info("\n" + "="*80)
    main_logger.info("EVALUATION RESULTS SUMMARY")
    main_logger.info("="*80)
    for key, value in results.items():
        main_logger.info(f"\n{key}:")
        main_logger.info(pprint.pformat(value))
    
    main_logger.info("\n" + "="*80)
    main_logger.info("Pipeline Completed Successfully")
    main_logger.info("="*80)
    main_logger.close()


if __name__ == "__main__":
    main()
