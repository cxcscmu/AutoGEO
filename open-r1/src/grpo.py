# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import shutil  
import glob   
import re      
import datasets
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from open_r1.configs import GRPOConfig, GRPOScriptArguments
from open_r1.rewards import get_reward_funcs
from open_r1.utils import get_dataset, get_model, get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import GRPOTrainer, ModelConfig, TrlParser, get_peft_config

import torch
logger = logging.getLogger(__name__)

print(torch.version.cuda)
import torch
print("NCCL version (from torch):", torch.cuda.nccl.version())

import requests
try:
    r = requests.get("http://127.0.0.1:8000/health")
    print("✅ vLLM Server reachable:", r.status_code, r.text)
except Exception as e:
    print("❌ Cannot reach vLLM Server:", e)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class CustomSpaceSavingGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure push_in_progress attribute exists (required by transformers Trainer)
        if not hasattr(self, 'push_in_progress'):
            self.push_in_progress = None
    
    def log(self, logs, *args, **kwargs):
        """Override log method to handle wandb errors gracefully."""
        try:
            super().log(logs, *args, **kwargs)
        except Exception as e:
            # Check if it's a wandb-related error (AuthenticationError, HTTPError 502, etc.)
            error_str = str(e).lower()
            error_type_str = str(type(e)).lower()
            is_wandb_error = (
                "wandb" in error_type_str 
                or "authentication" in error_str 
                or "502" in error_str
                or "bad gateway" in error_str
            )
            
            if is_wandb_error:
                logger.warning(f"Wandb logging failed (training continues): {type(e).__name__}: {e}")
                # Filter out wandb.Table objects and try to log other metrics
                try:
                    filtered_logs = {}
                    for k, v in logs.items():
                        # Skip wandb.Table objects and None values
                        if v is None:
                            continue
                        v_type_str = str(type(v)).lower()
                        if "wandb" not in v_type_str and "table" not in v_type_str:
                            filtered_logs[k] = v
                    # Only try to log if we have non-wandb metrics
                    if filtered_logs:
                        # Use the parent's log method but skip wandb integration
                        # by temporarily removing wandb from report_to
                        original_report_to = getattr(self.args, 'report_to', None)
                        try:
                            if original_report_to and "wandb" in original_report_to:
                                # Create a copy of report_to without wandb
                                filtered_report_to = [r for r in original_report_to if r != "wandb"]
                                self.args.report_to = filtered_report_to
                                super().log(filtered_logs, *args, **kwargs)
                        finally:
                            # Restore original report_to
                            if original_report_to:
                                self.args.report_to = original_report_to
                except Exception as inner_e:
                    logger.debug(f"Failed to log filtered metrics: {inner_e}")
            else:
                # Re-raise non-wandb errors
                raise
    
    def _rotate_checkpoints(self, use_mtime=False, output_dir=None):
        super()._rotate_checkpoints(use_mtime=use_mtime, output_dir=output_dir)
        checkpoints = self._sorted_checkpoints(use_mtime=use_mtime, output_dir=output_dir)
        if len(checkpoints) <= 1:
            return
        for checkpoint_dir in checkpoints[:-1]:
            optimizer_state_paths = glob.glob(os.path.join(checkpoint_dir, "global_step*"))
            
            for path in optimizer_state_paths:
                if os.path.isdir(path):
                    logger.info(
                        f"Deleting optimizer state directory to save space: {path}"
                    )
                    try:
                        shutil.rmtree(path)
                    except OSError as e:
                        logger.error(f"Error deleting directory {path}: {e}")
    def _sorted_checkpoints(self, use_mtime=False, output_dir=None) -> list[str]:
        ordering_and_checkpoint_path = []
        glob_checkpoints = glob.glob(os.path.join(output_dir or self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-*"))

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{PREFIX_CHECKPOINT_DIR}-([0-9]+)", path)
                if regex_match is not None and regex_match.groups() is not None:
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        return checkpoints_sorted

def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            logger.info(f"✓ Checkpoint detected in output_dir: {last_checkpoint}")
        else:
            logger.info(f"  No checkpoint found in output_dir: {training_args.output_dir}")
    else:
        logger.info(f"  Output directory does not exist: {training_args.output_dir}")
    
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"✓ Will automatically resume training from checkpoint: {last_checkpoint}")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # Load the dataset
    dataset = get_dataset(script_args)

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)

    ##############
    # Load model #
    ##############
    logger.info("*** Loading model ***")
    model = get_model(model_args, training_args)

    # Get reward functions from the registry
    reward_funcs = get_reward_funcs(script_args)

    # Format into conversation
    def make_conversation(example, prompt_column: str = script_args.dataset_prompt_column):
        prompt = []

        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})

        if prompt_column not in example:
            raise ValueError(f"Dataset Question Field Error: {prompt_column} is not supported.")

        prompt.append({"role": "user", "content": example[prompt_column]})
        return {"prompt": prompt}

    dataset = dataset.map(make_conversation)

    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "") 
    os.environ["POLICY_CUDA_ID"] = "0"

    model = get_model(model_args, training_args)
    model.to("cuda:" + os.environ["POLICY_CUDA_ID"])

    #############################
    # Initialize the GRPO trainer
    #############################
    trainer = CustomSpaceSavingGRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None),
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        logger.info(f"✓ Resuming from explicitly specified checkpoint: {checkpoint}")
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
        logger.info(f"✓ Resuming from auto-detected checkpoint: {checkpoint}")
    else:
        logger.info("  Starting training from scratch (no checkpoint found)")
    
    if checkpoint is not None:
        logger.info(f"✓ Training will resume from: {checkpoint}")
    else:
        logger.info("  Training will start from the beginning")
    
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    # Align the model's generation config with the tokenizer's eos token
    # to avoid unbounded generation in the transformers `pipeline()` function
    trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
