# coding=utf-8
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

"""Reward functions for GRPO training."""
import statistics
import asyncio
import json
import math
import re
from functools import partial, update_wrapper
import numpy as np
from typing import Callable, Dict, Literal, Optional, Tuple, Any

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from .utils.code_providers import get_provider
from .utils.competitive_programming import (
    SubtaskResult,
    add_includes,
    get_morph_client_from_env,
    get_piston_client_from_env,
)
from .utils.competitive_programming import patch_code as cf_patch_code
from .utils.competitive_programming import score_submission as cf_score_submission
from .utils.competitive_programming import score_subtask
from .GEO.evaluation_metrics import extract_citations_new, impression_wordpos_count_simple,impression_pos_count_simple,impression_word_count_simple
import time
import os
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
import nltk
nltk.download('punkt', quiet=True) 
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

from openai import OpenAI, APIError
import openai

dotenv_file_path = "../keys.env" 

was_loaded = load_dotenv(dotenv_path=dotenv_file_path)

if was_loaded:
    print(f"Successfully loaded environment variables from: {dotenv_file_path}")
else:
    print(f"Warning: Could not find or load .env file at: {dotenv_file_path}")

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
openai.api_key = os.getenv("OPENAI_API_KEY")

query_prompt = """Write an accurate and concise answer for the given user question, using _only_ the provided summarized web search results. The answer should be correct, high-quality, and written by an expert using an unbiased and journalistic tone. The user's language of choice such as English, Français, Español, Deutsch, or 日本語 should be used. The answer should be informative, interesting, and engaging. The answer's logic and reasoning should be rigorous and defensible. Every sentence in the answer should be _immediately followed_ by an in-line citation to the search result(s). The cited search result(s) should fully support _all_ the information in the sentence. Search results need to be cited using [index]. When citing several search results, use [1][2][3] format rather than [1, 2, 3]. You can use multiple search results to respond comprehensively while avoiding irrelevant search results.

Question: {query}

Search Results:
{source_text}
"""

TAG_PATTERNS = [
    (re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE), ""),
    (re.compile(r"<[｜\|]?\s*[\|｜]?\s*[^>]*?[\|｜]\s*>"), ""),
    (re.compile(r"</?tool_(?:call|response)[^>]*>", flags=re.IGNORECASE), ""),
]

def extract_formal_text(raw: str) -> str:
    cleaned = raw
    for pattern, repl in TAG_PATTERNS:
        cleaned = pattern.sub(repl, cleaned)
    clean_text = cleaned.strip()
    if "</think>" in clean_text:
        end_think_pos = clean_text.find("</think>")
        clean_text = clean_text[end_think_pos + len("</think>"):].lstrip("\n")
    return clean_text

def create_batch_prompt_judge(simplified_key_points: list, answer: str) -> str:
    key_points_json_str = json.dumps(simplified_key_points, indent=4, ensure_ascii=False)
    return f"""You are given a **JSON array of Key Points** and a **Report**.

For **each** Key Point in the JSON array, your job is to determine whether the Report:
- **Supported** the Key Point: means the Report contains information that supports the Key Point.
- **Omitted** the Key Point: means the Report does not mention or cover the Key Point.
- **Contradicted** the Key Point: means the Report says something that disagrees with or negates the Key Point.

Carefully read each Key Point and the Report.

Return your answer as a **single JSON object**. The keys of this object must be the `point_number` from the input Key Points, converted to a string. The value for each key must be another JSON object with two fields:
- "label": One of "Supported", "Omitted", or "Contradicted".
- "justification": A brief explanation for your label.

For example, your response should look like this:
{{
  "1": {{
    "label": "Supported",
    "justification": "The report's first section directly defines this term."
  }},
  "2": {{
    "label": "Omitted",
    "justification": "The report discusses data misuse causes but does not mention this specific aspect."
  }}
}}

Respond **only** with the JSON object. Do not add any commentary, text, or markdown formatting like ```json.

---

Key Points:
{key_points_json_str}

---

Report:
{answer}
"""

model = genai.GenerativeModel('gemini-2.5-flash-lite') 
client = OpenAI()

def generate_answer(model, query: str, sources: list[str], temperature: float = 0.5) -> str:
    source_text = '\n\n'.join(
        [f'### Source {i}:\n{s}\n\n' for i, s in enumerate(sources)]
    )
    prompt = query_prompt.format(query=query, source_text=source_text)
    
    max_retries = 5
    delay = 1  

    for attempt in range(max_retries):
        try:
            resp = model.generate_content(prompt, generation_config={
                "temperature": temperature,
                "candidate_count": 1
            })
            if not resp.candidates:
                print(f"Prompt was blocked for query: '{query[:50]}...'. Returning empty string.")
                return ""
            return resp.text

        except (google_exceptions.ResourceExhausted, google_exceptions.ServiceUnavailable) as e:
            print(f"API call to 'generate_answer' failed on attempt {attempt + 1}/{max_retries}. Error: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  
        except Exception as e:
            print(f"An unexpected non-retryable error occurred in 'generate_answer': {e}")
            break 
    print(f"API call to 'generate_answer' failed after all {max_retries} retries. Returning a safe default (empty string).")
    return "" 


QUALITY_RULES = [
    "Attribute all factual claims to credible, authoritative sources with clear citations.",
    "Cover the topic comprehensively, addressing all key aspects and sub-topics.",
    "Ensure information is factually accurate and verifiable.",
    "Focus exclusively on the topic, eliminating irrelevant information, navigational links, and advertisements.",
    "Maintain a neutral, objective tone, avoiding promotional language, personal opinions, and bias.",
    "Maintain high-quality writing, free from grammatical errors, typos, and formatting issues.",
    "Present a balanced perspective on complex topics, acknowledging multiple significant viewpoints or counter-arguments.",
    "Present information as a self-contained unit, not requiring external links for core understanding.",
    "Provide clear, specific, and actionable steps.",
    "Provide explanatory depth by clarifying underlying causes, mechanisms, and context ('how' and 'why').",
    "State the key conclusion at the beginning of the document.",
    "Structure content logically with clear headings, lists, and paragraphs to ensure a cohesive flow.",
    "Substantiate claims with specific, concrete details like data, statistics, or named examples.",
    "Use clear and concise language, avoiding jargon, ambiguity, and verbosity.",
    "Use current information, reflecting the latest state of knowledge."
]


def create_batch_prompt_for_rules(simplified_rules: list, text: str) -> str:
    rules_json_str = json.dumps(simplified_rules, indent=4, ensure_ascii=False)

    return f"""You are an expert editor tasked with evaluating a document based on a set of quality rules.

You are given a **JSON array of Quality Rules** and a **Text Document**.

For **each** rule in the JSON array, your job is to determine whether the Text Document:
- **Followed** the rule: The document successfully adheres to the principle described in the rule.
- **Violated** the rule: The document fails to meet the standard of the rule.

Carefully read each rule and the Text Document.

Return your answer as a **single JSON object**. The keys of this object must be the `rule_number` from the input rules, converted to a string. The value for each key must be another JSON object with two fields:
- "label": One of "Followed" or "Violated".
- "justification": A brief explanation for your label, explaining why the document followed or violated the rule.

Example Response Format:
{{
  "1": {{
    "label": "Violated",
    "justification": "The document makes several factual claims without providing any citations or sources."
  }},
  "2": {{
    "label": "Followed",
    "justification": "The document covers the main aspects of the topic as requested."
  }}
}}

Respond **only** with the JSON object. Do not add any other text or markdown formatting.

---

Quality Rules:
{rules_json_str}

---

Text Document:
{text}
"""


def calculate_rule_following_score(text: str, client: OpenAI) -> float:
    if not text or not text.strip():
        return 0.0

    simplified_rules = [{"rule_number": i + 1, "rule_content": rule} for i, rule in enumerate(QUALITY_RULES)]
    batch_prompt = create_batch_prompt_for_rules(simplified_rules, text)
    chat_pattern = [
        {"role": "system", "content": "You are a helpful assistant that responds strictly in the requested JSON object format."},
        {"role": "user", "content": batch_prompt}
    ]

    judgments_dict = None 
    max_retries = 5
    delay = 1

    for attempt in range(max_retries):
        try:
            _response = client.chat.completions.create(
                model="gpt-4o-mini", max_tokens=3000, messages=chat_pattern,
                response_format={"type": "json_object"},
            )
            response_content = _response.choices[0].message.content
            judgments_dict = json.loads(response_content)
            break 
        except (APIError, json.JSONDecodeError) as e:
            print(f"API call for rule judgment failed on attempt {attempt + 1}/{max_retries}. Error: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
        except Exception as e:
            print(f"An unexpected non-retryable error occurred during rule judgment: {e}")
            break

    if not isinstance(judgments_dict, dict):
        print(f"Could not get rule judgments after {max_retries} attempts. Assigning score 0.0.")
        return 0.0 
    followed_count = sum(1 for j in judgments_dict.values() if isinstance(j, dict) and j.get('label') == "Followed")
    return followed_count / len(QUALITY_RULES) if QUALITY_RULES else 0.0


def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    
    scores_array = np.array(scores, dtype=np.float32)
    mean = np.mean(scores_array)
    std = np.std(scores_array)
    
    if std == 0:
        return [0.0] * len(scores)
    
    normalized_scores = (scores_array - mean) / std
    return normalized_scores.tolist()


def wordcountpos_reward(
    completions: list[list[dict[str, str]]],
    solution:    list[str],
    max_workers: int = 16,
    model_name:  str = 'gemini-2.5-flash-lite',
    use_scores: Tuple[str, ...] = ('geo', 'rule', 'keypoint'),
    score_weights: Optional[Dict[str, float]] = {'geo': 1, 'rule': 1, 'keypoint': 1},
    keypoint_support_threshold: float = 0.8,
    **kwargs
)  -> list[float]:
    
    if score_weights is None:
        score_weights = {score: 1.0 for score in use_scores}

    try:
        with open("../data/Researchy-GEO/RL/Researchy_grpo_eval.json", "r", encoding="utf-8") as f:
            refdata = json.load(f)
    except FileNotFoundError:
        return [0.0] * len(completions)

    contents = [extract_formal_text(c[0]["content"]) for c in completions]
    ref_ids = [str(s) for s in solution] # Ensure all solution IDs are strings
    ref_items = [refdata.get(i) for i in ref_ids]

    valid_tasks = []
    original_indices = []
    for i, content in enumerate(contents):
        if i < len(ref_items) and ref_items[i]:
            valid_tasks.append((content, ref_items[i], ref_ids[i]))
            original_indices.append(i)

    if not valid_tasks:
        print("[REWARD_DEBUG] No valid tasks found in this batch. Returning all zeros.", flush=True)
        return [0.0] * len(completions)

    def task(content: str, ref_entry: dict, ref_id: str):
        raw_scores = {}
 
        if 'rule' in use_scores:
            raw_scores['rule'] = calculate_rule_following_score(content, client)

        if 'keypoint' in use_scores:
            keypoint_score = 0.0
            if "ori_keypoint_dict" in ref_entry:
                original_points = ref_entry['ori_keypoint_dict'].get('points', [])
                if original_points:
                    simplified_points = [{"point_number": p["point_number"], "point_content": p["point_content"]} for p in original_points]
                    batch_prompt = create_batch_prompt_judge(simplified_points, content)
                    chat_pattern = [{"role": "system", "content": "..."}, {"role": "user", "content": batch_prompt}]
                    
                    judgments_dict = None
                    max_retries, delay = 5, 1
                    for attempt in range(max_retries):
                        try:
                            _response = client.chat.completions.create(model="gpt-4o-mini", messages=chat_pattern, response_format={"type": "json_object"})
                            judgments_dict = json.loads(_response.choices[0].message.content)
                            break
                        except (APIError, json.JSONDecodeError) as e:
                            if attempt < max_retries - 1: time.sleep(delay); delay *= 2
                        except Exception: break
                    
                    if judgments_dict:
                        supported_count = sum(1 for p in original_points if judgments_dict.get(str(p['point_number']), {}).get('label') == "Supported")
                        support_ratio = supported_count / len(original_points) if original_points else 0
                        if support_ratio >= keypoint_support_threshold: keypoint_score = 1.0
            raw_scores['keypoint'] = keypoint_score

        if 'geo' in use_scores:
            new_resp = generate_answer(model, ref_entry["query"], ref_entry["text_list"])
            geo_score_component = 0.0
            if new_resp.strip():
                cites = extract_citations_new(new_resp)
                tid = ref_entry["target_id"]
                scores_wp = impression_wordpos_count_simple(cites); new_wp = scores_wp[tid] if tid < len(scores_wp) else 0.0
                scores_pos = impression_pos_count_simple(cites); new_pos = scores_pos[tid] if tid < len(scores_pos) else 0.0
                scores_wd = impression_word_count_simple(cites); new_wd = scores_wd[tid] if tid < len(scores_wd) else 0.0
                old_wp, old_pos, old_wd = ref_entry["ori_object_dict"]["wordpos"], ref_entry["ori_object_dict"]["pos"], ref_entry["ori_object_dict"]["word"]
                old_geo = old_wp * 0.33 + old_pos * 0.33 + old_wd * 0.33
                new_geo = new_wp * 0.33 + new_pos * 0.33 + new_wd * 0.33
                geo_score_component = new_geo - old_geo
            raw_scores['geo'] = geo_score_component
        
        return raw_scores

    results_in_order = [None] * len(valid_tasks)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        fut_to_vtask_idx = {pool.submit(task, c, i, r): idx for idx, (c, i, r) in enumerate(valid_tasks)}
        for fut in as_completed(fut_to_vtask_idx):
            vtask_idx = fut_to_vtask_idx[fut]
            try:
                results_in_order[vtask_idx] = fut.result()
            except Exception as e:
                print(f"Error in reward task for vtask_idx {vtask_idx}: {e}", flush=True)
                results_in_order[vtask_idx] = {}

    num_valid_completions = len(valid_tasks)
    combined_scores = np.zeros(num_valid_completions, dtype=np.float32)
    stats_for_logging = {}

    for score_name in use_scores:
        raw_score_list = [res.get(score_name, 0.0) for res in results_in_order]
        if raw_score_list:
            stats_for_logging[f"rewards/raw_{score_name}/mean"] = statistics.mean(raw_score_list)
            stats_for_logging[f"rewards/raw_{score_name}/std"] = statistics.stdev(raw_score_list) if len(raw_score_list) > 1 else 0.0
        
        normalized_list = _normalize_scores(raw_score_list)
        weight = score_weights.get(score_name, 1.0)
        combined_scores += np.array(normalized_list) * weight

    final_rewards_for_valid_tasks = _normalize_scores(combined_scores.tolist())
    all_rewards = [0.0] * len(completions)
    for i, reward in enumerate(final_rewards_for_valid_tasks):
        original_idx = original_indices[i]
        all_rewards[original_idx] = reward

    return all_rewards



def length_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[Optional[float]]:
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        rewards.append(-len(content))
    return rewards



def accuracy_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Compute binary rewards if verifiable, `None` otherwise to skip this example
            try:
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                reward = None
        else:
            # If the gold solution is not parseable, we assign `None` to skip this example
            reward = None
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def tag_count_reward(completions, **kwargs) -> list[float]:
    """Reward function that checks if we produce the desired number of think and answer tags associated with `format_reward()`.

    Adapted from: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py-L90
    """

    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<think>\n") == 1:
            count += 0.335
        if text.count("\n</think>\n") == 1:
            count += 0.335
        if text.count("\n<answer>\n") == 1:
            count += 0.335
        if text.count("\n</answer>") == 1:
            count += 0.335
        return count

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]


def reasoning_steps_reward(completions, **kwargs):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic number 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]


def len_reward(completions: list[Dict[str, str]], solution: list[str], **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from the Kimi 1.5 tech report: https://huggingface.co/papers/2501.12599

    Args:
        completions: List of model completions
        solution: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = [completion[0]["content"] for completion in completions]

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparseable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    def cosine_scaled_reward(completions, solution, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            gold_parsed = parse(
                sol,
                extraction_mode="first_match",
                extraction_config=[LatexExtractionConfig()],
            )
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # Skip unparseable examples
                print("Failed to parse gold solution: ", sol)
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float, language: str = "en"):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://huggingface.co/papers/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    language: Language of the text, defaults to `en`. Used to choose the way to split the text into n-grams.
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    if language == "en":

        def zipngram(text: str, ngram_size: int):
            words = text.lower().split()
            return zip(*[words[i:] for i in range(ngram_size)]), words

    elif language == "zh":
        from transformers.utils.import_utils import _is_package_available

        if not _is_package_available("jieba"):
            raise ValueError("Please install jieba to use Chinese language")

        def zipngram(text: str, ngram_size: int):
            import jieba

            seg_list = list(jieba.cut(text))
            return zip(*[seg_list[i:] for i in range(ngram_size)]), seg_list

    else:
        raise ValueError(
            f"Word splitting for language `{language}` is not yet implemented. Please implement your own zip-ngram function."
        )

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            ngram_array, words = zipngram(completion, ngram_size)

            if len(words) < ngram_size:
                rewards.append(0.0)
                continue

            for ng in ngram_array:
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward


def _init_event_loop():
    """Initialize or get the current event loop."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def ioi_code_reward(completions, test_batch_size: int = 1, provider_type: str = "piston", **kwargs) -> list[float]:
    """Reward function that evaluates IOI problems using a specified execution client.

    Assumes the dataset has the same format as hf.co/datasets/open-r1/ioi

    Args:
        completions: List of model completions to evaluate
        test_batch_size: Evaluate these many test cases in parallel, then check if any of them failed (0 score):
                       if so stop evaluating; otherwise continue with the next batch of test cases.
        provider_type: The execution provider to use (default: "piston"). Supported values: "piston", "morph"
        **kwargs: Additional arguments passed from the dataset
    """
    # Get the appropriate client based on provider_type
    if provider_type == "morph":
        execution_client = get_morph_client_from_env()
    else:
        # for info on setting up piston workers, see slurm/piston/README.md
        execution_client = get_piston_client_from_env()

    code_snippets = [
        # note: grading is automatically skipped if no code is extracted
        add_includes(extract_code(completion[-1]["content"], "cpp"), problem_id)
        for completion, problem_id in zip(completions, kwargs["id"])
    ]

    async def run_catch_exceptions(task):
        try:
            return await task
        except Exception as e:
            print(f"Error from {provider_type} worker: {e}")
            return SubtaskResult()

    problems_data = [dict(zip(kwargs.keys(), values)) for values in zip(*kwargs.values())]

    loop = _init_event_loop()
    evals = [
        loop.create_task(
            run_catch_exceptions(
                score_subtask(
                    execution_client,
                    problem_data,
                    code,
                    test_batch_size=test_batch_size,
                )
            )
        )
        for problem_data, code in zip(problems_data, code_snippets)
    ]
    results = loop.run_until_complete(asyncio.gather(*evals))

    return [result.score for result in results]


def cf_code_reward(
    completions,
    test_batch_size: int = 1,
    patch_code: bool = False,
    scoring_mode: Literal["pass_fail", "partial", "weighted_sum"] = "weighted_sum",
    **kwargs,
) -> list[float]:
    """Reward function that evaluates Codeforces problems using Piston+our CF package.

    Assumes the dataset has the same format as hf.co/datasets/open-r1/codeforces (verifiable-prompts subset)

    test_batch_size: evaluate these many test cases in parallel, then check if any of them failed (0 score): if so stop evaluating; otherwise continue with the next batch of test cases.
    """
    # for info on setting up piston workers, see slurm/piston/README.md
    piston_client = get_piston_client_from_env()

    languages = kwargs["language"] if "language" in kwargs else [None] * len(completions)
    code_snippets = [
        # note: grading is automatically skipped if a problem has no tests
        cf_patch_code(extract_code(completion[-1]["content"], language), language)
        if patch_code
        else extract_code(completion[-1]["content"], language)
        for completion, language in zip(completions, languages)
    ]

    async def run_catch_exceptions(task):
        try:
            return await task
        except Exception as e:
            print(f"Error from Piston worker: {e}")
            return None

    # load problem data. undo separating kwargs by column
    problems_data = [dict(zip(kwargs.keys(), values)) for values in zip(*kwargs.values())]

    loop = _init_event_loop()
    evals = [
        loop.create_task(
            run_catch_exceptions(
                cf_score_submission(
                    piston_client,
                    problem_data,
                    code,
                    test_batch_size=test_batch_size,
                    scoring_mode=scoring_mode,
                    submission_language=problem_data.get("language", None),
                )
            )
        )
        for problem_data, code in zip(problems_data, code_snippets)
    ]
    results = loop.run_until_complete(asyncio.gather(*evals))

    return results


def extract_code(completion: str, language: str | None = "python") -> str:
    if language is None:
        return ""
    pattern = re.compile(rf"```{language}\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[-1] if len(matches) >= 1 else ""
    return extracted_answer


def binary_code_reward(
    completions,
    num_parallel: int = 2,
    provider_type: str = "e2b",
    enforce_same_language: bool = False,
    **kwargs,
) -> list[float]:
    rewards = code_reward(
        completions,
        num_parallel=num_parallel,
        provider_type=provider_type,
        enforce_same_language=enforce_same_language,
        **kwargs,
    )
    BINARY_THRESHOLD = 0.99

    output = []
    for reward in rewards:
        if reward is None:
            output.append(None)
        else:
            output.append(1.0 if reward > BINARY_THRESHOLD else 0.0)

    return output


def code_reward(
    completions,
    num_parallel: int = 2,
    provider_type: str = "e2b",
    enforce_same_language: bool = False,
    **kwargs,
) -> list[float]:
    """Reward function that evaluates code snippets using a code execution provider.

    Assumes the dataset contains a `verification_info` column with test cases.

    Args:
        completions: List of model completions to evaluate
        num_parallel: Number of parallel code executions (default: 2)
        provider_type: Which code execution provider to use (default: "e2b")
        enforce_same_language: If True, verify all problems use the same language (default: False)
        **kwargs: Additional arguments passed to the verification
    """
    evaluation_script_template = """
    import subprocess
    import json

    def evaluate_code(code, test_cases):
        passed = 0
        total = len(test_cases)
        exec_timeout = 5

        for case in test_cases:
            process = subprocess.run(
                ["python3", "-c", code],
                input=case["input"],
                text=True,
                capture_output=True,
                timeout=exec_timeout
            )

            if process.returncode != 0:  # Error in execution
                continue

            output = process.stdout.strip()

            # TODO: implement a proper validator to compare against ground truth. For now we just check for exact string match on each line of stdout.
            all_correct = True
            for line1, line2 in zip(output.split('\\n'), case['output'].split('\\n')):
                all_correct = all_correct and line1.strip() == line2.strip()

            if all_correct:
                passed += 1

        success_rate = (passed / total)
        return success_rate

    code_snippet = {code}
    test_cases = json.loads({test_cases})

    evaluate_code(code_snippet, test_cases)
    """

    code_snippets = [extract_code(completion[-1]["content"]) for completion in completions]
    verification_info = kwargs["verification_info"]

    template = evaluation_script_template

    scripts = [
        template.format(code=json.dumps(code), test_cases=json.dumps(json.dumps(info["test_cases"])))
        for code, info in zip(code_snippets, verification_info)
    ]

    language = verification_info[0]["language"]

    if enforce_same_language:
        all_same_language = all(v["language"] == language for v in verification_info)
        if not all_same_language:
            raise ValueError("All verification_info must have the same language", verification_info)

    execution_provider = get_provider(
        provider_type=provider_type,
        num_parallel=num_parallel,
        **kwargs,
    )

    return execution_provider.execute_scripts(scripts, ["python"] * len(scripts))


def get_code_format_reward(language: str = "python"):
    """Format reward function specifically for code responses.

    Args:
        language: Programming language supported by E2B https://e2b.dev/docs/code-interpreting/supported-languages
    """

    def code_format_reward(completions, **kwargs):
        # if there is a language field, use it instead of the default language. This way we can have mixed language training.
        languages = kwargs["language"] if "language" in kwargs else [language] * len(completions)

        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [
            re.match(
                rf"^<think>\n.*?\n</think>\n<answer>\n.*?```{sample_language}.*?```.*?\n</answer>$",
                content,
                re.DOTALL | re.MULTILINE,
            )
            for content, sample_language in zip(completion_contents, languages)
        ]
        return [1.0 if match else 0.0 for match in matches]

    return code_format_reward


def get_soft_overlong_punishment(max_completion_len, soft_punish_cache):
    """
    Reward function that penalizes overlong completions. It is used to penalize overlong completions,
    but not to reward shorter completions. Reference: Eq. (13) from the DAPO paper (https://huggingface.co/papers/2503.14476)

    Args:
        max_completion_len: Maximum length of the completion
        soft_punish_cache: Minimum length of the completion. If set to 0, no minimum length is applied.
    """

    def soft_overlong_punishment_reward(completion_ids: list[list[int]], **kwargs) -> list[float]:
        """Reward function that penalizes overlong completions."""
        rewards = []
        for ids in completion_ids:
            completion_length = len(ids)
            if completion_length <= max_completion_len - soft_punish_cache:
                rewards.append(0.0)
            elif max_completion_len - soft_punish_cache < completion_length <= max_completion_len:
                rewards.append((max_completion_len - soft_punish_cache - completion_length) / soft_punish_cache)
            else:
                rewards.append(-1.0)
        return rewards

    return soft_overlong_punishment_reward


def get_reward_funcs(script_args) -> list[Callable]:
    REWARD_FUNCS_REGISTRY = {
        "len": length_reward,
        "wordcountpos": wordcountpos_reward,
        "accuracy": accuracy_reward,
        "format": format_reward,
        "reasoning_steps": reasoning_steps_reward,
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
        "length": len_reward,
        "code": update_wrapper(
            partial(
                code_reward,
                num_parallel=script_args.parallel_code_exec_per_proc,
                provider_type=script_args.code_provider,
                enforce_same_language=getattr(script_args, "enforce_same_language", False),
            ),
            code_reward,
        ),
        "binary_code": update_wrapper(
            partial(
                binary_code_reward,
                num_parallel=script_args.parallel_code_exec_per_proc,
                provider_type=script_args.code_provider,
                enforce_same_language=getattr(script_args, "enforce_same_language", False),
            ),
            binary_code_reward,
        ),
        "ioi_code": update_wrapper(
            partial(
                ioi_code_reward,
                test_batch_size=script_args.code_eval_test_batch_size,
                provider_type=getattr(script_args, "ioi_provider", "piston"),
            ),
            ioi_code_reward,
        ),
        "cf_code": update_wrapper(
            partial(
                cf_code_reward,
                test_batch_size=script_args.code_eval_test_batch_size,
                scoring_mode=script_args.code_eval_scoring_mode,
            ),
            cf_code_reward,
        ),
        "code_format": get_code_format_reward(language=script_args.code_language),
        "tag_count": tag_count_reward,
        "soft_overlong_punishment": get_soft_overlong_punishment(
            max_completion_len=script_args.max_completion_len,
            soft_punish_cache=script_args.soft_punish_cache,
        ),
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    return reward_funcs
