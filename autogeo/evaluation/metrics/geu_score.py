import openai
import os
import re
import time
import glob
import threading
import concurrent.futures
import logging
from openai import OpenAI

import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Literal
from pydantic import BaseModel, Field, ValidationError, create_model
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm

load_dotenv("keys.env")
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')

# Disable HTTP request logs from OpenAI SDK and underlying HTTP libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except openai.OpenAIError:
    client = None
    logging.error("OpenAI client could not be initialized. Check if OPENAI_API_KEY is set.")


def preprocess_data_for_evaluation(
    filename: str, 
    engine_llm: str, 
    rewrite_method_name: str,
    key_points_dir: str = "key_point",
    text_list_name: str = "text_list"
) -> Dict[str, Dict[str, Any]]:
    """Preprocess data for GEU evaluation.
    
    Args:
        filename: Path to JSON file containing evaluation data
        engine_llm: Method name used for generating response
        rewrite_method_name: Method name used for document improvement (or "original" for original)
        key_points_dir: Directory containing keypoint files (default: "key_point")
        text_list_name: Field name for text list in data (default: "text_list")
        
    Returns:
        Dictionary mapping question_id to processed data
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        logging.error(f"Error: The file '{filename}' was not found.")
        return {}
    except json.JSONDecodeError:
        logging.error(f"Error: The file '{filename}' is not a valid JSON file.")
        return {}

    processed_data = {}
    is_researchy_file = "data/Researchy-GEO/test" in filename

    for question_id, item_data in data.items():
        try:
            response_field_name = f"{engine_llm}_response"
            text_field_name = f"{rewrite_method_name}_text"
            query = item_data['query']
            subquestions = item_data.get('subquestions', []) 
            original_text_list = item_data[text_list_name][:] 
            target_id = item_data.get('target_id', -1)
            
            # Check if response exists, if not, generate it
            if response_field_name not in item_data:
                logging.info(f"Missing {response_field_name} for question_id '{question_id}'. Generating response...")
                try:
                    from ..generative_engine import generate_answer_gemini, generate_answer_gpt, generate_answer_claude
                    
                    # Generate response based on the engine_llm
                    if "gpt" in engine_llm:
                        generated_response = generate_answer_gpt(query, original_text_list)
                    elif "claude" in engine_llm:
                        generated_response = generate_answer_claude(query, original_text_list)
                    elif "gemini" in engine_llm:
                        generated_response = generate_answer_gemini(query, original_text_list)
                    else:
                        logging.warning(f"Unknown engine_llm '{engine_llm}'. Cannot generate response automatically.")
                        continue
                    
                    # Save the generated response back to the data
                    item_data[response_field_name] = generated_response
                    logging.info(f"Successfully generated {response_field_name} for question_id '{question_id}'")
                    
                    # Also save geo_score if not exists
                    if f"{engine_llm}_geo_score" not in item_data:
                        from .geo_score import extract_citations_new, impression_wordpos_count_simple, impression_word_count_simple, impression_pos_count_simple
                        citations = extract_citations_new(generated_response)
                        item_data[f"{engine_llm}_geo_score"] = {
                            'wordpos': impression_wordpos_count_simple(citations)[target_id],
                            'word': impression_word_count_simple(citations)[target_id],
                            'pos': impression_pos_count_simple(citations)[target_id]
                        }
                    
                    # Save updated data back to file
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=4)
                    logging.info(f"Saved generated response to {filename}")
                    
                except Exception as e:
                    logging.error(f"Failed to generate response for question_id '{question_id}': {e}")
                    continue
            
            original_response = item_data[response_field_name] 
            
            if rewrite_method_name == "original":
                final_text_list = original_text_list
            else:
                replacement_text = item_data[text_field_name]
                if 0 <= target_id < len(original_text_list):
                    final_text_list = original_text_list
                    final_text_list[target_id] = replacement_text
                else:
                    logging.warning(f"For q_id '{question_id}', target_id '{target_id}' is invalid. Using original text_list.")
                    final_text_list = original_text_list

            processed_item = {
                "query": query,
                "subquestions": subquestions,
                "ori_response": original_response,
                "text_list": final_text_list
            }

            if is_researchy_file:
                current_dir = Path(__file__).resolve().parent
                key_point_file = current_dir.parent.parent.parent / "data/Researchy-GEO" / Path(key_points_dir) / f"{question_id}_aggregated.json"
                if key_point_file.exists():
                    with open(key_point_file, 'r', encoding='utf-8') as kp_f:
                        kp_data = json.load(kp_f)
                        if "key_points" in kp_data and kp_data["key_points"]:
                            processed_item["keypoint_list"] = kp_data["key_points"]
                        else:
                            logging.warning(f"Key point file for {question_id} is empty or missing 'key_points' field. Skipping this question.")
                else:
                    logging.warning(f"Key point file not found for {question_id} at {key_point_file}. Skipping this question.")

            processed_data[question_id] = processed_item
        except KeyError as e:
            logging.warning(f"KeyError: Missing key {e} for question_id '{question_id}'. Skipping this question.")
            continue
        except Exception as e:
            logging.error(f"An unexpected error occurred while processing question_id '{question_id}': {e}")
            continue

    return processed_data


class CitationSupportValues(str):
    FULL = "full_support"
    PARTIAL = "partial_support"
    NONE = "no_support"

    @classmethod
    def score(cls, value: str) -> float:
        return {
            cls.NONE: 0.0,
            cls.PARTIAL: 0.5,
            cls.FULL: 1.0,
        }.get(value, 0.0)


class CitationSupport(BaseModel):
    support: Literal["full_support", "partial_support", "no_support"] = Field(..., description="The level of support provided by the citation.")
    justification: str = Field(..., description="A brief justification for the support level.")

class ClaimEntry(BaseModel):
    claim_id: int
    claim: str
    source_indices: List[int] = Field(description="A list of integer indices (e.g., [0], [1, 2]) corresponding to the provided documents. Empty if no source is cited.")

class ClaimsModel(BaseModel):
    claims: List[ClaimEntry]

class QualityEvaluation(BaseModel):
    rating: int = Field(..., ge=0, le=10, description="An integer rating from 0 to 10.")
    justification: str = Field(..., description="A brief justification for the rating.")

class KeyPointJudgment(BaseModel):
    label: Literal["Supported", "Omitted", "Contradicted"]
    justification: str


NEW_PROMPTS = {
    "claim_extractor": """You are an information extraction expert.
Given a report, extract all distinct factual claims. For each claim, identify the source indices it cites (e.g., [0], [1], [2, 3]).

Return a JSON object with a "claims" list, where each entry has:
- "claim_id": A sequential integer starting from 1.
- "claim": A concise, complete sentence of the claim.
- "source_indices": A list of integer indices cited for this claim. If no source is cited, return an empty list [].

**IMPORTANT**:
- Only extract factual claims, not opinions or summaries.
- The source indices must be integers extracted directly from citations like `[0]` or `[1, 2]`.

Report to process:
\"\"\"
{answer}
\"\"\"

Return the JSON object and nothing else.""",

    "citation_checker": """You are a meticulous fact-checker. Evaluate if a "Statement" is supported by the "Source Text".
Respond strictly in JSON format with "support" ('full_support', 'partial_support', or 'no_support') and a brief "justification".

- "full_support": All information in the statement is directly supported by the source text.
- "partial_support": Some parts are supported, but other parts are not.
- "no_support": The source text does not support the statement.

Statement: "{claim}"

Source Text:
\"\"\"
{document_content}
\"\"\"

Your JSON response:""",

    "quality_evaluator": """You are a strict expert evaluator. Assess the quality of an "Answer" to a "Question" based ONLY on the criterion of **{criterion_name}**.

**Criterion: {criterion_name}**
{criterion_description}

**Question:**
{question}

**Answer:**
{answer}

Provide your rating as a JSON object with two keys:
1. "rating": An integer from 0 (poor) to 10 (excellent).
2. "justification": A brief, harsh justification explaining why the answer earned that rating based on the criterion.

**Do not be generous.** High scores are for outstanding answers.

Your JSON response:""",

    "keypoint_judge": """You are given a JSON array of Key Points and a Report.
For EACH Key Point, determine if the Report:
- **Supported** it: The Report contains information that supports the Key Point.
- **Omitted** it: The Report does not mention or cover the Key Point.
- **Contradicted** it: The Report says something that disagrees with the Key Point.

Return your answer as a single JSON object. The keys must be the `point_number` (as a string). The value must be an object with "label" and "justification".

Example: { "1": { "label": "Supported", "justification": "..." } }

Respond ONLY with the JSON object.

---
Key Points:

{key_points_json_str}

---
Report:
{answer}
""",

"completeness": """You are an expert evaluator. Your task is to assess how well the "Main Answer" holistically covers the key "Aspects to Cover" provided in a list. Consider if the main ideas of the aspects are present and adequately explained. Answer on a scale of 1 to 5. Your response MUST begin with a single digit from 1 to 5, followed by a newline and a brief explanation.

- 5: Excellent coverage. The answer comprehensively discusses all or nearly all of the listed aspects.
- 4: Good coverage. The answer discusses most of the key aspects well, but some might be superficial or minor ones are missed.
- 3: Moderate coverage. The answer discusses some of the aspects, but misses several major ones or treats them too briefly.
- 2: Poor coverage. The answer only vaguely alludes to one or two aspects but fails to provide substantive information.
- 1: No coverage. The answer almost completely ignores the provided list of aspects.

Aspects to Cover:
\"\"\"
{aspects_list_str}
\"\"\"

Main Answer:
\"\"\"
{main_answer}
\"\"\"

Your Rating (1-5):""",
}


def call_llm_for_json(prompt: str, pydantic_model: BaseModel, max_retries: int = 3) -> Optional[BaseModel]:
    """Call LLM and parse response as Pydantic model.
    
    Args:
        prompt: Prompt text for LLM
        pydantic_model: Pydantic model class for validation
        max_retries: Maximum number of retry attempts (default: 3)
        
    Returns:
        Validated Pydantic model instance, or None if all retries fail
    """
    if not client:
        return None
        
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            parsed_json = json.loads(content)
            return pydantic_model.model_validate(parsed_json)
        except (json.JSONDecodeError, ValidationError) as e:
            logging.warning(f"Failed to parse or validate JSON on attempt {attempt + 1}. Error: {e}. Retrying...")
            time.sleep(2 ** attempt)
        except Exception as e:
            logging.error(f"API call failed with unexpected error: {e}. Attempt {attempt + 1}. Retrying...")
            time.sleep(2 ** attempt)
    
    logging.error(f"Failed to get a valid JSON response for model {pydantic_model.__name__} after all retries.")
    return None

def calculate_citation_quality(response: str, documents: List[str]) -> Tuple[Optional[float], Optional[float]]:
    """Calculate citation quality metrics (precision and recall).
    
    Args:
        response: Generated answer text with citations
        documents: List of source document texts
        
    Returns:
        Tuple of (precision_score, recall_score), or (None, None) if calculation fails
    """
    extractor_prompt = NEW_PROMPTS["claim_extractor"].format(answer=response)
    claims_data = call_llm_for_json(extractor_prompt, ClaimsModel)
    if not claims_data or not claims_data.claims:
        # logging.warning("No claims were extracted. Citation quality cannot be calculated.")
        return None, None

    total_claims = len(claims_data.claims)
    cited_claims = [c for c in claims_data.claims if c.source_indices]
    num_cited_claims = len(cited_claims)
    recall_score = num_cited_claims / total_claims if total_claims > 0 else 0.0
    if not cited_claims:
        precision_score = 0.0 if recall_score == 0.0 else None
        return precision_score, recall_score

    precision_scores = []
    for claim in cited_claims:
        claim_support_scores = []
        for doc_index in claim.source_indices:
            if 0 <= doc_index < len(documents):
                checker_prompt = NEW_PROMPTS["citation_checker"].format(
                    claim=claim.claim,
                    document_content=documents[doc_index]
                )
                support_result = call_llm_for_json(checker_prompt, CitationSupport)
                if support_result:
                    score = CitationSupportValues.score(support_result.support)
                    claim_support_scores.append(score)
            # else:
            #     logging.warning(f"Invalid document index {doc_index} for claim: '{claim.claim[:50]}...'")

        if claim_support_scores:
            precision_scores.append(max(claim_support_scores))
    precision_score = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
    return precision_score, recall_score


QUALITY_CRITERIA = [
    {"name": "Clarity", "description": "Assess how clearly and rigorously the answer is structured. High-quality responses are like in-depth reports with distinct, non-overlapping points and strong logical flow. Penalize redundancy, ambiguity, and filler."},
    {"name": "Depth", "description": "Assess comprehensiveness and analytical depth. Excellent reports show critical thinking and nuance, not just surface-level facts. Word count does not equal depth."},
    {"name": "Balance", "description": "Evaluate fairness and objectivity. Excellent reports present multiple perspectives impartially, especially for controversial topics. Poor reports are biased or one-sided."},
    {"name": "Breadth", "description": "Evaluate how many distinct and relevant subtopics, perspectives, or contexts are covered. Excellent reports provide a wide-ranging yet focused exploration."},
    {"name": "Support", "description": "Evaluate the extent to which key claims are substantiated by credible evidence from the provided text. Claims should be clearly linked to sources. Vague references are unacceptable."},
    {"name": "Insightfulness", "description": "Assess originality and value. Excellent reports go beyond common knowledge, offering original synthesis or thought-provoking connections. Recommendations must be concrete and actionable."}
]

def calculate_quality_dimensions(query: str, response: str) -> Dict[str, Optional[float]]:
    """Calculate quality dimension scores for a response.
    
    Args:
        query: User query text
        response: Generated answer text
        
    Returns:
        Dictionary mapping quality dimension names to scores (0.0-1.0)
    """
    quality_scores = {}
    for criterion in QUALITY_CRITERIA:
        prompt = NEW_PROMPTS["quality_evaluator"].format(
            criterion_name=criterion["name"],
            criterion_description=criterion["description"],
            question=query,
            answer=response
        )
        eval_result = call_llm_for_json(prompt, QualityEvaluation)
        
        if eval_result:
            quality_scores[criterion["name"]] = eval_result.rating / 10.0
        else:
            quality_scores[criterion["name"]] = None
            logging.error(f"Failed to get quality score for criterion: {criterion['name']}")
    
    return quality_scores


def calculate_keypoint_coverage(keypoint_list: List[Dict], response: str) -> Tuple[Optional[float], Optional[float], Dict]:
    """Calculate keypoint coverage metrics (KPR and KPC).
    
    Args:
        keypoint_list: List of keypoint dictionaries with point_number and point_content
        response: Generated answer text to evaluate
        
    Returns:
        Tuple of (keypoint_recall, keypoint_precision, judgments_dict)
    """
    if not keypoint_list:
        return None, None, {}

    simplified_points = [
        {"point_number": p["point_number"], "point_content": p["point_content"]}
        for p in keypoint_list
    ]
    
    key_points_json_str = json.dumps(simplified_points, indent=2, ensure_ascii=False)
    
    prompt = f"""You are given a JSON array of Key Points and a Report.
For EACH Key Point, determine if the Report:
- **Supported** it: The Report contains information that supports the Key Point.
- **Omitted** it: The Report does not mention or cover the Key Point.
- **Contradicted** it: The Report says something that disagrees with the Key Point.

Return your answer as a single JSON object. The keys must be the `point_number` (as a string). The value must be an object with "label" and "justification".

Example: {{ "1": {{ "label": "Supported", "justification": "..." }} }}

Respond ONLY with the JSON object.

---
Key Points:
{key_points_json_str}

---
Report:
{response}
"""

    judgment_fields = {
        str(p["point_number"]): (KeyPointJudgment, ...) for p in simplified_points
    }
    JudgmentsModel = create_model('JudgmentsModel', **judgment_fields)
    judgments_dict_raw = call_llm_for_json(prompt, JudgmentsModel)
    if not judgments_dict_raw:
        logging.error("Failed to get valid judgments for key points.")
        return None, None, {}

    try:
        judgments_dict = judgments_dict_raw.model_dump()
    except AttributeError:
        judgments_dict = judgments_dict_raw.dict()
    
    supported_count = 0
    contradicted_count = 0
    total_points = len(simplified_points)

    for point_num_str, judgment in judgments_dict.items():
        label = judgment.get("label")
        if label == "Supported":
            supported_count += 1
        elif label == "Contradicted":
            contradicted_count += 1
    
    kpr = supported_count / total_points if total_points > 0 else 0.0
    kpc = contradicted_count / total_points if total_points > 0 else 0.0
    return kpr, kpc, judgments_dict



def process_single_question(question_id: str, item_data: dict, metrics_to_run: List[str]) -> Tuple[str, Dict[str, Any]]:
    """Process a single question to calculate GEU metrics.
    
    Args:
        question_id: Unique identifier for the question
        item_data: Dictionary containing question data (query, text_list, ori_response, keypoint_list, etc.)
        metrics_to_run: List of metrics to calculate (e.g., ["citation_quality", "quality_dimensions", "keypoint_coverage"])
        
    Returns:
        Tuple of (question_id, results_dict) where results_dict contains calculated metrics
    """
    final_scores = {}

    if "citation_quality" in metrics_to_run:
        precision, recall = calculate_citation_quality(item_data["ori_response"], item_data["text_list"])
        final_scores["Precision"] = precision
        final_scores["Recall"] = recall

    if "quality_dimensions" in metrics_to_run:
        quality_scores = calculate_quality_dimensions(item_data["query"], item_data["ori_response"])
        final_scores.update(quality_scores)

    if "keypoint_coverage" in metrics_to_run:
        if "keypoint_list" in item_data:
            kpr, kpc, _ = calculate_keypoint_coverage(item_data["keypoint_list"], item_data["ori_response"])
            final_scores["KPR"] = kpr
            final_scores["KPC"] = kpc
        else:
            # For keypoints not available, set KPR and KPC to None
            # logging.warning(f"Metric 'keypoint_coverage' requested for QID {question_id}, but no keypoint_list was found.")
            final_scores["KPR"] = None
            final_scores["KPC"] = None

    # logging.info(f"Completed GEU evaluation for question {question_id}")
    return question_id, final_scores



def evaluate_ge_utility(
    data: dict, 
    question_id_list: list[str],
    metrics_to_run: List[str],
    max_workers: int = 5,
    pbar: Optional[tqdm] = None
) -> Dict[str, Dict[str, Any]]:
    if not client:
        logging.error("FATAL: OpenAI client not initialized. Aborting.")
        return {}
        
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='EvalWorker') as executor:
        future_to_qid = {
            executor.submit(process_single_question, qid, data[qid], metrics_to_run): qid 
            for qid in question_id_list if qid in data
        }
        for future in concurrent.futures.as_completed(future_to_qid):
            qid = future_to_qid[future]
            try:
                q_id, scores_dict = future.result()
                results[q_id] = scores_dict
                if pbar is not None:
                    pbar.update(1)
            except Exception as e:
                logging.error(f"QID {qid} generated a critical exception: {e}")
                results[qid] = {"error": str(e)}
                if pbar is not None:
                    pbar.update(1)
                
    return results



def _preprocess_single_question_geu(
    qid: str,
    data: dict,
    engine_llm: str,
    rewrite_method_name: str,
    metrics_to_evaluate: list,
    text_list_name: str,
    filename: str,
    log: Any,
    file_lock: threading.Lock
) -> tuple[Optional[str], Optional[dict], bool]:
    """Preprocess a single question for GEU score calculation.
    
    Returns:
        (qid, processed_item, should_skip) tuple
        - qid: question ID if should process, None if should skip
        - processed_item: processed data dict if should process, None otherwise
        - should_skip: True if question should be skipped (already has complete GEU score)
    """
    if qid not in data:
        return None, None, False
    
    # Check if GEU score already exists for this question
    dic_name = engine_llm + "_geu_score"
    geu_score_data = data[qid].get(dic_name)
    
    if geu_score_data and isinstance(geu_score_data, dict) and len(geu_score_data) > 0:
        # Map metric names to actual stored keys
        existing_keys = set(geu_score_data.keys())
        all_metrics_present = True
        missing_keys = []
        
        for metric in metrics_to_evaluate:
            if metric == "citation_quality":
                # citation_quality stores as Precision and Recall
                if "Precision" not in existing_keys or "Recall" not in existing_keys:
                    all_metrics_present = False
                    missing_keys.append("Precision/Recall")
                elif geu_score_data.get("Precision") is None or geu_score_data.get("Recall") is None:
                    all_metrics_present = False
                    missing_keys.append("Precision/Recall (None)")
            elif metric == "quality_dimensions":
                # quality_dimensions stores as multiple keys (Clarity, Depth, Balance, etc.)
                # Check if at least one quality dimension key exists
                quality_keys = ["Clarity", "Depth", "Balance", "Breadth", "Support", "Insightfulness"]
                has_quality_dimension = any(key in existing_keys and geu_score_data.get(key) is not None for key in quality_keys)
                if not has_quality_dimension:
                    all_metrics_present = False
                    missing_keys.append("quality_dimensions")
            elif metric == "keypoint_coverage":
                # keypoint_coverage stores as KPR and KPC
                if "KPR" not in existing_keys or "KPC" not in existing_keys:
                    all_metrics_present = False
                    missing_keys.append("KPR/KPC")
                elif geu_score_data.get("KPR") is None or geu_score_data.get("KPC") is None:
                    all_metrics_present = False
                    missing_keys.append("KPR/KPC (None)")
        
        if all_metrics_present:
            # All required metrics already exist, skip this question
            return None, None, True
        # else:
        #     log.info(f"Question {qid} has partial GEU score. Missing keys: {', '.join(missing_keys)}. Will recalculate.")
    
    try:
        response_field_name = f"{engine_llm}_response"
        text_field_name = f"{rewrite_method_name}_text"
        item_data = data[qid]
        query = item_data['query']
        subquestions = item_data.get('subquestions', []) 
        original_text_list = item_data[text_list_name][:] 
        target_id = item_data.get('target_id', -1)
        
        # Check if response exists, if not, generate it
        if response_field_name not in item_data or not item_data[response_field_name]:
            log.info(f"Missing {response_field_name} for question_id '{qid}'. Generating response...")
            try:
                from ..generative_engine import generate_answer_gemini, generate_answer_gpt, generate_answer_claude
                
                if "gpt" in engine_llm:
                    generated_response = generate_answer_gpt(query, original_text_list, model_name=engine_llm)
                elif "claude" in engine_llm:
                    generated_response = generate_answer_claude(query, original_text_list, model_name=engine_llm)
                elif "gemini" in engine_llm:
                    generated_response = generate_answer_gemini(query, original_text_list, model_name=engine_llm)
                else:
                    log.warning(f"Unknown engine_llm '{engine_llm}'. Skipping question {qid}.")
                    return None, None, False
                
                # Save response and geo_score (thread-safe)
                with file_lock:
                    item_data[response_field_name] = generated_response
                    
                    # Also save geo_score if not exists
                    if f"{engine_llm}_geo_score" not in item_data:
                        from .geo_score import extract_citations_new, impression_wordpos_count_simple, impression_word_count_simple, impression_pos_count_simple
                        citations = extract_citations_new(generated_response)
                        item_data[f"{engine_llm}_geo_score"] = {
                            'wordpos': impression_wordpos_count_simple(citations)[target_id],
                            'word': impression_word_count_simple(citations)[target_id],
                            'pos': impression_pos_count_simple(citations)[target_id]
                        }
                    
                    # Save updated data back to file immediately
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=4)
                log.info(f"Saved generated response to {filename}")
            except Exception as e:
                log.error(f"Failed to generate response for question_id '{qid}': {e}")
                return None, None, False
        
        original_response = item_data[response_field_name]
        
        if rewrite_method_name == "original":
            final_text_list = original_text_list
        else:
            replacement_text = item_data[text_field_name]
            if 0 <= target_id < len(original_text_list):
                final_text_list = original_text_list
                final_text_list[target_id] = replacement_text
            else:
                log.warning(f"For q_id '{qid}', target_id '{target_id}' is invalid. Using original text_list.")
                final_text_list = original_text_list

        processed_item = {
            "query": query,
            "subquestions": subquestions,
            "ori_response": original_response,
            "text_list": final_text_list
        }

        # Add keypoint data if available
        is_researchy_file = "Researchy-GEO" in filename
        if is_researchy_file:
            current_dir = Path(__file__).resolve().parent
            key_point_file = current_dir.parent.parent.parent / "data/Researchy-GEO" / Path("key_point") / f"{qid}_aggregated.json"
            # log.info(f"Loading Researchy-GEO key point file: {key_point_file}")
            if key_point_file.exists():
                try:
                    with open(key_point_file, 'r', encoding='utf-8') as kp_f:
                        kp_data = json.load(kp_f)
                        if "key_points" in kp_data and kp_data["key_points"]:
                            processed_item["keypoint_list"] = kp_data["key_points"]
                except json.JSONDecodeError:
                    log.error(f"Error decoding JSON from key point file: {key_point_file}")

        return qid, processed_item, False
    except KeyError as e:
        log.warning(f"KeyError: Missing key {e} for question_id '{qid}'. Skipping this question.")
        return None, None, False
    except Exception as e:
        log.error(f"An unexpected error occurred while processing question_id '{qid}': {e}")
        return None, None, False


def geu_score(
    num_examples: Optional[int],
    engine_llm: str,
    rewrite_method_name: str,
    metrics_to_evaluate: list,
    max_workers: int = 64,
    data_dir: str = "data/Researchy-GEO/test",
    text_list_name: str = "text_list",
    logger: Optional[Any] = None
) -> None:
    """Calculate and save GEU scores for documents.
    
    Args:
        num_examples: Number of examples to process (None for all)
        engine_llm: LLM method name used for generating response
        rewrite_method_name: Method name used for document improvement
        metrics_to_evaluate: List of metrics to evaluate (e.g., ["citation_quality", "quality_dimensions", "keypoint_coverage"])
        max_workers: Maximum number of worker threads (default: 64)
        data_dir: Directory containing data chunks
        text_list_name: Field name for text list in data (default: "text_list")
        logger: Optional logger instance to use (if None, uses logging module)
    """
    # Use provided logger or fall back to logging module
    log = logger if logger is not None else logging
    
    # Calculate total questions across all chunks
    total_questions = num_examples
    if num_examples is None:
        total_questions = 0
        chunk_files = sorted(glob.glob(f"{data_dir}/datachunk_*.json"))
        for filename in chunk_files:
            try:
                with open(filename, 'r', encoding="utf-8") as f:
                    data = json.load(f)
                total_questions += len(data.keys())
            except:
                pass
    
    # Create progress bar
    pbar = tqdm(total=total_questions, desc="GEU score evaluation", unit="question", dynamic_ncols=True) if total_questions is not None else None
    
    processed_count = 0
    chunk_idx = 0
    while num_examples is None or processed_count < num_examples:
        i = chunk_idx
        filename = f"{data_dir}/datachunk_{i}.json"
        if not os.path.exists(filename):
            log.warning(f"File not found, skipping: {filename}")
            break

        # log.info(f"Processing file: {filename}")

        try:
            with open(filename, 'r', encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            log.error(f"Error reading {filename}: {e}. Skipping.")
            chunk_idx += 1
            continue

        # Limit the number of questions to process based on remaining count
        # Sort question IDs to ensure consistent order across runs
        all_question_ids = sorted(list(data.keys()))
        if num_examples is None:
            question_ids_to_preprocess = all_question_ids
            # log.info(f"Processing {len(question_ids_to_preprocess)} questions from this chunk (processing all)")
        else:
            remaining = num_examples - processed_count
            if remaining <= 0:
                break
            question_ids_to_preprocess = all_question_ids[:remaining]
            log.info(f"Processing {len(question_ids_to_preprocess)} questions from this chunk (remaining: {remaining}/{num_examples})")

        # Preprocess only the selected questions (parallel)
        processed_data = {}
        file_lock = threading.Lock()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='GEUPreprocess') as executor:
            # Submit all preprocessing tasks
            future_to_qid = {
                executor.submit(
                    _preprocess_single_question_geu,
                    qid,
                    data,
                    engine_llm,
                    rewrite_method_name,
                    metrics_to_evaluate,
                    text_list_name,
                    filename,
                    log,
                    file_lock
                ): qid
                for qid in question_ids_to_preprocess
            }
            
            # Process completed tasks
            for future in concurrent.futures.as_completed(future_to_qid):
                qid = future_to_qid[future]
                try:
                    result_qid, processed_item, should_skip = future.result()
                    if should_skip:
                        # Question already has complete GEU score
                        if num_examples is None or processed_count < num_examples:
                            processed_count += 1
                            if pbar is not None:
                                pbar.update(1)
                            # if num_examples is None:
                            #     log.info(f"Question {qid} already has complete GEU score (metrics: {', '.join(metrics_to_evaluate)}), skipping... ({processed_count} total)")
                            # else:
                            #     log.info(f"Question {qid} already has complete GEU score (metrics: {', '.join(metrics_to_evaluate)}), skipping... ({processed_count}/{num_examples})")
                    elif result_qid and processed_item:
                        # Successfully preprocessed
                        processed_data[result_qid] = processed_item
                except Exception as e:
                    log.error(f'Error preprocessing question {qid}: {e}')

        if not processed_data:
            log.warning(f"No processable data found in {filename} for the specified methods. Skipping.")
            chunk_idx += 1
            continue
            
        question_ids_to_process = list(processed_data.keys())

        evaluation_results = evaluate_ge_utility(
            data=processed_data,
            question_id_list=question_ids_to_process,
            metrics_to_run=metrics_to_evaluate,
            max_workers=max_workers,
            pbar=pbar
        )

        # Reuse the same file lock for thread-safe writes
        for qid, scores_dict in evaluation_results.items():
            if num_examples is not None and processed_count >= num_examples:
                break
            if qid in data:
                dic_name = engine_llm + "_geu_score"
                if dic_name not in data[qid]:
                    data[qid][dic_name] = {}
                data[qid][dic_name].update(scores_dict)
                processed_count += 1
                
                # Save immediately after each update (thread-safe)
                try:
                    with file_lock:
                        with open(filename, "w", encoding="utf-8") as f:
                            json.dump(data, f, ensure_ascii=False, indent=4)
                except IOError as e:
                    log.error(f"Error writing updated results to {filename}: {e}")
            else:
                log.warning(f"Question ID {qid} from evaluation results not found in the original file data.")
        
        if num_examples is not None and processed_count >= num_examples:
            break
        chunk_idx += 1
    
    if pbar is not None:
        pbar.close()
    
    if num_examples is None:
        log.info(f"GEU score evaluation completed. Processed {processed_count} examples in total.")
    else:
        log.info(f"GEU score evaluation completed. Processed {processed_count}/{num_examples} examples.")


if __name__ == "__main__":
    geu_score(
        num_examples=100,
        engine_llm="gemini-2.5-flash-lite",
        rewrite_method_name="autogeo_api_researchy_geo_gemini",
        metrics_to_evaluate=["citation_quality", "quality_dimensions", "keypoint_coverage"],
        max_workers=10,
        data_dir="data/Researchy-GEO/test",
        text_list_name="text_list"
    )
