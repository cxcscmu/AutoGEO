"""
Unified document rewriter for AutoGEO using preference rules.
"""
import os
import json
from typing import Optional, List, Tuple

from ..config import Dataset, LLMName
from ..utils import call_gemini


def _load_rules_from_file(
    dataset: str,
    engine_llm: str,
    rule_path: Optional[str] = None
) -> Tuple[Optional[List[str]], Optional[str]]:
    """Load extracted rules from file if available.
    
    Args:
        dataset: Name of the dataset
        engine_llm: External generative engine name
        rule_path: Optional explicit rule file path
        
    Returns:
        Tuple of (rules, rule_file_path) where rules is List of rules if file exists, None otherwise,
        and rule_file_path is the path to the rule file used (or None if using defaults)
    """
    if rule_path:
        try:
            with open(rule_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict) and 'filtered_rules' in data:
                    return data['filtered_rules'], rule_path
                if isinstance(data, list):
                    return data, rule_path
        except Exception as exc:
            print(f"Warning: Failed to load rules from custom path {rule_path}: {exc}")

    import glob
    
    # Try multiple possible paths
    possible_paths = [
        f"data/{dataset}/rule_sets/{engine_llm}*/merged_rules.json",
    ]
    
    for pattern in possible_paths:
        matches = glob.glob(pattern)
        if matches:
            # Use the most recent match
            latest_file = max(matches, key=os.path.getmtime)
            try:
                with open(latest_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and 'filtered_rules' in data:
                        return data['filtered_rules'], latest_file
                    elif isinstance(data, list):
                        return data, latest_file
            except Exception as e:
                print(f"Warning: Could not load rules from {latest_file}: {e}")
                continue
    
    return None, None


def _get_default_rules(dataset: str, engine_llm: str) -> List[str]:
    """Get default rules for a dataset when extracted rules are unavailable.
    
    These are fallback rules used when extracted rules are not available.
    """
    # Default rules for Researchy-GEO (research queries) gemini (gemini-2.5-flash-lite)
    autogeo_researchy_geo_gemini_rules = [
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
    
    # Default rules for GEO-Bench (research queries) gemini (gemini-2.5-flash-lite)
    autogeo_geo_bench_gemini_rules = [
        "Ensure all information is factually accurate and verifiable, citing credible sources.",
        "Ensure information is current and up-to-date, especially for time-sensitive topics.",
        "Ensure the document is self-contained and comprehensive, providing all necessary context and sub-topic information.",
        "Explain the underlying mechanisms and principles (the 'why' and 'how'), not just surface-level facts.",
        "Maintain a singular focus on the core topic, excluding tangential information, promotional content, and document 'noise' (e.g., navigation, ads).",
        "Organize content with a clear, logical hierarchy, using elements like headings, lists, and tables.",
        "Present a balanced and objective view on debatable topics, including multiple significant perspectives.",
        "Provide specific, actionable guidance, such as step-by-step instructions, for procedural topics.",
        "State the primary conclusion directly at the beginning of the document.",
        "Use clear and unambiguous language, defining technical terms, acronyms, and jargon upon first use.",
        "Use specific, concrete details and examples instead of abstract generalizations.",
        "Write concisely, eliminating verbose language, redundancy, and filler content."
    ]
    
    # Default rules for E-commerce (commercial queries) gemini (gemini-2.5-flash-lite)
    autogeo_ecommerce_gemini_rules = [
        "Ensure all information is factually accurate, verifiable, and current for the topic.",
        "Establish credibility by citing authoritative sources, providing evidence, or demonstrating clear expertise.",
        "Justify recommendations and claims with clear reasoning, context, or comparative analysis like pros and cons.",
        "Organize content with a clear, logical structure using elements like headings, lists, and tables to facilitate scanning and parsing.",
        "Present information objectively, avoiding promotional bias and including balanced perspectives where applicable.",
        "Provide actionable information, such as step-by-step instructions or clear recommendations.",
        "Provide specific, verifiable details such as names, model numbers, technical specifications, and quantifiable data.",
        "Structure content into modular, self-contained units, such as distinct paragraphs or list items for each concept.",
        "Use clear, simple, and unambiguous language, defining any necessary technical terms or jargon.",
        "Write concisely, eliminating verbose language, filler content, and unnecessary repetition."
    ]

    # Default rules for Researchy-GEO (research queries) gpt (gpt-4o-mini)
    autogeo_researchy_geo_gpt_rules = [
        "Attribute all claims to specific, credible, and authoritative sources.",
        "Create a self-contained document, free from non-informational content like advertisements, navigation, or paywalls.",
        "Ensure all content is strictly relevant to the core topic, excluding tangential or unrelated information.",
        "Ensure all information is factually accurate, verifiable, and internally consistent.",
        "Ensure content is fully accessible without requiring logins, subscriptions, or payments.",
        "Ensure information is current and up-to-date, especially for time-sensitive topics.",
        "Explain underlying mechanisms and causal relationships (the 'how' and 'why'), not just descriptive facts.",
        "Maintain a neutral and objective tone, prioritizing factual information over subjective opinions or biased language.",
        "Maintain a purely informational purpose, avoiding promotional, persuasive, or interactive content.",
        "Organize content with a clear, logical structure, using elements like headings and lists to improve readability.",
        "Present a balanced perspective on complex topics by including multiple relevant viewpoints or counterarguments.",
        "Present information with a cohesive, logical flow, avoiding fragmented or contradictory statements.",
        "Provide comprehensive coverage of the topic, addressing its key facets, nuances, and relevant context.",
        "Provide specific, actionable guidance when the topic involves a task or problem-solving.",
        "State the key conclusion directly at the beginning of the document.",
        "Substantiate claims with specific evidence, such as quantifiable data or concrete examples.",
        "Use clear, concise, and unambiguous language, defining essential jargon and eliminating filler content."
    ]
    
    # Default rules for GEO-Bench (research queries) gpt (gpt-4o-mini)
    autogeo_geo_bench_gpt_rules = [
        "Address the topic comprehensively, covering all essential sub-topics and necessary context.",
        "Define essential terms, acronyms, and jargon upon their first use.",
        "Ensure all factual information is accurate, verifiable, and internally consistent.",
        "Ensure content is free from illegal, unethical, or harmful information.",
        "Ensure each document is self-contained, providing all necessary information on the topic without requiring external links.",
        "Explain the 'why' and 'how' behind facts, clarifying underlying principles and mechanisms.",
        "Explicitly differentiate between similar or easily confused concepts.",
        "For complex or debatable subjects, present multiple significant viewpoints in a balanced way.",
        "For procedural content, provide clear, numbered, step-by-step instructions.",
        "For time-sensitive topics, ensure information is current and clearly display its publication or last-updated date.",
        "Maintain a neutral, objective tone, clearly distinguishing facts from opinions.",
        "Maintain a singular focus on the core topic, excluding tangential or promotional content.",
        "Organize content with a clear, logical hierarchy using headings, lists, and tables.",
        "State the primary conclusion at the beginning of the document.",
        "Structure content into atomic units, where each paragraph or section addresses a single idea.",
        "Use clear, simple, and unambiguous language.",
        "Use concrete examples, analogies, or case studies to illustrate complex concepts.",
        "Use specific, concrete details like names, dates, and statistics instead of generalizations.",
        "Write concisely, eliminating repetition, filler words, and verbose phrasing."
    ]
    
    # Default rules for E-commerce (commercial queries) gpt (gpt-4o-mini)
    autogeo_ecommerce_gpt_rules = [
        "Be complete and thorough, covering all key aspects and a sufficient range of options.",
        "Clearly define the document's scope, especially for broad or ambiguous topics.",
        "Ensure all factual information is accurate, verifiable, and objective.",
        "Ensure information is up-to-date, clearly indicating its publication or last-updated date.",
        "Ensure the document is a complete, self-contained unit, not truncated or missing essential information.",
        "Establish credibility by citing authoritative sources or explaining the methodology for arriving at conclusions.",
        "Maintain a neutral tone, free from bias, promotional language, and unsubstantiated claims.",
        "Organize content logically with a clear, hierarchical structure using elements like headings, lists, and tables for easy parsing.",
        "Present information concisely, eliminating verbose language, filler words, and unnecessary introductions.",
        "Prioritize the most critical information by placing it at the beginning of the document or relevant section.",
        "Provide actionable content, such as step-by-step instructions or clear recommendations.",
        "Provide context and explain the reasoning behind recommendations, conclusions, or complex information.",
        "Structure data in a way that allows for direct evaluation, such as in a table or a pros-and-cons list.",
        "Use simple, direct, and unambiguous language, defining any necessary technical jargon.",
        "Use specific, quantifiable details like names, metrics, and technical specifications instead of vague generalizations."
    ]

    # Default rules for Researchy-GEO (research queries) claude (claude-3-haiku)
    autogeo_researchy_geo_claude_rules = [
        "Cover the topic comprehensively by addressing all its key facets and relevant sub-topics.",
        "Dedicate each paragraph or self-contained section to a single, distinct idea.",
        "Ensure a cohesive narrative flow where ideas connect logically rather than appearing as disconnected facts.",
        "Ensure all information is factually accurate, internally consistent, and up-to-date.",
        "Ensure the document is self-contained, providing all necessary context without requiring readers to follow external links.",
        "Ensure the full text is programmatically accessible, without requiring logins, paywalls, or user interaction.",
        "Focus exclusively on a single topic, removing all tangential information, advertisements, and navigational elements.",
        "Illustrate concepts and support arguments with specific details, concrete examples, or data.",
        "Maintain a neutral, objective tone, clearly distinguishing facts from opinions and avoiding biased or promotional language.",
        "Organize content with a clear, logical hierarchy using headings, lists, or tables to facilitate machine parsing.",
        "Present a balanced perspective on debatable topics by acknowledging multiple significant viewpoints or counterarguments.",
        "Provide clear, actionable steps or practical guidance for procedural topics.",
        "Provide explanatory depth by detailing the underlying mechanisms, causes, and effects ('how' and 'why').",
        "State the primary conclusion directly at the beginning of the document.",
        "Substantiate all claims with citations to credible, authoritative sources.",
        "Use clear and unambiguous language, defining specialized or technical terms upon their first use.",
        "Write concisely, eliminating repetitive phrasing, filler content, and unnecessary verbosity."
    ]
    
    # Default rules for GEO-Bench (research queries) claude (claude-3-haiku)
    autogeo_geo_bench_claude_rules = [
        "Cite authoritative sources to support claims and establish credibility.",
        "Cover the topic comprehensively, providing depth by explaining the underlying 'why' and 'how'.",
        "Ensure all information is factually accurate, verifiable, and internally consistent.",
        "Ensure each document is self-contained and can be understood without external context.",
        "Focus on a single topic, writing concisely and eliminating irrelevant or repetitive content.",
        "For task-oriented topics, provide actionable guidance like step-by-step instructions.",
        "Indicate the timeliness of information with clear publication or revision dates.",
        "Maintain a neutral, objective tone, prioritizing facts over opinions or promotional language.",
        "Present multiple perspectives and counterarguments for complex or debatable topics.",
        "Provide specific details, such as names, dates, statistics, and concrete examples, to support claims and illustrate concepts.",
        "Segment content into discrete units, where each paragraph or list item addresses a single idea.",
        "State the key conclusion at the beginning of the document.",
        "Use clear structural elements like headings, lists, and tables to organize content logically.",
        "Use clear, unambiguous language, and define technical terms or acronyms on their first use."
    ]
    
    # Default rules for E-commerce (commercial queries) claude (claude-3-haiku)
    autogeo_ecommerce_claude_rules = [
        "Eliminate all tangential or promotional information.",
        "Ensure all information is factually accurate and verifiable, supporting claims with citations to authoritative sources.",
        "Ensure core content is directly accessible, without requiring logins, paywalls, or complex navigation.",
        "Keep information current for time-sensitive topics and clearly state its timeliness.",
        "Maintain an objective, neutral tone and present a balanced perspective, including relevant pros and cons or alternative viewpoints where applicable.",
        "Maintain internal consistency in terminology, formatting, and data presentation, especially for comparable items.",
        "Organize content using a clear, logical, and consistent structure with elements like headings, lists, and tables to facilitate automated parsing.",
        "Provide actionable content, such as step-by-step instructions or direct recommendations.",
        "Provide context or rationale to explain the reasoning behind data, recommendations, or claims.",
        "Structure content into discrete, self-contained units, with each paragraph or section addressing a single concept.",
        "The document should provide the complete core information, without requiring navigation to external links for essential information.",
        "Use specific, quantifiable details like names, model numbers, and metrics instead of vague generalizations.",
        "Write with clarity and conciseness, using simple, direct language and eliminating unnecessary jargon, repetition, and filler."
    ]
    
    # Map dataset names to rules
    dataset_rules_map = {
        (Dataset.RESEARCHY_GEO, LLMName.GEMINI): autogeo_researchy_geo_gemini_rules,
        (Dataset.GEO_BENCH, LLMName.GEMINI): autogeo_geo_bench_gemini_rules,
        (Dataset.ECOMMERCE, LLMName.GEMINI): autogeo_ecommerce_gemini_rules,
        (Dataset.RESEARCHY_GEO, LLMName.GPT): autogeo_researchy_geo_gpt_rules,
        (Dataset.GEO_BENCH, LLMName.GPT): autogeo_geo_bench_gpt_rules,
        (Dataset.ECOMMERCE, LLMName.GPT): autogeo_ecommerce_gpt_rules,
        (Dataset.RESEARCHY_GEO, LLMName.CLAUDE): autogeo_researchy_geo_claude_rules,
        (Dataset.GEO_BENCH, LLMName.CLAUDE): autogeo_geo_bench_claude_rules,
        (Dataset.ECOMMERCE, LLMName.CLAUDE): autogeo_ecommerce_claude_rules,
    }
    
    # Convert string to enum
    try:
        dataset_enum = Dataset(dataset)
        
        # Extract LLM type from full model name (e.g., "gemini-2.5-flash-lite" -> "gemini")
        engine_llm_lower = engine_llm.lower()
        engine_llm_enum = None
        
        # Check if engine_llm contains any of the LLMName enum values
        for llm_name in LLMName:
            if llm_name.value in engine_llm_lower:
                engine_llm_enum = llm_name
                break
        
        # If no match found, try direct conversion
        if engine_llm_enum is None:
            engine_llm_enum = LLMName(engine_llm_lower)
        
        return dataset_rules_map.get((dataset_enum, engine_llm_enum), autogeo_researchy_geo_gemini_rules)
    except (ValueError, KeyError):
        print("No rules found, using default rules", {"dataset": dataset, "engine_llm": engine_llm})
        return autogeo_researchy_geo_gemini_rules


def rewrite_document(
    document: str,
    dataset: str = None,
    engine_llm: str = None,
    rule_path: Optional[str] = None
) -> str:
    """Rewrite a document using AutoGEO rules.
    
    Args:
        document: Original document text to rewrite
        dataset: Name of the dataset (Researchy-GEO, E-commerce, or GEO-Bench)
        engine_llm: External generative engine LLM (gemini, gpt, or claude)
        rule_path: Optional explicit rule file path
        
    Returns:
        Rewritten document text
    """
    # Get rules (try extracted first, fallback to defaults)
    rules, rule_file_path = _load_rules_from_file(dataset, engine_llm, rule_path)
    
    if not rules:
        # print("No rules found, using default rules")
        rules = _get_default_rules(dataset, engine_llm)
        rule_file_path = None
    
    # Format rules as string
    rules_string = "- " + "\n- ".join(rules)
    
    # Create prompt
    user_prompt = f"""Here is the source:
{document}

You are given a website document as a source. This source, along with other sources, will be used by a language model (LLM) to generate answers to user questions, with each line in the generated answer being cited with its original source. Your task, as the owner of the source, is to **rewrite your document in a way that maximizes its visibility and impact in the LLM's final answer, ensuring your source is more likely to be quoted and cited**.

You can regenerate the provided source so that it strictly adheres to the "Quality Guidelines", and you can also apply any other methods or techniques, as long as they help your rewritten source text rank higher in terms of relevance, authority, and impact in the LLM's generated answers.

## Quality Guidelines to Follow:

{rules_string}
""".strip()
    
    return call_gemini(user_prompt, model_name="gemini-2.5-pro")


def get_rewrite_prompt_template(dataset: str, engine_llm: str, 
                                use_extracted_rules: bool = True) -> str:
    """Get the rewrite prompt template without calling the API.
    
    Useful for testing or when you only need the prompt.
    
    Args:
        dataset: Name of the dataset
        engine_llm: Engine LLM name
        use_extracted_rules: Whether to try loading extracted rules
        
    Returns:
        Prompt template string
    """
    rules = None
    if use_extracted_rules:
        rules = _load_rules_from_file(dataset, engine_llm)
    
    if not rules:
        rules = _get_default_rules(dataset, engine_llm)
    
    rules_string = "- " + "\n- ".join(rules)
    
    return f"""You are given a website document as a "source". This source, along with other sources, will be used by a language model (LLM) to generate answers to user questions, with each line in the generated answer being cited with its original source. Your task, as the owner of the source, is to **rewrite your document in a way that maximizes its visibility and impact in the LLM's final answer, ensuring your source is more likely to be quoted and cited**.

You can regenerate the provided "source" so that it strictly adheres to the "Quality Guidelines", and you can also apply any other methods or techniques, as long as they help your rewritten source text rank higher in terms of relevance, authority, and impact in the LLM's generated answers.

## Quality Guidelines to Follow:

{rules_string}
""".strip()

