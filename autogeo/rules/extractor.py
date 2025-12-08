import json
import re
from typing import List, Optional
from .llm_client import get_llm_response 


def get_rule_extraction_prompt(explanation_text: str, winner: str) -> str:
    """Generate prompt for extracting rules from explanation text.
    
    Args:
        explanation_text: Explanation text from LLM
        winner: Winner document identifier ("doc_a" or "doc_b")
        
    Returns:
        Formatted prompt string for rule extraction
    """
    winner_name = "Document A" if winner == "doc_a" else "Document B"
    few_shot = """
Example 1:
["The document should directly address the core question posed by the user query."]
Example 2:
["The document should use clear headings and lists to structure information for easy parsing.", "The document should provide specific, actionable details rather than general, high-level statements."]
"""
    return f"""
[Instruction]
Based on the following explanation about why {winner_name} was preferred, extract a set of general, reusable rules that define a high-quality source document for a RAG system.
These rules should be objective and deterministic principles. Below are a few examples:
{few_shot}

Return the list as a JSON array of strings. Do not use ```json```. Output the JSON array directly. If no clear rules can be extracted, return an empty JSON array [].

[Explanation]
{explanation_text}
"""


def _fix_json_string(text: str) -> str:
    """Fix common JSON issues in LLM responses, especially unescaped quotes.
    
    Args:
        text: JSON string that may have formatting issues
        
    Returns:
        Fixed JSON string
    """
    # Remove markdown code blocks
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    
    # Find the JSON array boundaries
    first_bracket = text.find('[')
    if first_bracket == -1:
        return text
    
    last_bracket = text.rfind(']')
    if last_bracket == -1 or last_bracket < first_bracket:
        return text
    
    json_part = text[first_bracket:last_bracket + 1]
    
    # Fix unescaped quotes inside string values
    # Strategy: parse character by character, tracking string boundaries
    result = []
    i = 0
    in_string = False
    escape_next = False
    
    while i < len(json_part):
        char = json_part[i]
        
        if escape_next:
            result.append(char)
            escape_next = False
            i += 1
            continue
        
        if char == '\\':
            result.append(char)
            escape_next = True
            i += 1
            continue
        
        if char == '"':
            if not in_string:
                # Opening quote - start of a string
                in_string = True
                result.append(char)
            else:
                # Inside a string - check if this is a closing quote
                # Look ahead (skip whitespace) to determine context
                j = i + 1
                while j < len(json_part) and json_part[j] in [' ', '\n', '\r', '\t']:
                    j += 1
                
                # If next non-whitespace char is comma, bracket, or end, it's a closing quote
                if j >= len(json_part) or json_part[j] in [',', ']']:
                    in_string = False
                    result.append(char)
                else:
                    # Quote inside string value - needs escaping
                    result.append('\\"')
            i += 1
            continue
        
        result.append(char)
        i += 1
    
    return ''.join(result)


def get_extracted_rules(explanation_text: str, winner: str, *, llm_args: dict) -> Optional[List[str]]:
    """Extract rules from explanation text using LLM.
    
    Args:
        explanation_text: Explanation text from LLM
        winner: Winner document identifier ("doc_a" or "doc_b")
        llm_args: Dictionary of LLM arguments (model, api keys, etc.)
        
    Returns:
        List of extracted rules, or empty list if extraction fails
    """
    if not explanation_text:
        return []
    prompt = get_rule_extraction_prompt(explanation_text, winner)
    extracted_text = get_llm_response(prompt=prompt, **llm_args)
    
    # Try multiple parsing strategies
    cleaned_text = _fix_json_string(extracted_text)
    
    try:
        rules = json.loads(cleaned_text)
        if isinstance(rules, list):
            # Filter out empty strings and None values
            rules = [r for r in rules if r and isinstance(r, str) and r.strip()]
            return rules
        return []
    except json.JSONDecodeError as e:
        # Try to extract rules manually using regex as fallback
        try:
            # Pattern to match strings in JSON array: "..." or '...'
            # This is a fallback when JSON parsing fails
            pattern = r'"([^"\\]*(\\.[^"\\]*)*)"'
            matches = re.findall(pattern, cleaned_text)
            if matches:
                # Extract the full matched strings
                full_pattern = r'"((?:[^"\\]|\\.)*)"'
                rules = re.findall(full_pattern, cleaned_text)
                # Unescape the strings
                rules = [r.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t') 
                        for r in rules if r.strip()]
                if rules:
                    return rules
        except Exception:
            pass
        
        # If all else fails, log warning and return empty
        # This is just a warning - the example will be skipped but processing continues
        print(f"[WARNING] Failed to parse LLM response as JSON: {str(e)[:100]}")
        print(f"         Response preview: {extracted_text[:150]}...")
        print(f"         This example will be skipped. Processing continues normally.")
        return []
    except (TypeError, AttributeError) as e:
        # This is also just a warning
        print(f"[WARNING] Unexpected error parsing LLM response: {str(e)[:100]}")
        print(f"         This example will be skipped. Processing continues normally.")
        return []
