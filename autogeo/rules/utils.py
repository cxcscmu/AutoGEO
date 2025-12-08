import random
import json
from datasets import Dataset
from typing import Dict, Any

def load_engine_preference_dataset(file_path: str) -> Dataset:
    """Load preference dataset from JSON file.
    
    Args:
        file_path: Path to JSON file containing preference data
        
    Returns:
        Hugging Face Dataset object with preference examples
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    records = []
    for doc_id, content in data.items():
        records.append({
            "id": doc_id,
            "query": content["query"],
            "good_document": content["good_document"],
            "bad_document": content["bad_document"],
        })
    return Dataset.from_list(records)


def prepare_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare RAG example by randomly assigning document order.
    
    Args:
        example: Dictionary with query, good_document, and bad_document
        
    Returns:
        Dictionary with document_a, document_b, winner, and original content
    """
    try:
        # Validate required fields
        required_fields = ["id", "query", "good_document", "bad_document"]
        for field in required_fields:
            if field not in example:
                raise ValueError(f"Missing required field: {field}")
        
        if random.random() < 0.5:
            document_a = example["good_document"]
            document_b = example["bad_document"]
            winner = "doc_a"
        else:
            document_a = example["bad_document"]
            document_b = example["good_document"]
            winner = "doc_b"
        return {
            "id": example["id"],
            "query": example["query"],
            "document_a": document_a,
            "document_b": document_b,
            "winner": winner,
            "good_document_content": example["good_document"],
            "bad_document_content": example["bad_document"],
        }
    except Exception as e:
        # Return a minimal valid structure with error info to prevent subprocess crash
        example_id = example.get("id", "unknown")
        return {
            "id": example_id,
            "query": example.get("query", ""),
            "document_a": example.get("good_document", ""),
            "document_b": example.get("bad_document", ""),
            "winner": "doc_a",
            "good_document_content": example.get("good_document", ""),
            "bad_document_content": example.get("bad_document", ""),
            "_error": str(e)
        }
