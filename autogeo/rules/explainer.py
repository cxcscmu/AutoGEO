from typing import Tuple
from .llm_client import get_llm_response 


def get_explanation_prompt(
    query: str,
    document_a: str,
    document_b: str,
    winner: str,
) -> str:
    """Generate prompt for LLM to explain document preference.
    
    Args:
        query: User query text
        document_a: First document text
        document_b: Second document text
        winner: Winner document identifier ("doc_a" or "doc_b")
        
    Returns:
        Formatted prompt string for explanation generation
    """
    winner_name = "Document A" if winner == "doc_a" else "Document B"
    
    return f"""
[Task]
You are an expert AI analyst. Your task is to analyze two documents that were retrieved by a RAG (Retrieval-Augmented Generation) system to answer a user's query.

One document ("the winning document") was heavily used by the RAG system to generate its final answer, indicating a higher relevance or quality. The other document was used less.

Please provide a detailed explanation for why the RAG system likely preferred the winning document.

Consider factors such as:
- Directness: Does it directly answer the user's query?
- Completeness: Does it provide a comprehensive answer?
- Relevance: Is the content on-topic or does it contain irrelevant noise?
- Structure: Is the document well-structured (e.g., with headings, lists) making information easier to extract?
- Accuracy and Specificity: Is the information precise, using specific data or examples?
- Conciseness: Does it provide the necessary information without excessive verbosity?

[User Query]
{query}

[Document A]
{document_a}

[Document B]
{document_b}

[Winning Document]: {winner_name}

[Your Explanation]
Provide your analysis below, explaining the strengths of the winning document and the weaknesses of the other in relation to the user's query.
"""


def get_explanation_response(
    query: str,
    document_a: str,
    document_b: str,
    winner: str,
    *,
    llm_args: dict,
) -> Tuple[str, None]:
    """Get explanation response from LLM for document preference.
    
    Args:
        query: User query text
        document_a: First document text
        document_b: Second document text
        winner: Winner document identifier ("doc_a" or "doc_b")
        llm_args: Dictionary of LLM arguments (model, api keys, etc.)
        
    Returns:
        Tuple of (explanation_text, None)
    """
    prompt = get_explanation_prompt(query, document_a, document_b, winner)
    explanation_text = get_llm_response(prompt=prompt, **llm_args)
    return explanation_text, None