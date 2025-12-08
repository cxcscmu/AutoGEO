import time
import os
import google.generativeai as genai
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
from typing import List

load_dotenv("keys.env")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


query_prompt = """Write an accurate and concise answer for the given user question, using _only_ the provided summarized web search results. The answer should be correct, high-quality, and written by an expert using an unbiased and journalistic tone. The user's language of choice such as English, Français, Español, Deutsch, or 日本語 should be used. The answer should be informative, interesting, and engaging. The answer's logic and reasoning should be rigorous and defensible. Every sentence in the answer should be _immediately followed_ by an in-line citation to the search result(s). The cited search result(s) should fully support _all_ the information in the sentence. Search results need to be cited using [index]. When citing several search results, use [1][2][3] format rather than [1, 2, 3]. You can use multiple search results to respond comprehensively while avoiding irrelevant search results.

Question: {query}

Search Results:
{source_text}
"""


MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 15  


def generate_answer_gemini(query: str, sources: List[str], model_name: str = 'gemini-2.5-flash-lite') -> str:
    """Generate answer using Gemini with RAG.
    
    Args:
        query: User query text
        sources: List of source document texts
        model_name: Gemini model name (default: 'gemini-2.5-flash-lite')
        
    Returns:
        Generated answer text with citations
        
    Raises:
        Exception: If API call fails after all retries
    """
    source_text = '\n\n'.join([f'### Source {idx}:\n{source}' for idx, source in enumerate(sources)])
    prompt = query_prompt.format(query=query, source_text=source_text)
    for attempt in range(MAX_RETRIES):
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                raise

def generate_answer_gpt(query: str, sources: List[str], model_name: str = 'gpt-4o-mini') -> str:
    """Generate answer using GPT with RAG.
    
    Args:
        query: User query text
        sources: List of source document texts
        model_name: OpenAI model name (default: 'gpt-4o-mini')
        
    Returns:
        Generated answer text with citations
        
    Raises:
        Exception: If API call fails after all retries
    """
    source_text = '\n\n'.join([f'### Source {idx}:\n{source}\n\n' for idx, source in enumerate(sources)])
    prompt = query_prompt.format(query=query, source_text=source_text)
    messages = [{"role": "user", "content": prompt}]
    for attempt in range(MAX_RETRIES):
        try:
            response = openai_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.5 
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                raise Exception(f"Failed to generate answer with GPT after {MAX_RETRIES} attempts: {e}")

def generate_answer_claude(query: str, sources: List[str], model_name: str = 'claude-3-haiku-20240307') -> str:
    """Generate answer using Claude with RAG.
    
    Args:
        query: User query text
        sources: List of source document texts
        model_name: Claude model name (default: 'claude-3-haiku-20240307')
        
    Returns:
        Generated answer text with citations
        
    Raises:
        Exception: If API call fails after all retries
    """
    source_text = '\n\n'.join([f'### Source {idx}:\n{source}\n\n' for idx, source in enumerate(sources)])
    system_prompt = query_prompt.split("Question: {query}")[0].strip()
    user_content = f"Question: {query}\n\nSearch Results:\n{source_text}"
    messages = [{"role": "user", "content": user_content}]
    for attempt in range(MAX_RETRIES):
        try:
            response = anthropic_client.messages.create(
                model=model_name,
                system=system_prompt,
                messages=messages,
                max_tokens=4096, 
                temperature=0.5
            )
            return response.content[0].text
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                raise Exception(f"Failed to generate answer with Claude after {MAX_RETRIES} attempts: {e}")

