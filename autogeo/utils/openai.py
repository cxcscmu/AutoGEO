"""
OpenAI API client for AutoGEO.
"""
import os
import time
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI
from .constants import COMMON_SYSTEM_PROMPT, MAX_RETRIES, RETRY_DELAY_SECONDS

load_dotenv("keys.env")

# Global client instance
_openai_client: Optional[OpenAI] = None


def _get_openai_client() -> OpenAI:
    """Get or create OpenAI client.
    
    Returns:
        OpenAI client instance
        
    Raises:
        ValueError: If OPENAI_API_KEY not found in environment
    """
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def call_openai(
    user_prompt: str, 
    model_name: str = "gpt-4o", 
    temperature: float = 0.7,
    system_prompt: str = COMMON_SYSTEM_PROMPT
) -> str:
    """Call OpenAI API with retry logic.
    
    Args:
        user_prompt: User prompt text
        model_name: OpenAI model name (default: "gpt-4o")
        temperature: Sampling temperature (default: 0.7)
        system_prompt: System prompt text (default: COMMON_SYSTEM_PROMPT)
        
    Returns:
        Response text from OpenAI
        
    Raises:
        Exception: If API call fails after all retries
    """
    client = _get_openai_client()
    
    for attempt in range(MAX_RETRIES):
        try:
            print(f"Running OpenAI Model - Attempt {attempt + 1}/{MAX_RETRIES}")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature
            )
            print("Response Done")
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in calling OpenAI API: {e}")
            if attempt < MAX_RETRIES - 1:
                print(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                print(f"API call failed after {MAX_RETRIES} attempts.")
                raise


