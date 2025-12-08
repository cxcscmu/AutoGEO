"""
Anthropic (Claude) API client for AutoGEO.
"""
import os
import time
from typing import Optional
from dotenv import load_dotenv
from anthropic import Anthropic
from .constants import COMMON_SYSTEM_PROMPT, MAX_RETRIES, RETRY_DELAY_SECONDS

load_dotenv("keys.env")

# Global client instance
_anthropic_client: Optional[Anthropic] = None


def _get_anthropic_client() -> Anthropic:
    """Get or create Anthropic client.
    
    Returns:
        Anthropic client instance
        
    Raises:
        ValueError: If ANTHROPIC_API_KEY not found in environment
    """
    global _anthropic_client
    if _anthropic_client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        _anthropic_client = Anthropic(api_key=api_key)
    return _anthropic_client


def call_claude(
    user_prompt: str, 
    model_name: str = "claude-3-5-sonnet-20241022", 
    temperature: float = 0.7,
    system_prompt: str = COMMON_SYSTEM_PROMPT
) -> str:
    """Call Anthropic (Claude) API with retry logic.
    
    Args:
        user_prompt: User prompt text
        model_name: Claude model name (default: "claude-3-5-sonnet-20241022")
        temperature: Sampling temperature (default: 0.7)
        system_prompt: System prompt text (default: COMMON_SYSTEM_PROMPT)
        
    Returns:
        Response text from Claude
        
    Raises:
        Exception: If API call fails after all retries
    """
    client = _get_anthropic_client()
    
    for attempt in range(MAX_RETRIES):
        try:
            print(f"Running Claude Model - Attempt {attempt + 1}/{MAX_RETRIES}")
            messages = [
                {"role": "user", "content": user_prompt}
            ]
            response = client.messages.create(
                model=model_name,
                messages=messages,
                system=system_prompt,
                max_tokens=4096,
                temperature=temperature
            )
            print("Response Done")
            return response.content[0].text
        except Exception as e:
            print(f"Error in calling Anthropic API: {e}")
            if attempt < MAX_RETRIES - 1:
                print(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                print(f"API call failed after {MAX_RETRIES} attempts.")
                raise


