"""
LLM client wrapper for rules module.

This module provides a unified interface to LLM APIs, using the centralized
LLM clients from utils while supporting API key parameters for flexibility.
"""
import os
from typing import Optional
from ..utils import call_gemini, call_openai, call_claude


def get_llm_response(
    prompt: str,
    model: str,
    google_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
) -> str:
    """Get response from appropriate LLM based on model name.
    
    This function uses the centralized LLM clients from utils, with support
    for passing API keys as parameters. If API keys are provided, they are
    temporarily set in the environment for the API call.
    
    Args:
        prompt: Prompt text
        model: Model name (must contain 'gemini', 'gpt', or 'claude')
        google_api_key: Google API key for Gemini (optional, uses env var if not provided)
        openai_api_key: OpenAI API key for GPT (optional, uses env var if not provided)
        anthropic_api_key: Anthropic API key for Claude (optional, uses env var if not provided)
        
    Returns:
        Response text from the LLM
        
    Raises:
        ValueError: If model type is not supported
    """
    model_lower = model.lower()
    
    # Store original env vars
    original_env = {}
    
    try:
        # Temporarily set API keys in environment if provided
        if google_api_key:
            original_env['GOOGLE_API_KEY'] = os.environ.get('GOOGLE_API_KEY')
            os.environ['GOOGLE_API_KEY'] = google_api_key
        if openai_api_key:
            original_env['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY')
            os.environ['OPENAI_API_KEY'] = openai_api_key
        if anthropic_api_key:
            original_env['ANTHROPIC_API_KEY'] = os.environ.get('ANTHROPIC_API_KEY')
            os.environ['ANTHROPIC_API_KEY'] = anthropic_api_key
        
        # Call appropriate LLM based on model name
        if "gemini" in model_lower:
            # For Gemini, combine prompt as user_prompt (utils/gemini.py expects system + user)
            return call_gemini(user_prompt=prompt, model_name=model)
        elif "gpt" in model_lower:
            return call_openai(user_prompt=prompt, model_name=model)
        elif "claude" in model_lower:
            return call_claude(user_prompt=prompt, model_name=model)
        else:
            supported_models = "'gemini', 'gpt', 'claude'"
            raise ValueError(f"Unsupported model type: {model}. Supported keywords: {supported_models}.")
    finally:
        # Restore original environment variables
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value