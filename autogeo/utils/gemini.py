"""
Gemini API client for AutoGEO.
"""
import os
import time
from dotenv import load_dotenv
import google.generativeai as genai
from .constants import COMMON_SYSTEM_PROMPT, MAX_RETRIES, RETRY_DELAY_SECONDS

load_dotenv("keys.env")

# Initialize Gemini client
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def call_gemini(
    user_prompt: str, 
    system_prompt: str = COMMON_SYSTEM_PROMPT, 
    model_name: str = "gemini-2.5-pro", 
    temperature: float = 0.7
) -> str:
    """Call Gemini API with retry logic.
    
    Args:
        user_prompt: User prompt text
        system_prompt: System prompt text (default: COMMON_SYSTEM_PROMPT)
        model_name: Gemini model name (default: "gemini-2.5-pro")
        temperature: Sampling temperature (default: 0.7)
        
    Returns:
        Response text from Gemini
        
    Raises:
        Exception: If API call fails after all retries or content is blocked
    """
    messages = f"system_prompt: {system_prompt}\n\nuser_prompt: {user_prompt}"
    
    for attempt in range(MAX_RETRIES):
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(messages)
            return response.text
        except Exception as e:
            error_str = str(e)
            # Check if it's a PROHIBITED_CONTENT error - don't retry for these
            if "PROHIBITED_CONTENT" in error_str or "block_reason" in error_str:
                raise Exception(f"Content blocked by Gemini API (PROHIBITED_CONTENT): {error_str}")
            
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                raise


