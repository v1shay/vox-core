# LLM structured output module
"""
LLM structured output module for formatting transcripts.

Provides mode-specific prompts and OpenAI API integration for
structured output generation.
"""
from dotenv import load_dotenv
import os
import openai
from typing import Optional, Dict

load_dotenv() 

# API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_prompt_template(mode: str) -> str:
    """
    Get prompt template for the specified mode.
    
    Args:
        mode: One of 'meeting', 'study', 'recitation', 'interview'
    
    Returns:
        Prompt template string
    """
    templates = {
        "meeting": """You are a meeting notes assistant. Structure the following transcript into organized meeting notes.

Format the output as:
## Meeting Notes

### Key Points
- [Main points discussed]

### Action Items
- [Tasks and assignments]

### Decisions Made
- [Decisions and outcomes]

### Next Steps
- [Follow-up actions]

Transcript:
{transcript}""",

        "study": """You are a study notes assistant. Structure the following transcript into organized study notes.

Format the output as:
## Study Notes

### Main Concepts
- [Key concepts covered]

### Important Details
- [Specific details and facts]

### Questions & Answers
- [Questions raised and answers provided]

### Summary
[Brief summary of the content]

Transcript:
{transcript}""",

        "recitation": """You are a recitation notes assistant. Structure the following transcript into organized recitation notes.

Format the output as:
## Recitation Notes

### Topics Covered
- [Main topics discussed]

### Key Points
- [Important points made]

### Examples & Practice
- [Examples or practice problems covered]

### Notes
[Additional notes and observations]

Transcript:
{transcript}""",

        "interview": """You are an interview notes assistant. Structure the following transcript into organized interview notes.

Format the output as:
## Interview Notes

### Questions Asked
- [Questions posed during the interview]

### Responses
- [Key responses and answers]

### Key Points Discussed
- [Important topics and points]

### Follow-up Items
- [Items to follow up on]

Transcript:
{transcript}"""
    }
    
    if mode not in templates:
        raise ValueError(f"Unknown mode: {mode}. Must be one of: {list(templates.keys())}")
    
    return templates[mode]


def format_transcript(mode: str, transcript: str) -> Optional[str]:
    """
    Format transcript using OpenAI API with mode-specific prompt.
    
    Args:
        mode: One of 'meeting', 'study', 'recitation', 'interview'
        transcript: Raw transcript text to format
    
    Returns:
        Formatted structured output, or None if error
    """
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. "
            "Please set it with: export OPENAI_API_KEY='your-key-here'"
        )
    
    if len(transcript.strip()) < 50:
        return None  # Skip if transcript too short
    
    # Get prompt template and format with transcript
    prompt_template = get_prompt_template(mode)
    prompt = prompt_template.format(transcript=transcript)
    
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that structures transcripts into organized notes."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        return response.choices[0].message.content.strip()
        
    except openai.AuthenticationError:
        raise ValueError("Invalid OpenAI API key. Please check your OPENAI_API_KEY environment variable.")
    except openai.APIError as e:
        raise ValueError(f"OpenAI API error: {e}")
    except Exception as e:
        raise ValueError(f"Error calling OpenAI API: {e}")
