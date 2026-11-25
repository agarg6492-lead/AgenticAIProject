"""
config.py

Centralized configuration file for the
Story Image Agent project.

Responsibilities:
- Load environment variables
- Configure model settings
- Define evaluation thresholds
- Configure output locations
- Store project-wide constants

This approach prevents hardcoding
values throughout the codebase.
"""

from pathlib import Path
import os

from dotenv import load_dotenv

# ==========================================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================================

load_dotenv()

# ==========================================================
# PROJECT PATHS
# ==========================================================

BASE_DIR = Path('.')

OUTPUT_DIR = BASE_DIR / "outputs"

# Create output directory automatically
OUTPUT_DIR.mkdir(
    exist_ok=True
)

# ==========================================================
# GROQ CONFIGURATION
# ==========================================================

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)

VISION_MODEL = os.getenv(
    "VISION_MODEL",
    "qwen/qwen3.8-27b"
)

MODEL_TEMPERATURE = 0

# ==========================================================
# LANGSMITH CONFIGURATION
# ==========================================================

LANGCHAIN_API_KEY = os.getenv(
    "LANGCHAIN_API_KEY"
)

LANGCHAIN_PROJECT = os.getenv(
    "LANGCHAIN_PROJECT",
    "StoryPromptAgent"
)

LANGCHAIN_TRACING_V2 = os.getenv(
    "LANGCHAIN_TRACING_V2",
    "true"
)

# ==========================================================
# EVALUATION THRESHOLDS
# ==========================================================

# Prompt must be semantically close to
# expected output.
SEMANTIC_SIMILARITY_THRESHOLD = 0.70

# Prompt must remain grounded in story.
HALLUCINATION_THRESHOLD = 0.70

# Minimum quality score assigned by
# evaluation layer.
PROMPT_QUALITY_THRESHOLD = 9.0

# ==========================================================
# AGENT SETTINGS
# ==========================================================

# Allow future reflection loops.
ENABLE_REFLECTION = False

# Future use if iterative regeneration
# is implemented.
MAX_REFLECTION_ATTEMPTS = 3

# ==========================================================
# FILE NAMING
# ==========================================================

PROMPT_SUFFIX = "_prompt.txt"

# ==========================================================
# LOGGING
# ==========================================================

LOG_FORMAT = (
    "%(asctime)s | "
    "%(levelname)s | "
    "%(name)s | "
    "%(message)s"
)

# ==========================================================
# SYSTEM PROMPT
# ==========================================================

VISION_SYSTEM_PROMPT = """
Analyze the uploaded story image.

Extract:

- Story Name
- Story Summary
- Characters
- Setting
- Moral
- Detailed Image Generation Prompt

Return a JSON object ONLY.

DO NOT:

- wrap the response in markdown
- use ```json
- use code fences
- add explanations
- add comments
- add introductory text
- add trailing text

The response must start with { and end with }.

Required JSON schema:
{
  "story_name": "",
  "story_summary": "",
  "characters": [],
  "setting": "",
  "moral": "",
  "image_prompt": ""
}
"""