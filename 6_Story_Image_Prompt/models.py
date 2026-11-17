"""
models.py

Contains all shared data models for the Story Image Agent.

Responsibilities:
1. Graph state definition (LangGraph)
2. LLM structured output schemas
3. Evaluation result schemas

Keeping all models in a single file makes the project
easy to navigate and avoids excessive file creation.
"""

from typing import Optional
from typing import TypedDict

from pydantic import BaseModel
from pydantic import Field


# ==========================================================
# LANGGRAPH STATE
# ==========================================================

class StoryState(TypedDict, total=False):
    """
    Shared workflow state.

    This state is passed between all LangGraph nodes.

    Lifecycle:

    Image
      ↓
    Vision Analysis
      ↓
    Story Extraction
      ↓
    Evaluation
      ↓
    Quality Gate
      ↓
    Save Prompt
    """

    # Input
    image_b64: str

    # Story attributes extracted by Vision LLM
    story_name: str
    story_summary: str
    characters: list[str]
    setting: str
    moral: str

    # Final generated prompt
    image_prompt: str

    # Evaluation fields
    semantic_similarity: float
    hallucination_score: float
    prompt_quality: float

    characters_present: bool
    setting_present: bool
    moral_present: bool

    # Approval status
    approved: bool

    # Optional feedback
    reflection_feedback: str


# ==========================================================
# STRUCTURED VISION OUTPUT
# ==========================================================

class StoryExtraction(BaseModel):
    """
    Output schema expected from the vision model.

    The LLM should generate data matching this
    structure after analyzing the uploaded image.
    """

    story_name: str = Field(
        description="Name or title of the story."
    )

    story_summary: str = Field(
        description="Short summary of the story."
    )

    characters: list[str] = Field(
        description="Main characters visible in story."
    )

    setting: str = Field(
        description="Primary story setting."
    )

    moral: str = Field(
        description="Moral or lesson of the story."
    )

    image_prompt: str = Field(
        description="Detailed image generation prompt."
    )


# ==========================================================
# EVALUATION OUTPUT
# ==========================================================

class EvaluationResult(BaseModel):
    """
    Consolidated evaluation result.

    Produced after all evaluation steps
    have completed.
    """

    semantic_similarity: float

    hallucination_score: float

    prompt_quality: float

    characters_present: bool

    setting_present: bool

    moral_present: bool

    approved: bool


# ==========================================================
# PROMPT STORAGE MODEL
# ==========================================================

class SavedPrompt(BaseModel):
    """
    Metadata about a saved prompt.
    """

    story_name: str

    file_path: str

    prompt: str


# ==========================================================
# REFLECTION OUTPUT
# ==========================================================

class ReflectionFeedback(BaseModel):
    """
    Stores feedback when a prompt
    fails the quality check.
    """

    approved: bool

    feedback: Optional[str] = None