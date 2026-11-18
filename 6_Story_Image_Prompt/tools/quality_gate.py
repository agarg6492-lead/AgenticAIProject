"""
Quality Gate Tool.

Final approval mechanism.

Decides whether a generated
prompt should be:

- Saved
- Rejected
- Re-generated

Based on configurable thresholds.
"""

from langchain_core.tools import tool

from config import (
    SEMANTIC_SIMILARITY_THRESHOLD,
    HALLUCINATION_THRESHOLD,
    PROMPT_QUALITY_THRESHOLD
)

from utils.logging import get_logger

logger = get_logger(__name__)


@tool
def quality_gate(
    semantic_similarity: float,
    hallucination_score: float,
    prompt_quality: float,
    characters_present: bool,
    setting_present: bool,
    moral_present: bool
) -> bool:
    """
    Approve or reject prompt.

    Returns:
        True -> Approved

        False -> Rejected
    """

    approved = (

        semantic_similarity >=
        SEMANTIC_SIMILARITY_THRESHOLD

        and

        hallucination_score >=
        HALLUCINATION_THRESHOLD

        and

        prompt_quality >=
        PROMPT_QUALITY_THRESHOLD

        and

        characters_present

        and

        setting_present

        and

        moral_present
    )

    logger.info(
        f"Prompt Approved: {approved}"
    )

    return approved