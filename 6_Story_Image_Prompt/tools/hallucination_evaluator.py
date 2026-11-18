"""
Hallucination Evaluation Tool.

Measures how closely the generated
image prompt aligns with story facts.

Current Version:
Keyword overlap methodology.

Future Versions:
- LLM Judge
- Natural Language Inference
- Graph Based Validation
"""

from langchain_core.tools import tool

from utils.logging import get_logger

logger = get_logger(__name__)


@tool
def evaluate_hallucination(
    prompt: str,
    story_context: str
) -> float:
    """
    Compute hallucination score.

    Score Range:
        0.0 -> Highly hallucinated
        1.0 -> Fully grounded

    Args:
        prompt:
            Generated prompt.

        story_context:
            Story summary.

    Returns:
        Hallucination score.
    """

    context_words = set(
        story_context.lower().split()
    )

    prompt_words = set(
        prompt.lower().split()
    )

    overlap = (
        len(
            context_words
            .intersection(
                prompt_words
            )
        )
    )

    total = max(
        len(context_words),
        1
    )

    score = overlap / total

    logger.info(
        f"Hallucination Score: {score}"
    )

    return score