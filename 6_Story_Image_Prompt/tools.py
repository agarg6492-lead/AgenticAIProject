"""
tools.py

All business tools used by the workflow.
"""

from pathlib import Path

from config import (
    OUTPUT_DIR,
    PROMPT_SUFFIX,
    SEMANTIC_SIMILARITY_THRESHOLD,
    HALLUCINATION_THRESHOLD,
    PROMPT_QUALITY_THRESHOLD
)


def evaluate_story_coverage(
    prompt: str,
    characters: list[str],
    setting: str,
    moral: str
) -> dict:
    """
    Verify whether generated prompt contains
    important story information.
    """

    prompt_lower = prompt.lower()

    characters_present = any(
        c.lower() in prompt_lower
        for c in characters
    )

    setting_present = (
        setting.lower()
        in prompt_lower
    )

    moral_present = any(
        word.lower() in prompt_lower
        for word in moral.split()
    )

    return {
        "characters_present":
            characters_present,

        "setting_present":
            setting_present,

        "moral_present":
            moral_present
    }


def evaluate_hallucination(
    prompt: str,
    story_summary: str
) -> float:
    """
    Simple hallucination detector based
    on keyword overlap.
    """

    prompt_words = set(
        prompt.lower().split()
    )

    story_words = set(
        story_summary.lower().split()
    )

    overlap = len(
        prompt_words.intersection(
            story_words
        )
    )

    return overlap / max(
        len(story_words),
        1
    )


def quality_gate(
    semantic_similarity: float,
    hallucination_score: float,
    prompt_quality: float,
    characters_present: bool,
    setting_present: bool,
    moral_present: bool
) -> bool:
    """
    Decide whether prompt should
    be saved.
    """

    return (

        semantic_similarity
        >=
        SEMANTIC_SIMILARITY_THRESHOLD

        and

        hallucination_score
        >=
        HALLUCINATION_THRESHOLD

        and

        prompt_quality
        >=
        PROMPT_QUALITY_THRESHOLD

        and
        characters_present

        and
        setting_present

        and
        moral_present
    )


def save_prompt(
    story_name: str,
    prompt: str
) -> str:
    """
    Save approved prompt.
    """

    filename = (
        OUTPUT_DIR
        /
        f"{story_name}"
        f"{PROMPT_SUFFIX}"
    )

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(prompt)

    return str(filename)
