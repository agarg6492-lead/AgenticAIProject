"""
LangGraph State Definition.

This file defines the shared workflow
state that travels across graph nodes.

Every node can read and update these
fields.
"""

from typing import TypedDict


class StoryState(TypedDict):
    """
    Shared graph state.

    Attributes:
        image_b64:
            Base64 encoded image.

        story_name:
            Story title extracted by agent.

        characters:
            Story characters.

        setting:
            Story location/environment.

        moral:
            Story moral.

        image_prompt:
            Final generated image prompt.

        semantic_similarity:
            Similarity score between
            expected and generated prompts.

        hallucination_score:
            Degree of factual grounding.

        prompt_quality:
            Overall prompt quality score.

        approved:
            Final approval status.
    """

    image_b64: str

    story_name: str

    characters: list[str]

    setting: str

    moral: str

    image_prompt: str

    semantic_similarity: float

    hallucination_score: float

    prompt_quality: float

    approved: bool