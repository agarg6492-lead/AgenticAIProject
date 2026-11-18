"""
Story Extraction Tool.

Extracts structured story information from
the JSON response returned by the vision model.

Responsibilities:
- Extract Character List
- Extract Setting
- Extract Moral
- Extract Story Name

Input:
Vision LLM JSON

Output:
Structured Story Information
"""

from langchain_core.tools import tool

from utils.logging import get_logger

logger = get_logger(__name__)


@tool
def extract_story_data(
    story_json: dict
) -> dict:
    """
    Extract structured story attributes.

    Args:
        story_json:
            Structured JSON generated
            by Qwen Vision Model.

    Returns:
        Dictionary containing:

        - story_name
        - characters
        - setting
        - moral
    """

    logger.info(
        "Extracting story metadata."
    )

    result = {
        "story_name":
            story_json.get(
                "story_name",
                ""
            ),

        "characters":
            story_json.get(
                "characters",
                []
            ),

        "setting":
            story_json.get(
                "setting",
                ""
            ),

        "moral":
            story_json.get(
                "moral",
                ""
            )
    }

    logger.info(
        "Story extraction completed."
    )

    return result