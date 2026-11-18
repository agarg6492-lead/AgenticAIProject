"""
Save Prompt Tool.

Persists approved prompts to disk.

File Naming Strategy:

<story_name>_prompt.txt

Example:

Aladdin_prompt.txt
"""

from pathlib import Path

from langchain_core.tools import tool

from config import OUTPUT_DIR

from utils.logging import get_logger

logger = get_logger(__name__)


@tool
def save_prompt(
    story_name: str,
    prompt: str
) -> str:
    """
    Save approved image prompt.

    Args:
        story_name:
            Extracted story name.

        prompt:
            Final approved prompt.

    Returns:
        File path.
    """

    file_path = (
        Path(OUTPUT_DIR)
        /
        f"{story_name}_prompt.txt"
    )

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(prompt)

    logger.info(
        f"Prompt Saved: {file_path}"
    )

    return str(file_path)