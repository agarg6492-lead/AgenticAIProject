"""
Reflection Tool.

Provides prompt improvement feedback.

Invoked when quality gate fails.
"""

from langchain_core.tools import tool


@tool
def reflection_tool(
    evaluation: dict
) -> str:
    """
    Generate prompt improvement guidance.

    Args:
        evaluation:
            Evaluation metrics.

    Returns:
        Feedback string.
    """

    issues = []

    if not evaluation[
        "characters_present"
    ]:
        issues.append(
            "Missing story characters."
        )

    if not evaluation[
        "setting_present"
    ]:
        issues.append(
            "Missing setting."
        )

    if not evaluation[
        "moral_present"
    ]:
        issues.append(
            "Missing moral."
        )

    if (
        evaluation["prompt_quality"]
        < 9
    ):
        issues.append(
            "Prompt lacks visual richness."
        )

    return "\n".join(
        issues
    )