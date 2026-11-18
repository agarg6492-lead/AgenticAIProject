"""
Story Vision Agent.

Responsible for:

1. Reading story image.
2. Understanding story.
3. Extracting:
    - Characters
    - Setting
    - Moral
4. Generating image prompt.

Uses:
    qwen/qwen3.8-27b
"""

import json

from langchain_groq import ChatGroq

from langchain_core.messages import HumanMessage

from config import VISION_MODEL

from utils.logging import get_logger

logger = get_logger(__name__)


class StoryVisionAgent:
    """
    Vision-powered story understanding agent.
    """

    def __init__(self):

        self.llm = ChatGroq(
            model=VISION_MODEL,
            temperature=0
        )

    def analyze_image(
        self,
        image_b64: str
    ) -> dict:
        """
        Analyze story image.

        Args:
            image_b64:
                Base64 encoded image.

        Returns:
            Structured JSON.
        """

        prompt = """
        Analyze this story image.

        Return ONLY valid JSON.

        {
          "story_name":"",
          "characters":[],
          "setting":"",
          "moral":"",
          "story_summary":"",
          "image_prompt":""
        }
        """

        response = self.llm.invoke(
            [
                HumanMessage(
                    content=[
                        {
                            "type":
                                "image_url",

                            "image_url": {
                                "url":
                                f"data:image/png;base64,{image_b64}"
                            }
                        },
                        {
                            "type":
                                "text",

                            "text":
                                prompt
                        }
                    ]
                )
            ]
        )

        logger.info(
            "Vision analysis completed."
        )

        return json.loads(
            response.content
        )
