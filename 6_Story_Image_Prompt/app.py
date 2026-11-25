"""
app.py

Entry point.
"""

from workflow import (
    build_graph
)

from models import (
    StoryState
)

#from PIL import Image
import base64


def load_image_as_base64(
    image_path: str
):

    with open(
        image_path,
        "rb"
    ) as file:

        return (
            base64
            .b64encode(
                file.read()
            )
            .decode("utf-8")
        )


workflow = build_graph()

image_b64 = load_image_as_base64(
    "sample_story.png"
)

state: StoryState = {
    "image_b64": image_b64
}

result = workflow.invoke(
    state
)

print(
    result["story_name"]
)

print(
    result["image_prompt"]
)

print(
    result["approved"]
)
