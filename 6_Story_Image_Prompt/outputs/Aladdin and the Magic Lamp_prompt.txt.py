"""
Image helper utilities.

Currently supports:

- Loading image as base64
- Converting image path to base64

Future extensions:

- Image compression
- Streamlit uploads
- OCR preprocessing
"""

import base64

from pathlib import Path


def image_to_base64(
    image_path: str
) -> str:
    """
    Convert local image into
    base64 string.

    Args:
        image_path:
            Local image path.

    Returns:
        Base64 encoded image.
    """

    image_path = Path(
        image_path
    )

    with open(
        image_path,
        "rb"
    ) as image_file:

        return base64.b64encode(
            image_file.read()
        ).decode("utf-8")


def build_base64_url(
    image_b64: str
) -> str:
    """
    Convert base64 image into
    model-compatible URL structure.

    Args:
        image_b64:
            Base64 encoded image.

    Returns:
        Data URL.
    """

    return (
        f"data:image/png;base64,"
        f"{image_b64}"
    )