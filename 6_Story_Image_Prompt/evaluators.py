"""
evaluators.py

LangSmith evaluators.
"""

from sentence_transformers import (
    SentenceTransformer
)

from sklearn.metrics.pairwise import (
    cosine_similarity
)

import numpy as np

_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def semantic_match(run, example):
    """
    Semantic similarity evaluator.
    """

    expected = example.outputs["answer"]

    actual = run.outputs["answer"]

    vectors = _model.encode(
        [
            expected,
            actual
        ]
    )

    score = cosine_similarity(
        vectors[0].reshape(1, -1),
        vectors[1].reshape(1, -1)
    )[0][0]

    return {
        "key":
            "semantic_similarity",

        "score":
            float(score)
    }


def token_usage(
    run,
    example
):
    """
    Log token usage.
    """

    usage = run.outputs.get(
        "token_usage",
        {}
    )

    return {
        "key":
            "total_tokens",

        "score":
            usage.get(
                "total_tokens",
                0
            )
    }


def latency(
    run,
    example
):
    """
    Runtime latency.
    """

    duration = (
        run.end_time
        -
        run.start_time
    )

    return {
        "key":
            "latency",

        "score":
            duration.total_seconds()
    }