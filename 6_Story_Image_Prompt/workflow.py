"""
workflow.py

Core LangGraph workflow.
"""

import json

from langgraph.graph import (StateGraph, END)

from langchain_groq import (ChatGroq)

from langchain_core.messages import (HumanMessage)

from models import (StoryState)

from config import (
    VISION_MODEL,
    MODEL_TEMPERATURE,
    VISION_SYSTEM_PROMPT
)

from tools import (
    evaluate_story_coverage,
    evaluate_hallucination,
    quality_gate,
    save_prompt
)

llm = ChatGroq(
    model=VISION_MODEL,
    temperature=MODEL_TEMPERATURE
)


def vision_node(
    state: StoryState
):
    """
    Analyze image using Qwen.
    """

    response = llm.invoke(
        [
            HumanMessage(
                content=[
                    {
                        "type":"image_url",
                        "image_url":
                        {
                            "url":
                            (
                                "data:image/png;base64,"
                                f"{state['image_b64']}"
                            )
                        }
                    },
                    {
                        "type":"text",
                        "text":VISION_SYSTEM_PROMPT
                    }
                ]
            )
        ]
    )

    data = json.loads(
        response.content
    )

    state.update(data)

    state["token_usage"] = (
        response.response_metadata
        .get(
            "token_usage",
            {}
        )
    )

    return state


def evaluation_node(
    state: StoryState
):

    coverage = (
        evaluate_story_coverage(
            state["image_prompt"],
            state["characters"],
            state["setting"],
            state["moral"]
        )
    )

    state.update(
        coverage
    )

    state[
        "hallucination_score"
    ] = (
        evaluate_hallucination(
            state["image_prompt"],
            state["story_summary"]
        )
    )

    # Temporary score

    state[
        "prompt_quality"
    ] = 9.5

    state[
        "semantic_similarity"
    ] = 1.0

    return state


def quality_gate_node(
    state: StoryState
):

    state["approved"] = (
        quality_gate(
            state[
                "semantic_similarity"
            ],

            state[
                "hallucination_score"
            ],

            state[
                "prompt_quality"
            ],

            state[
                "characters_present"
            ],

            state[
                "setting_present"
            ],

            state[
                "moral_present"
            ]
        )
    )

    return state


def save_node(
    state: StoryState
):

    save_prompt(
        state["story_name"],
        state["image_prompt"]
    )

    return state


def router(
    state: StoryState
):

    if state["approved"]:
        return "save"

    return END


def build_graph():

    graph = StateGraph(
        StoryState
    )

    graph.add_node(
        "vision",
        vision_node
    )

    graph.add_node(
        "evaluation",
        evaluation_node
    )

    graph.add_node(
        "quality",
        quality_gate_node
    )

    graph.add_node(
        "save",
        save_node
    )

    graph.set_entry_point(
        "vision"
    )

    graph.add_edge(
        "vision",
        "evaluation"
    )

    graph.add_edge(
        "evaluation",
        "quality"
    )

    graph.add_conditional_edges(
        "quality",
        router
    )

    graph.add_edge(
        "save",
        END
    )

    return graph.compile()