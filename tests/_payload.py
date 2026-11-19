"""Models-API payload factories grounded in the live 2026-11-19 response
from https://opencode.ai/inference/go/v1/models (trimmed, not invented)."""

from llm_opencode import api_model_to_entry

CHAT_BASE_URL = "https://opencode.ai/inference/go/openai/v1"
ANTHROPIC_BASE_URL = "https://opencode.ai/inference/go/anthropic/v1"
GOOGLE_BASE_URL = "https://opencode.ai/inference/go/google/v1beta"


def api_model(model_id="glm-5.3", **overrides):
    model = {
        "id": model_id,
        "object": "model",
        "created": 1795050567,
        "owned_by": "opencode",
        "name": "GLM-5.3",
        "canonical_id": "zhipuai/glm-5.3",
        "type": "chat",
        "description": (
            "Flagship GLM model for long-horizon coding, agents, and "
            "complex project delivery"
        ),
        "family": "glm",
        "open_weights": True,
        "knowledge": "",
        "release_date": "2026-08-14",
        "last_updated": "2026-08-14",
        "modalities": {"input": ["text"], "output": ["text"]},
        "capabilities": {
            "tools": {"supported": True},
            "reasoning": {"supported": True, "effort": ["low", "high", "max"]},
            "structured_output": True,
            "temperature": True,
            "top_p": True,
            "top_k": True,
            "system_updates": True,
            "effort_updates": False,
        },
        "limit": {"context": 1000000, "output": 131072},
        "cost": {
            "text": {
                "input": {"price": 1.4, "cache": {"read": 0.26}},
                "output": {"price": 4.4},
            },
        },
        "api": {
            "chat-completions": {
                "base_url": CHAT_BASE_URL,
                "sdk": {
                    "aisdk": "@ai-sdk/openai-compatible",
                    "opencodeai": "@opencode/ai/providers/openai-compatible",
                },
                "compatibility": {"reasoning_field": "reasoning_content"},
                "capabilities": {"tools": {"choice": ["auto"], "updates": []}},
            },
        },
        "recommended_api": "chat-completions",
        "go": {
            "monthly_usage": {"go": 500, "go-plus": 1000, "go-teams": 505},
            "training": False,
            "zdr": True,
            "limited_regions": False,
            "featured": True,
        },
    }
    model.update(overrides)
    return model


def anthropic_api_model(model_id="claude-haiku-5-5", **overrides):
    model = api_model(
        model_id=model_id,
        name="Claude Haiku 5.5",
        canonical_id="anthropic/claude-haiku-5-5",
        family="claude-haiku",
        open_weights=False,
        knowledge="2026-06",
        release_date="2026-10-07",
        last_updated="2026-10-07",
        description="Fast Claude model for responsive assistance",
        modalities={"input": ["text", "image/png"], "output": ["text"]},
        capabilities={
            "tools": {"supported": True},
            "reasoning": {
                "supported": True,
                "toggle": True,
                "effort": ["low", "medium", "high", "xhigh", "max"],
            },
            "structured_output": True,
            "temperature": False,
            "top_p": False,
            "top_k": False,
            "system_updates": True,
            "effort_updates": True,
        },
        limit={"context": 1000000, "output": 128000},
        cost={
            "text": {
                "input": {
                    "price": 0.1,
                    "cache": {"read": 0.01, "write": 0.125},
                    "overrides": [
                        {
                            "when": {"context": {"gt": 100000}},
                            "price": 0.5,
                            "cache": {"read": 0.05, "write": 0.625},
                        }
                    ],
                },
                "output": {
                    "price": 0.5,
                    "overrides": [
                        {"when": {"context": {"gt": 100000}}, "price": 2.5}
                    ],
                },
            },
        },
        api={
            "messages": {
                "base_url": ANTHROPIC_BASE_URL,
                "sdk": {
                    "aisdk": "@ai-sdk/anthropic",
                    "opencodeai": "@opencode/ai/providers/anthropic",
                },
                "capabilities": {
                    "tools": {
                        "strict": True,
                        "choice": ["auto", "none", "required", "tool"],
                        "search": True,
                        "updates": ["add", "remove", "replace"],
                    }
                },
            }
        },
        recommended_api="messages",
        go={
            "monthly_usage": {"go": 15, "go-plus": 60, "go-teams": 20},
            "training": False,
            "zdr": False,
            "limited_regions": False,
            "featured": True,
        },
    )
    model.update(overrides)
    return model


def responses_api_model(model_id="gpt-6-luna", **overrides):
    model = api_model(
        model_id=model_id,
        name="GPT-6 Luna",
        canonical_id="openai/gpt-6-luna",
        family="gpt-luna",
        open_weights=False,
        knowledge="2026-05-18",
        release_date="2026-09-22",
        last_updated="2026-09-22",
        description="OpenAI's most efficient model for focused, high-volume tasks",
        modalities={"input": ["text", "image/png"], "output": ["text"]},
        capabilities={
            "tools": {"supported": True},
            "reasoning": {
                "supported": True,
                "effort": ["none", "low", "medium", "high", "xhigh", "max"],
            },
            "structured_output": True,
            "temperature": True,
            "top_p": True,
            "top_k": False,
            "system_updates": True,
            "effort_updates": True,
        },
        limit={"context": 1050000, "input": 922000, "output": 128000},
        cost={
            "text": {
                "input": {
                    "price": 0.1,
                    "cache": {"read": 0.01, "write": {"30m": 0.125, "12h": 0.25}},
                    "overrides": [
                        {
                            "when": {"context": {"gt": 272000}},
                            "price": 0.2,
                            "cache": {
                                "read": 0.02,
                                "write": {"30m": 0.25, "12h": 0.5},
                            },
                        }
                    ],
                },
                "output": {
                    "price": 0.5,
                    "overrides": [
                        {"when": {"context": {"gt": 272000}}, "price": 0.75}
                    ],
                },
            },
        },
        api={
            "responses": {
                "base_url": CHAT_BASE_URL,
                "sdk": {
                    "aisdk": "@ai-sdk/openai",
                    "opencodeai": "@opencode/ai/providers/openai/responses",
                },
                "capabilities": {
                    "tools": {
                        "strict": True,
                        "choice": ["auto", "none", "required", "tool"],
                        "search": True,
                        "freeform": ["text", "lark", "regex"],
                        "updates": ["add", "remove", "replace"],
                    }
                },
            }
        },
        recommended_api="responses",
        go={
            "monthly_usage": {"go": 15, "go-plus": 60, "go-teams": 20},
            "training": False,
            "zdr": False,
            "limited_regions": False,
            "featured": True,
        },
    )
    model.update(overrides)
    return model


def google_api_model(model_id="gemini-4-argon", **overrides):
    model = api_model(
        model_id=model_id,
        name="Gemini 4 Argon",
        canonical_id="google/gemini-4-argon",
        family="gemini",
        open_weights=False,
        knowledge="2026-03",
        release_date="2026-10-13",
        last_updated="2026-10-13",
        description="Google's flagship reasoning model",
        modalities={"input": ["text", "image/png"], "output": ["text"]},
        capabilities={
            "tools": {"supported": True},
            "reasoning": {"supported": True, "effort": ["low", "medium", "high"]},
            "cache": {"implicit": True},
            "structured_output": True,
            "temperature": False,
            "top_p": False,
            "top_k": False,
            "system_updates": False,
            "effort_updates": False,
        },
        limit={"context": 1048576, "output": 65536},
        cost={
            "text": {
                "input": {"price": 2, "cache": {"read": 0.1}},
                "output": {"price": 10},
                "audio": {"input": {"price": 2}},
                "multipliers": [{"when": {"date": {"gte": "2026-12-01"}}, "factor": 2}],
            },
        },
        api={
            "generate-content": {
                "base_url": GOOGLE_BASE_URL,
                "sdk": {
                    "aisdk": "@ai-sdk/google",
                    "opencodeai": "@opencode/ai/providers/google",
                },
                "capabilities": {
                    "tools": {"choice": ["auto", "none", "required", "tool"], "updates": []}
                },
            }
        },
        recommended_api="generate-content",
        go={
            "monthly_usage": {"go": 30, "go-plus": 120, "go-teams": 35},
            "training": False,
            "zdr": False,
            "limited_regions": False,
            "featured": True,
        },
    )
    model.update(overrides)
    return model


def free_api_model(model_id="beam-free", **overrides):
    model = api_model(
        model_id=model_id,
        name="Beam Free",
        canonical_id="reflection/beam-1.0-501b-a23b",
        family="beam",
        knowledge="2026-06-30",
        release_date="2026-10-05",
        last_updated="2026-10-16",
        description="Sparse mixture-of-experts reasoning model",
        cost={
            "text": {
                "input": {"price": 0, "cache": {"read": 0}},
                "output": {"price": 0},
            },
        },
        go={
            "monthly_usage": None,
            "training": False,
            "zdr": False,
            "limited_regions": True,
            "featured": True,
        },
    )
    model.update(overrides)
    return model


def entry(payload=None, **overrides):
    """A unified snapshot entry, derived through the real transform.

    Tests never hand-write the unified shape: it is always
    :func:`llm_opencode.api_model_to_entry` applied to a payload factory,
    with per-test field overrides applied after.
    """
    unified = api_model_to_entry(payload if payload is not None else api_model())
    unified.update(overrides)
    return unified


def snapshot(*entries, fetched_at=1, warnings=None):
    return {"fetched_at": fetched_at, "warnings": warnings or [], "models": list(entries)}
