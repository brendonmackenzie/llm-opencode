from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import llm

from tests._fakes import make_message


def _snapshot(*entries):
    return {"scraped_at": 1, "warnings": [], "models": list(entries)}


def _fake_chat_completion(content="Hello"):
    completion = MagicMock()
    completion.model_dump.return_value = {"id": "chatcmpl-test"}
    completion.usage.model_dump.return_value = {
        "prompt_tokens": 4,
        "completion_tokens": 2,
        "total_tokens": 6,
    }
    completion.choices = [
        SimpleNamespace(
            message=SimpleNamespace(content=content, tool_calls=None),
        )
    ]
    return completion


@patch("openai.OpenAI")
@patch("llm.get_key", return_value="sk-test")
@patch("llm_opencode.fetch_unified_models")
def test_openai_protocol_prompt(mock_fetch, mock_get_key, mock_openai_cls):
    mock_fetch.return_value = _snapshot(
        {
            "model_id": "deepseek-v4-flash",
            "matched": True,
            "endpoint": "https://opencode.ai/inference/go/v1/chat/completions",
            "ai_sdk_package": "@ai-sdk/openai-compatible",
        }
    )
    chat_completions = mock_openai_cls.return_value.chat.completions
    chat_completions.create.return_value = _fake_chat_completion("Hello")
    conversation = llm.get_model("opencode-go/deepseek-v4-flash").conversation()
    response = conversation.prompt(
        "Say hello in one word", stream=False
    )

    assert str(response) == "Hello"
    kwargs = mock_openai_cls.return_value.chat.completions.create.call_args[1]
    assert kwargs["model"] == "deepseek-v4-flash"
    assert kwargs["messages"] == [
        {"role": "user", "content": "Say hello in one word"}
    ]
    assert (
        kwargs["extra_headers"]["x-opencode-session"] == conversation.id
    )


@patch("llm_opencode.Anthropic")
@patch("llm.get_key", return_value="sk-test")
@patch("llm_opencode.fetch_unified_models")
def test_anthropic_protocol_prompt(mock_fetch, mock_get_key, mock_anthropic_cls):
    mock_fetch.return_value = _snapshot(
        {
            "model_id": "minimax-m3",
            "matched": True,
            "endpoint": "https://opencode.ai/inference/go/v1/messages",
            "ai_sdk_package": "@ai-sdk/anthropic",
        }
    )
    mock_anthropic_cls.return_value.messages.create.return_value = make_message("Hello")
    conversation = llm.get_model("opencode-go/minimax-m3").conversation()
    response = conversation.prompt(
        "Say hello in one word", stream=False
    )

    assert str(response) == "Hello"
    kwargs = mock_anthropic_cls.return_value.messages.create.call_args[1]
    assert kwargs["model"] == "minimax-m3"
    assert kwargs["messages"] == [{"role": "user", "content": "Say hello in one word"}]
    assert kwargs["max_tokens"] == 4096
    default_headers = mock_anthropic_cls.call_args[1]["default_headers"]
    assert (
        default_headers["x-opencode-session"] == conversation.id
    )
