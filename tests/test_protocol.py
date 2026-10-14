from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import llm

from llm_opencode import _get_protocol
from tests._fakes import make_message


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
@patch("llm_opencode.get_opencode_models")
def test_openai_protocol_prompt(mock_get_models, mock_get_key, mock_openai_cls):
    mock_get_models.return_value = [{"id": "deepseek-v4-flash"}]
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
@patch("llm_opencode.get_opencode_models")
def test_anthropic_protocol_prompt(mock_get_models, mock_get_key, mock_anthropic_cls):
    mock_get_models.return_value = [{"id": "minimax-m3"}]
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


def test_get_protocol():
    assert _get_protocol("glm-5") == "openai"
    assert _get_protocol("glm-5.1") == "openai"
    assert _get_protocol("deepseek-v4-flash") == "openai"
    assert _get_protocol("kimi-k2.5") == "openai"
    assert _get_protocol("minimax-m3") == "anthropic"
    assert _get_protocol("minimax-m2.7") == "anthropic"
    assert _get_protocol("qwen3.7-max") == "anthropic"
    assert _get_protocol("unknown-model") == "openai"


def test_get_protocol_responses():
    assert _get_protocol("gpt-6-luna") == "openai-responses"
    assert _get_protocol("gpt-5.6-luna") == "openai-responses"
    assert _get_protocol("grok-4.6") == "openai-responses"
    assert _get_protocol("grok-4.7") == "openai-responses"
    assert _get_protocol("muse-spark-1.2-contributor") == "openai-responses"
    assert _get_protocol("muse-spark-1.3-contributor") == "openai-responses"


def test_get_protocol_new_model_ids():
    assert _get_protocol("claude-haiku-5-5") == "anthropic"
    assert _get_protocol("qwen3.8-max") == "anthropic"
    assert _get_protocol("qwen3.8-flash") == "anthropic"
    assert _get_protocol("glm-5.3-flash") == "openai"
    assert _get_protocol("hy4-preview") == "openai"
