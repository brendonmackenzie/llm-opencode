from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import llm

from tests._fakes import make_message
from tests._payload import api_model, entry, snapshot


@patch("openai.OpenAI")
@patch("llm.get_key", return_value="sk-test")
@patch("llm_opencode.fetch_models_snapshot")
def test_openai_protocol_prompt(mock_fetch, mock_get_key, mock_openai_cls, user_dir):
    mock_fetch.return_value = snapshot(
        entry(api_model(model_id="deepseek-v4-flash", name="DeepSeek V4 Flash"))
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
@patch("llm_opencode.fetch_models_snapshot")
def test_anthropic_protocol_prompt(mock_fetch, mock_get_key, mock_anthropic_cls, user_dir):
    from tests._payload import anthropic_api_model

    mock_fetch.return_value = snapshot(entry(anthropic_api_model()))
    mock_anthropic_cls.return_value.messages.create.return_value = make_message("Hello")
    conversation = llm.get_model("opencode-go/claude-haiku-5-5").conversation()
    response = conversation.prompt(
        "Say hello in one word", stream=False
    )

    assert str(response) == "Hello"
    kwargs = mock_anthropic_cls.return_value.messages.create.call_args[1]
    assert kwargs["model"] == "claude-haiku-5-5"
    assert kwargs["messages"] == [{"role": "user", "content": "Say hello in one word"}]
    assert kwargs["max_tokens"] == 4096
    default_headers = mock_anthropic_cls.call_args[1]["default_headers"]
    assert (
        default_headers["x-opencode-session"] == conversation.id
    )


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
