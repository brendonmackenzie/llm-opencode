import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import llm
import pytest
from click.testing import CliRunner
from llm.cli import cli

from llm_opencode import (
    DownloadError,
    OpenCodeGoAnthropicAsyncChat,
    OpenCodeGoAnthropicChat,
    OpenCodeGoAsyncChat,
    OpenCodeGoChat,
    fetch_cached_json,
    register_models,
    _get_protocol,
)


def _get_key():
    return os.environ.get("OPENCODE_KEY", "sk-test")


class AsyncIteratorWrapper:
    def __init__(self, items):
        self._items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration


@pytest.mark.vcr
def test_openai_protocol_prompt():
    model = llm.get_model("opencode-go/deepseek-v4-flash")
    response = model.prompt("Say hello in one word", key=_get_key())
    text = str(response)
    assert isinstance(text, str)
    assert len(text) > 0


@pytest.mark.vcr
def test_anthropic_protocol_prompt():
    model = llm.get_model("opencode-go/minimax-m3")
    response = model.prompt("Say hello in one word", key=_get_key())
    text = str(response)
    assert isinstance(text, str)
    assert len(text) > 0


@pytest.mark.vcr
def test_llm_models():
    runner = CliRunner()
    result = runner.invoke(cli, ["models", "list"])
    assert result.exit_code == 0, result.output


def test_model_registration():
    models = llm.get_models()
    model_ids = [m.model_id for m in models]
    opencode_models = [m for m in model_ids if m.startswith("opencode-go/")]
    assert len(opencode_models) > 0


def test_get_protocol():
    assert _get_protocol("glm-5") == "openai"
    assert _get_protocol("glm-5.1") == "openai"
    assert _get_protocol("deepseek-v4-flash") == "openai"
    assert _get_protocol("kimi-k2.5") == "openai"
    assert _get_protocol("minimax-m3") == "anthropic"
    assert _get_protocol("minimax-m2.7") == "anthropic"
    assert _get_protocol("qwen3.7-max") == "anthropic"
    assert _get_protocol("unknown-model") == "openai"


def test_openai_async_chat_str():
    model = OpenCodeGoAsyncChat(
        model_id="opencode-go/test", model_name="test", api_base="https://example.com/v1"
    )
    assert str(model) == "OpenCode Go: opencode-go/test"


def test_anthropic_chat_str():
    model = OpenCodeGoAnthropicChat(model_id="opencode-go/minimax-m3")
    assert str(model) == "OpenCode Go: opencode-go/minimax-m3"


def test_anthropic_async_chat_str():
    model = OpenCodeGoAnthropicAsyncChat(model_id="opencode-go/minimax-m3")
    assert str(model) == "OpenCode Go: opencode-go/minimax-m3"


def test_anthropic_chat_custom_model_id():
    model = OpenCodeGoAnthropicChat(
        model_id="opencode-go/minimax-m3", anthropic_model_id="custom-model"
    )
    assert model.anthropic_model_id == "custom-model"


def test_anthropic_async_chat_custom_model_id():
    model = OpenCodeGoAnthropicAsyncChat(
        model_id="opencode-go/minimax-m3", anthropic_model_id="custom-model"
    )
    assert model.anthropic_model_id == "custom-model"


def test_anthropic_chat_default_model_id():
    model = OpenCodeGoAnthropicChat(model_id="opencode-go/minimax-m3")
    assert model.anthropic_model_id == "minimax-m3"


def test_anthropic_async_chat_default_model_id():
    model = OpenCodeGoAnthropicAsyncChat(model_id="opencode-go/minimax-m3")
    assert model.anthropic_model_id == "minimax-m3"


def test_build_messages_with_conversation():
    model = OpenCodeGoAnthropicChat(model_id="opencode-go/minimax-m3")

    mock_prev_response = MagicMock()
    mock_prev_response.prompt.prompt = "Previous question"
    mock_prev_response.text_or_raise.return_value = "Previous answer"

    mock_conversation = MagicMock()
    mock_conversation.responses = [mock_prev_response]

    mock_prompt = MagicMock()
    mock_prompt.prompt = "Current question"

    messages = model._build_messages(mock_prompt, mock_conversation)
    assert messages == [
        {"role": "user", "content": "Previous question"},
        {"role": "assistant", "content": "Previous answer"},
        {"role": "user", "content": "Current question"},
    ]


def test_build_messages_with_empty_user_content():
    model = OpenCodeGoAnthropicChat(model_id="opencode-go/minimax-m3")

    mock_prev_response = MagicMock()
    mock_prev_response.prompt.prompt = ""
    mock_prev_response.text_or_raise.return_value = "Answer"

    mock_conversation = MagicMock()
    mock_conversation.responses = [mock_prev_response]

    mock_prompt = MagicMock()
    mock_prompt.prompt = "New question"

    messages = model._build_messages(mock_prompt, mock_conversation)
    assert messages == [
        {"role": "assistant", "content": "Answer"},
        {"role": "user", "content": "New question"},
    ]


def test_async_build_messages_with_conversation():
    model = OpenCodeGoAnthropicAsyncChat(model_id="opencode-go/minimax-m3")

    mock_prev_response = MagicMock()
    mock_prev_response.prompt.prompt = "Previous question"
    mock_prev_response.text_or_raise.return_value = "Previous answer"

    mock_conversation = MagicMock()
    mock_conversation.responses = [mock_prev_response]

    mock_prompt = MagicMock()
    mock_prompt.prompt = "Current question"

    messages = model._build_messages(mock_prompt, mock_conversation)
    assert messages == [
        {"role": "user", "content": "Previous question"},
        {"role": "assistant", "content": "Previous answer"},
        {"role": "user", "content": "Current question"},
    ]


def test_async_build_messages_with_empty_user_content():
    model = OpenCodeGoAnthropicAsyncChat(model_id="opencode-go/minimax-m3")

    mock_prev_response = MagicMock()
    mock_prev_response.prompt.prompt = ""
    mock_prev_response.text_or_raise.return_value = "Answer"

    mock_conversation = MagicMock()
    mock_conversation.responses = [mock_prev_response]

    mock_prompt = MagicMock()
    mock_prompt.prompt = "New question"

    messages = model._build_messages(mock_prompt, mock_conversation)
    assert messages == [
        {"role": "assistant", "content": "Answer"},
        {"role": "user", "content": "New question"},
    ]


def test_build_messages_no_conversation():
    model = OpenCodeGoAnthropicChat(model_id="opencode-go/minimax-m3")

    mock_prompt = MagicMock()
    mock_prompt.prompt = "Hello"

    messages = model._build_messages(mock_prompt, None)
    assert messages == [{"role": "user", "content": "Hello"}]


@patch("llm_opencode.Anthropic")
def test_anthropic_prompt_no_stream(mock_anthropic_cls):
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client

    mock_text_block = MagicMock()
    mock_text_block.type = "text"
    mock_text_block.text = "Hello"

    mock_usage = MagicMock()
    mock_usage.input_tokens = 10
    mock_usage.output_tokens = 5

    mock_message = MagicMock()
    mock_message.content = [mock_text_block]
    mock_message.usage = mock_usage
    mock_message.model_dump.return_value = {"id": "msg_123"}

    mock_client.messages.create.return_value = mock_message

    model = OpenCodeGoAnthropicChat(model_id="opencode-go/minimax-m3")
    mock_prompt = MagicMock()
    mock_prompt.prompt = "Say hello"
    mock_prompt.system = None
    mock_prompt.options.max_tokens = None
    mock_prompt.options.temperature = None

    mock_response = MagicMock()

    chunks = list(
        model.execute(
            mock_prompt,
            stream=False,
            response=mock_response,
            conversation=None,
            key="sk-test",
        )
    )

    assert chunks == ["Hello"]
    mock_response.set_usage.assert_called_once_with(input=10, output=5)
    mock_client.messages.create.assert_called_once()


@patch("llm_opencode.Anthropic")
def test_anthropic_prompt_with_system(mock_anthropic_cls):
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client

    mock_text_block = MagicMock()
    mock_text_block.type = "text"
    mock_text_block.text = "Hello"

    mock_usage = MagicMock()
    mock_usage.input_tokens = 10
    mock_usage.output_tokens = 5

    mock_message = MagicMock()
    mock_message.content = [mock_text_block]
    mock_message.usage = mock_usage
    mock_message.model_dump.return_value = {"id": "msg_123"}

    mock_client.messages.create.return_value = mock_message

    model = OpenCodeGoAnthropicChat(model_id="opencode-go/minimax-m3")
    mock_prompt = MagicMock()
    mock_prompt.prompt = "Say hello"
    mock_prompt.system = "You are a helpful assistant"
    mock_prompt.options.max_tokens = None
    mock_prompt.options.temperature = None

    mock_response = MagicMock()

    list(
        model.execute(
            mock_prompt, stream=False, response=mock_response, conversation=None, key="sk-test"
        )
    )

    call_kwargs = mock_client.messages.create.call_args[1]
    assert "system" in call_kwargs
    assert call_kwargs["system"] == "You are a helpful assistant"


@patch("llm_opencode.Anthropic")
def test_anthropic_prompt_with_temperature(mock_anthropic_cls):
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client

    mock_text_block = MagicMock()
    mock_text_block.type = "text"
    mock_text_block.text = "Hello"

    mock_usage = MagicMock()
    mock_usage.input_tokens = 10
    mock_usage.output_tokens = 5

    mock_message = MagicMock()
    mock_message.content = [mock_text_block]
    mock_message.usage = mock_usage
    mock_message.model_dump.return_value = {"id": "msg_123"}

    mock_client.messages.create.return_value = mock_message

    model = OpenCodeGoAnthropicChat(model_id="opencode-go/minimax-m3")
    mock_prompt = MagicMock()
    mock_prompt.prompt = "Say hello"
    mock_prompt.system = None
    mock_prompt.options.max_tokens = 2048
    mock_prompt.options.temperature = 0.5

    mock_response = MagicMock()

    list(
        model.execute(
            mock_prompt, stream=False, response=mock_response, conversation=None, key="sk-test"
        )
    )

    call_kwargs = mock_client.messages.create.call_args[1]
    assert call_kwargs["max_tokens"] == 2048
    assert call_kwargs["temperature"] == 0.5


@patch("llm_opencode.Anthropic")
def test_anthropic_prompt_stream(mock_anthropic_cls):
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client

    mock_final_usage = MagicMock()
    mock_final_usage.input_tokens = 10
    mock_final_usage.output_tokens = 5
    mock_final_message = MagicMock()
    mock_final_message.usage = mock_final_usage
    mock_final_message.model_dump.return_value = {"id": "msg_123"}

    mock_stream_obj = MagicMock()
    mock_stream_obj.text_stream = ["\n\nHello", " world"]
    mock_stream_obj.get_final_message.return_value = mock_final_message

    mock_client.messages.stream.return_value.__enter__ = MagicMock(return_value=mock_stream_obj)
    mock_client.messages.stream.return_value.__exit__ = MagicMock(return_value=False)

    model = OpenCodeGoAnthropicChat(model_id="opencode-go/minimax-m3")
    mock_prompt = MagicMock()
    mock_prompt.prompt = "Say hello"
    mock_prompt.system = None
    mock_prompt.options.max_tokens = None
    mock_prompt.options.temperature = None

    mock_response = MagicMock()

    chunks = list(
        model.execute(
            mock_prompt, stream=True, response=mock_response, conversation=None, key="sk-test"
        )
    )

    assert chunks == ["Hello", " world"]
    mock_response.set_usage.assert_called_once_with(input=10, output=5)


@patch("llm_opencode.AsyncAnthropic")
async def test_anthropic_async_prompt_no_stream(mock_async_anthropic_cls):
    mock_client = AsyncMock()
    mock_async_anthropic_cls.return_value = mock_client

    mock_text_block = MagicMock()
    mock_text_block.type = "text"
    mock_text_block.text = "Hello"

    mock_usage = MagicMock()
    mock_usage.input_tokens = 10
    mock_usage.output_tokens = 5

    mock_message = MagicMock()
    mock_message.content = [mock_text_block]
    mock_message.usage = mock_usage
    mock_message.model_dump.return_value = {"id": "msg_123"}

    mock_client.messages.create.return_value = mock_message

    model = OpenCodeGoAnthropicAsyncChat(model_id="opencode-go/minimax-m3")
    mock_prompt = MagicMock()
    mock_prompt.prompt = "Say hello"
    mock_prompt.system = None
    mock_prompt.options.max_tokens = None
    mock_prompt.options.temperature = None

    mock_response = MagicMock()

    chunks = []
    async for chunk in model.execute(
        mock_prompt, stream=False, response=mock_response, conversation=None, key="sk-test"
    ):
        chunks.append(chunk)

    assert chunks == ["Hello"]
    mock_response.set_usage.assert_called_once_with(input=10, output=5)


@patch("llm_opencode.AsyncAnthropic")
async def test_anthropic_async_prompt_stream(mock_async_anthropic_cls):
    mock_client = AsyncMock()
    mock_async_anthropic_cls.return_value = mock_client

    mock_final_usage = MagicMock()
    mock_final_usage.input_tokens = 10
    mock_final_usage.output_tokens = 5
    mock_final_message = MagicMock()
    mock_final_message.usage = mock_final_usage
    mock_final_message.model_dump.return_value = {"id": "msg_123"}

    mock_stream_obj = MagicMock()
    mock_stream_obj.text_stream = AsyncIteratorWrapper(["\n\nHello", " world"])
    mock_stream_obj.get_final_message = AsyncMock(return_value=mock_final_message)

    class AsyncStreamContext:
        async def __aenter__(self):
            return mock_stream_obj

        async def __aexit__(self, *args):
            pass

    mock_client.messages.stream = MagicMock(return_value=AsyncStreamContext())

    model = OpenCodeGoAnthropicAsyncChat(model_id="opencode-go/minimax-m3")
    mock_prompt = MagicMock()
    mock_prompt.prompt = "Say hello"
    mock_prompt.system = None
    mock_prompt.options.max_tokens = None
    mock_prompt.options.temperature = None

    mock_response = MagicMock()

    chunks = []
    async for chunk in model.execute(
        mock_prompt, stream=True, response=mock_response, conversation=None, key="sk-test"
    ):
        chunks.append(chunk)

    assert chunks == ["Hello", " world"]
    mock_response.set_usage.assert_called_once_with(input=10, output=5)


@patch("llm_opencode.AsyncAnthropic")
async def test_anthropic_async_prompt_with_system(mock_async_anthropic_cls):
    mock_client = AsyncMock()
    mock_async_anthropic_cls.return_value = mock_client

    mock_text_block = MagicMock()
    mock_text_block.type = "text"
    mock_text_block.text = "Hello"

    mock_usage = MagicMock()
    mock_usage.input_tokens = 10
    mock_usage.output_tokens = 5

    mock_message = MagicMock()
    mock_message.content = [mock_text_block]
    mock_message.usage = mock_usage
    mock_message.model_dump.return_value = {"id": "msg_123"}

    mock_client.messages.create.return_value = mock_message

    model = OpenCodeGoAnthropicAsyncChat(model_id="opencode-go/minimax-m3")
    mock_prompt = MagicMock()
    mock_prompt.prompt = "Say hello"
    mock_prompt.system = "You are a helpful assistant"
    mock_prompt.options.max_tokens = None
    mock_prompt.options.temperature = None

    mock_response = MagicMock()

    async for _ in model.execute(
        mock_prompt, stream=False, response=mock_response, conversation=None, key="sk-test"
    ):
        pass

    call_kwargs = mock_client.messages.create.call_args[1]
    assert "system" in call_kwargs
    assert call_kwargs["system"] == "You are a helpful assistant"


@patch("llm_opencode.AsyncAnthropic")
async def test_anthropic_async_prompt_with_temperature(mock_async_anthropic_cls):
    mock_client = AsyncMock()
    mock_async_anthropic_cls.return_value = mock_client

    mock_text_block = MagicMock()
    mock_text_block.type = "text"
    mock_text_block.text = "Hello"

    mock_usage = MagicMock()
    mock_usage.input_tokens = 10
    mock_usage.output_tokens = 5

    mock_message = MagicMock()
    mock_message.content = [mock_text_block]
    mock_message.usage = mock_usage
    mock_message.model_dump.return_value = {"id": "msg_123"}

    mock_client.messages.create.return_value = mock_message

    model = OpenCodeGoAnthropicAsyncChat(model_id="opencode-go/minimax-m3")
    mock_prompt = MagicMock()
    mock_prompt.prompt = "Say hello"
    mock_prompt.system = None
    mock_prompt.options.max_tokens = 2048
    mock_prompt.options.temperature = 0.5

    mock_response = MagicMock()

    async for _ in model.execute(
        mock_prompt, stream=False, response=mock_response, conversation=None, key="sk-test"
    ):
        pass

    call_kwargs = mock_client.messages.create.call_args[1]
    assert call_kwargs["max_tokens"] == 2048
    assert call_kwargs["temperature"] == 0.5


@patch("llm.get_key", return_value=None)
def test_register_models_no_key(mock_get_key):
    register = MagicMock()
    register_models(register)
    register.assert_not_called()


def test_fetch_cached_json_cache_hit(tmp_path):
    cache_file = tmp_path / "cache.json"
    cache_data = {"data": [{"id": "model-1"}]}
    cache_file.write_text(json.dumps(cache_data))

    result = fetch_cached_json("https://example.com/api", cache_file, 3600)
    assert result == cache_data


@patch("httpx.get")
def test_fetch_cached_json_network_fetch(mock_httpx_get, tmp_path):
    mock_response = MagicMock()
    mock_response.json.return_value = {"data": [{"id": "model-1"}]}
    mock_response.raise_for_status.return_value = None
    mock_httpx_get.return_value = mock_response

    cache_file = tmp_path / "cache.json"

    result = fetch_cached_json("https://example.com/api", cache_file, 3600)
    assert result == {"data": [{"id": "model-1"}]}
    assert cache_file.exists()


@patch("httpx.get")
def test_fetch_cached_json_http_error_with_cache(mock_httpx_get, tmp_path):
    cache_file = tmp_path / "cache.json"
    cache_data = {"data": [{"id": "model-1"}]}
    cache_file.write_text(json.dumps(cache_data))
    os.utime(cache_file, (0, 0))

    mock_httpx_get.side_effect = httpx.HTTPError("Connection error")

    result = fetch_cached_json("https://example.com/api", cache_file, 3600)
    assert result == cache_data


@patch("httpx.get")
def test_fetch_cached_json_http_error_no_cache(mock_httpx_get, tmp_path):
    cache_file = tmp_path / "cache.json"

    mock_httpx_get.side_effect = httpx.HTTPError("Connection error")

    with pytest.raises(DownloadError):
        fetch_cached_json("https://example.com/api", cache_file, 3600)


@patch("llm_opencode.get_opencode_models")
def test_opencode_models_cli(mock_get_models):
    mock_get_models.return_value = [
        {"id": "glm-5"},
        {"id": "minimax-m3"},
    ]

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "models"])
    assert result.exit_code == 0, result.output
    assert "glm-5" in result.output
    assert "minimax-m3" in result.output
    assert "openai" in result.output
    assert "anthropic" in result.output


@patch("llm_opencode.get_opencode_models")
def test_opencode_models_cli_json(mock_get_models):
    mock_get_models.return_value = [
        {"id": "glm-5"},
        {"id": "minimax-m3"},
    ]

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "models", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data[0]["id"] == "glm-5"
