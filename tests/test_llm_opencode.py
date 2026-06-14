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

from tests._helpers import collect_chunks


def _get_key():
    return os.environ.get("OPENCODE_KEY", "sk-test")


@patch("llm_opencode.get_opencode_models")
@pytest.mark.vcr
def test_openai_protocol_prompt(mock_get_models, make_opencode_models):
    mock_get_models.return_value = make_opencode_models("deepseek-v4-flash")
    model = llm.get_model("opencode-go/deepseek-v4-flash")
    response = model.prompt("Say hello in one word", key=_get_key())
    text = str(response)
    assert isinstance(text, str)
    assert len(text) > 0


@patch("llm_opencode.get_opencode_models")
@pytest.mark.vcr
def test_anthropic_protocol_prompt(mock_get_models, make_opencode_models):
    mock_get_models.return_value = make_opencode_models("minimax-m3")
    model = llm.get_model("opencode-go/minimax-m3")
    response = model.prompt("Say hello in one word", key=_get_key())
    text = str(response)
    assert isinstance(text, str)
    assert len(text) > 0


@patch("llm_opencode.get_opencode_models")
@pytest.mark.vcr
def test_llm_models(mock_get_models, make_opencode_models):
    mock_get_models.return_value = make_opencode_models("deepseek-v4-flash", "minimax-m3")
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


def test_openai_chat_str():
    model = OpenCodeGoChat(
        model_id="opencode-go/test", model_name="test", api_base="https://example.com/v1"
    )
    assert str(model) == "OpenCode Go: opencode-go/test"


def test_anthropic_chat_str():
    model = OpenCodeGoAnthropicChat(model_id="opencode-go/minimax-m3")
    assert str(model) == "OpenCode Go: opencode-go/minimax-m3"


def test_anthropic_async_chat_str():
    model = OpenCodeGoAnthropicAsyncChat(model_id="opencode-go/minimax-m3")
    assert str(model) == "OpenCode Go: opencode-go/minimax-m3"


@pytest.mark.parametrize(
    "model_cls",
    [OpenCodeGoAnthropicChat, OpenCodeGoAnthropicAsyncChat],
    ids=["sync", "async"],
)
def test_anthropic_chat_custom_model_id(model_cls):
    model = model_cls(
        model_id="opencode-go/minimax-m3", anthropic_model_id="custom-model"
    )
    assert model.anthropic_model_id == "custom-model"


@pytest.mark.parametrize(
    "model_cls",
    [OpenCodeGoAnthropicChat, OpenCodeGoAnthropicAsyncChat],
    ids=["sync", "async"],
)
def test_anthropic_chat_default_model_id(model_cls):
    model = model_cls(model_id="opencode-go/minimax-m3")
    assert model.anthropic_model_id == "minimax-m3"


@pytest.mark.parametrize(
    "model_cls",
    [OpenCodeGoAnthropicChat, OpenCodeGoAnthropicAsyncChat],
    ids=["sync", "async"],
)
def test_build_messages_with_conversation(model_cls):
    model = model_cls(model_id="opencode-go/minimax-m3")

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


@pytest.mark.parametrize(
    "model_cls",
    [OpenCodeGoAnthropicChat, OpenCodeGoAnthropicAsyncChat],
    ids=["sync", "async"],
)
def test_build_messages_with_empty_user_content(model_cls):
    model = model_cls(model_id="opencode-go/minimax-m3")

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


@pytest.mark.parametrize(
    "model_cls",
    [OpenCodeGoAnthropicChat, OpenCodeGoAnthropicAsyncChat],
    ids=["sync", "async"],
)
def test_build_messages_multi_turn_conversation(model_cls):
    model = model_cls(model_id="opencode-go/minimax-m3")

    mock_prev1 = MagicMock()
    mock_prev1.prompt.prompt = "First question"
    mock_prev1.text_or_raise.return_value = "First answer"

    mock_prev2 = MagicMock()
    mock_prev2.prompt.prompt = "Second question"
    mock_prev2.text_or_raise.return_value = "Second answer"

    mock_conversation = MagicMock()
    mock_conversation.responses = [mock_prev1, mock_prev2]

    mock_prompt = MagicMock()
    mock_prompt.prompt = "Third question"

    messages = model._build_messages(mock_prompt, mock_conversation)
    assert messages == [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Second question"},
        {"role": "assistant", "content": "Second answer"},
        {"role": "user", "content": "Third question"},
    ]


@pytest.mark.parametrize(
    "model_cls",
    [OpenCodeGoAnthropicChat, OpenCodeGoAnthropicAsyncChat],
    ids=["sync", "async"],
)
def test_build_messages_with_none_prompt(model_cls):
    model = model_cls(model_id="opencode-go/minimax-m3")

    mock_prev_response = MagicMock()
    mock_prev_response.prompt.prompt = "Previous question"
    mock_prev_response.text_or_raise.return_value = "Previous answer"

    mock_conversation = MagicMock()
    mock_conversation.responses = [mock_prev_response]

    mock_prompt = MagicMock()
    mock_prompt.prompt = None

    messages = model._build_messages(mock_prompt, mock_conversation)
    assert messages == [
        {"role": "user", "content": "Previous question"},
        {"role": "assistant", "content": "Previous answer"},
    ]


def test_anthropic_prompt_no_stream(
    anthropic_chat_pair,
    make_message,
    make_prompt,
    anthropic_response,
):
    cfg = anthropic_chat_pair

    with patch(cfg["patch_path"]) as mock_cls:
        mock_client = AsyncMock() if cfg["is_async"] else MagicMock()
        mock_client.messages.create.return_value = make_message()
        mock_cls.return_value = mock_client

        chunks = collect_chunks(
            cfg["model"],
            make_prompt(),
            stream=False,
            response=anthropic_response,
            conversation=None,
            key="sk-test",
        )

    assert chunks == ["Hello"]
    anthropic_response.set_usage.assert_called_once_with(input=10, output=5)
    mock_client.messages.create.assert_called_once()


def test_anthropic_prompt_with_system(
    anthropic_chat_pair,
    make_message,
    make_prompt,
    anthropic_response,
):
    cfg = anthropic_chat_pair

    with patch(cfg["patch_path"]) as mock_cls:
        mock_client = AsyncMock() if cfg["is_async"] else MagicMock()
        mock_client.messages.create.return_value = make_message()
        mock_cls.return_value = mock_client

        collect_chunks(
            cfg["model"],
            make_prompt(system="You are a helpful assistant"),
            stream=False,
            response=anthropic_response,
            conversation=None,
            key="sk-test",
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "system" in call_kwargs
        assert call_kwargs["system"] == "You are a helpful assistant"


def test_anthropic_prompt_with_temperature(
    anthropic_chat_pair,
    make_message,
    make_prompt,
    anthropic_response,
):
    cfg = anthropic_chat_pair

    with patch(cfg["patch_path"]) as mock_cls:
        mock_client = AsyncMock() if cfg["is_async"] else MagicMock()
        mock_client.messages.create.return_value = make_message()
        mock_cls.return_value = mock_client

        collect_chunks(
            cfg["model"],
            make_prompt(max_tokens=2048, temperature=0.5),
            stream=False,
            response=anthropic_response,
            conversation=None,
            key="sk-test",
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["max_tokens"] == 2048
        assert call_kwargs["temperature"] == 0.5


def test_anthropic_prompt_stream(
    anthropic_chat_pair,
    make_stream,
    make_prompt,
    anthropic_response,
):
    cfg = anthropic_chat_pair

    with patch(cfg["patch_path"]) as mock_cls:
        ctx, _stream_obj, _final_message = make_stream()
        mock_client = AsyncMock() if cfg["is_async"] else MagicMock()
        if cfg["is_async"]:
            mock_client.messages.stream = MagicMock(return_value=ctx)
        else:
            mock_client.messages.stream.return_value = ctx
        mock_cls.return_value = mock_client

        chunks = collect_chunks(
            cfg["model"],
            make_prompt(),
            stream=True,
            response=anthropic_response,
            conversation=None,
            key="sk-test",
        )

    assert chunks == ["Hello", " world"]
    anthropic_response.set_usage.assert_called_once_with(input=10, output=5)


def test_anthropic_prompt_empty_content(
    anthropic_chat_pair,
    make_message,
    make_prompt,
    anthropic_response,
):
    cfg = anthropic_chat_pair

    with patch(cfg["patch_path"]) as mock_cls:
        mock_client = AsyncMock() if cfg["is_async"] else MagicMock()
        mock_client.messages.create.return_value = make_message(content=[])
        mock_cls.return_value = mock_client

        chunks = collect_chunks(
            cfg["model"],
            make_prompt(),
            stream=False,
            response=anthropic_response,
            conversation=None,
            key="sk-test",
        )

    assert chunks == []
    anthropic_response.set_usage.assert_called_once_with(input=10, output=5)


def test_anthropic_prompt_stream_only_whitespace(
    anthropic_chat_pair,
    make_stream,
    make_prompt,
    anthropic_response,
):
    cfg = anthropic_chat_pair

    with patch(cfg["patch_path"]) as mock_cls:
        ctx, _stream_obj, _final_message = make_stream(text_chunks=("\n\n", "   ", "\t"))
        mock_client = AsyncMock() if cfg["is_async"] else MagicMock()
        if cfg["is_async"]:
            mock_client.messages.stream = MagicMock(return_value=ctx)
        else:
            mock_client.messages.stream.return_value = ctx
        mock_cls.return_value = mock_client

        chunks = collect_chunks(
            cfg["model"],
            make_prompt(),
            stream=True,
            response=anthropic_response,
            conversation=None,
            key="sk-test",
        )

    assert chunks == []
    anthropic_response.set_usage.assert_called_once_with(input=10, output=5)


def test_anthropic_prompt_empty_user(
    anthropic_chat_pair,
    make_message,
    make_prompt,
    anthropic_response,
):
    cfg = anthropic_chat_pair

    with patch(cfg["patch_path"]) as mock_cls:
        mock_client = AsyncMock() if cfg["is_async"] else MagicMock()
        mock_client.messages.create.return_value = make_message()
        mock_cls.return_value = mock_client

        collect_chunks(
            cfg["model"],
            make_prompt(prompt_text=""),
            stream=False,
            response=anthropic_response,
            conversation=None,
            key="sk-test",
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["messages"] == []


@patch("llm.get_key", return_value=None)
def test_register_models_no_key(mock_get_key):
    register = MagicMock()
    register_models(register)
    register.assert_not_called()


@patch("llm_opencode.get_opencode_models")
@patch("llm.get_key", return_value="sk-test")
def test_register_models_with_valid_key(mock_get_key, mock_get_models):
    mock_get_models.return_value = [
        {"id": "glm-5"},
        {"id": "minimax-m3"},
    ]

    register = MagicMock()
    register_models(register)

    assert register.call_count == 2

    openai_args = register.call_args_list[0][0]
    assert isinstance(openai_args[0], OpenCodeGoChat)
    assert isinstance(openai_args[1], OpenCodeGoAsyncChat)
    assert openai_args[0].model_id == "opencode-go/glm-5"

    anthropic_args = register.call_args_list[1][0]
    assert isinstance(anthropic_args[0], OpenCodeGoAnthropicChat)
    assert isinstance(anthropic_args[1], OpenCodeGoAnthropicAsyncChat)
    assert anthropic_args[0].model_id == "opencode-go/minimax-m3"


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
def test_fetch_cached_json_stale_cache_network_success(mock_httpx_get, tmp_path):
    cache_file = tmp_path / "cache.json"
    old_data = {"data": [{"id": "old-model"}]}
    cache_file.write_text(json.dumps(old_data))
    os.utime(cache_file, (0, 0))

    new_data = {"data": [{"id": "new-model"}]}
    mock_response = MagicMock()
    mock_response.json.return_value = new_data
    mock_response.raise_for_status.return_value = None
    mock_httpx_get.return_value = mock_response

    result = fetch_cached_json("https://example.com/api", cache_file, 3600)
    assert result == new_data
    assert json.loads(cache_file.read_text()) == new_data


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
def test_fetch_cached_json_stale_cache_network_failure(mock_httpx_get, tmp_path):
    cache_file = tmp_path / "cache.json"
    stale_data = {"data": [{"id": "stale-model"}]}
    cache_file.write_text(json.dumps(stale_data))
    os.utime(cache_file, (0, 0))

    mock_httpx_get.side_effect = httpx.HTTPError("Connection error")

    result = fetch_cached_json("https://example.com/api", cache_file, 3600)
    assert result == stale_data


@patch("httpx.get")
def test_fetch_cached_json_http_error_no_cache(mock_httpx_get, tmp_path):
    cache_file = tmp_path / "cache.json"

    mock_httpx_get.side_effect = httpx.HTTPError("Connection error")

    with pytest.raises(DownloadError):
        fetch_cached_json("https://example.com/api", cache_file, 3600)


@patch("httpx.get")
def test_fetch_cached_json_invalid_json_cache(mock_httpx_get, tmp_path):
    cache_file = tmp_path / "cache.json"
    cache_file.write_text("not valid json")

    mock_httpx_get.side_effect = httpx.HTTPError("Connection error")

    with pytest.raises(DownloadError):
        fetch_cached_json("https://example.com/api", cache_file, 3600)


@patch("httpx.get")
def test_fetch_cached_json_fresh_cache_invalid_json_falls_back_to_network(
    mock_httpx_get, tmp_path
):
    cache_file = tmp_path / "cache.json"
    cache_file.write_text("not valid json")

    new_data = {"data": [{"id": "new-model"}]}
    mock_response = MagicMock()
    mock_response.json.return_value = new_data
    mock_response.raise_for_status.return_value = None
    mock_httpx_get.return_value = mock_response

    result = fetch_cached_json("https://example.com/api", cache_file, 3600)
    assert result == new_data
    assert json.loads(cache_file.read_text()) == new_data


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
