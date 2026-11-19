from unittest.mock import MagicMock, patch

import llm

from llm_opencode import (
    DownloadError,
    OpenCodeGoAnthropicAsyncChat,
    OpenCodeGoAnthropicChat,
    OpenCodeGoAsyncChat,
    OpenCodeGoChat,
    OpenCodeGoResponsesAsyncChat,
    OpenCodeGoResponsesChat,
    register_models,
)
from tests._payload import (
    anthropic_api_model,
    api_model,
    entry,
    responses_api_model,
    snapshot,
)


@patch("llm.get_key", return_value=None)
def test_register_models_no_key(mock_get_key):
    register = MagicMock()
    register_models(register)
    register.assert_not_called()


@patch("llm_opencode.fetch_models_snapshot")
@patch("llm.get_key", return_value="sk-test")
def test_register_models_snapshot_failure_registers_nothing(
    mock_get_key, mock_fetch, user_dir
):
    mock_fetch.side_effect = DownloadError("no net")

    register = MagicMock()
    register_models(register)

    register.assert_not_called()


@patch("llm_opencode.fetch_models_snapshot")
@patch("llm.get_key", return_value="sk-test")
def test_register_models_empty_snapshot(mock_get_key, mock_fetch, user_dir):
    mock_fetch.return_value = snapshot()

    register = MagicMock()
    register_models(register)

    register.assert_not_called()


@patch("llm_opencode.fetch_models_snapshot")
@patch("llm.get_key", return_value="sk-test")
def test_register_models_routes_by_recommended_api(mock_get_key, mock_fetch, user_dir):
    mock_fetch.return_value = snapshot(
        entry(api_model()),
        entry(anthropic_api_model()),
        entry(responses_api_model()),
    )

    register = MagicMock()
    register_models(register)

    assert register.call_count == 3

    chat_args = register.call_args_list[0][0]
    assert isinstance(chat_args[0], OpenCodeGoChat)
    assert isinstance(chat_args[1], OpenCodeGoAsyncChat)
    assert chat_args[0].model_id == "opencode-go/glm-5.3"
    assert chat_args[0].model_name == "glm-5.3"  # the wire name
    assert chat_args[0].api_base == "https://opencode.ai/inference/go/openai/v1"

    anthropic_args = register.call_args_list[1][0]
    assert isinstance(anthropic_args[0], OpenCodeGoAnthropicChat)
    assert isinstance(anthropic_args[1], OpenCodeGoAnthropicAsyncChat)
    assert anthropic_args[0].model_id == "opencode-go/claude-haiku-5-5"
    assert anthropic_args[0].anthropic_model_id == "claude-haiku-5-5"
    assert anthropic_args[0].api_base == "https://opencode.ai/inference/go/anthropic"

    responses_args = register.call_args_list[2][0]
    assert isinstance(responses_args[0], OpenCodeGoResponsesChat)
    assert isinstance(responses_args[1], OpenCodeGoResponsesAsyncChat)
    assert responses_args[0].model_id == "opencode-go/gpt-6-luna"
    assert responses_args[0].model_name == "gpt-6-luna"
    assert responses_args[0].api_base == "https://opencode.ai/inference/go/openai/v1"


@patch("llm_opencode.fetch_models_snapshot")
@patch("llm.get_key", return_value="sk-test")
def test_register_models_schema_support_follows_capabilities(
    mock_get_key, mock_fetch, user_dir
):
    mock_fetch.return_value = snapshot(
        entry(api_model()),  # structured_output: true
        entry(api_model(model_id="omen-alpha"), capabilities={"structured_output": False}),
        entry(responses_api_model()),
        entry(anthropic_api_model()),  # capabilities say true, wire family says no
    )

    register = MagicMock()
    register_models(register)

    chat = register.call_args_list[0][0]
    assert chat[0].supports_schema is True
    assert chat[1].supports_schema is True

    omen = register.call_args_list[1][0]
    assert omen[0].supports_schema is False
    assert omen[1].supports_schema is False

    responses = register.call_args_list[2][0]
    assert responses[0].supports_schema is True
    assert responses[1].supports_schema is True

    # The Anthropic custom class has no schema translation to the wire:
    # the flag stays honestly False even when the API advertises support.
    anthropic = register.call_args_list[3][0]
    assert anthropic[0].supports_schema is False
    assert anthropic[1].supports_schema is False


@patch("llm_opencode.fetch_models_snapshot")
@patch("llm.get_key", return_value="sk-test")
def test_register_models_skips_unknown_and_google_style(
    mock_get_key, mock_fetch, user_dir
):
    from tests._payload import google_api_model

    mock_fetch.return_value = snapshot(
        entry(api_model(model_id="omen-alpha", api={}, recommended_api="")),
        entry(google_api_model()),
        entry(api_model()),
    )

    register = MagicMock()
    register_models(register)

    assert register.call_count == 1
    assert register.call_args[0][0].model_id == "opencode-go/glm-5.3"


@patch("llm_opencode.fetch_models_snapshot")
@patch("llm.get_key", return_value="sk-test")
def test_register_models_novel_model_ids_need_no_hardcoded_list(
    mock_get_key, mock_fetch, user_dir
):
    mock_fetch.return_value = snapshot(entry(api_model(model_id="brand-new-model-99")))

    register = MagicMock()
    register_models(register)

    assert register.call_count == 1
    assert register.call_args[0][0].model_id == "opencode-go/brand-new-model-99"


@patch("llm.get_key", return_value="sk-test")
@patch("llm_opencode.read_snapshot")
def test_register_models_via_llm_hooks(mock_read, mock_get_key):
    mock_read.return_value = snapshot(
        entry(api_model()),
        entry(api_model(model_id="unroutable", api={}, recommended_api="")),
    )

    models = llm.get_models()
    opencode_models = [m.model_id for m in models if m.model_id.startswith("opencode-go/")]
    assert "opencode-go/glm-5.3" in opencode_models
    assert not any("unroutable" in model_id for model_id in opencode_models)


def test_openai_chat_str():
    model = OpenCodeGoChat(
        model_id="opencode-go/test", model_name="test", api_base="https://example.com/v1"
    )
    assert str(model) == "OpenCode Go: opencode-go/test"


def test_openai_responses_chat_str():
    model = OpenCodeGoResponsesChat(
        model_id="opencode-go/test", model_name="test", api_base="https://example.com/v1"
    )
    assert str(model) == "OpenCode Go: opencode-go/test"


def test_openai_responses_async_chat_str():
    model = OpenCodeGoResponsesAsyncChat(
        model_id="opencode-go/test", model_name="test", api_base="https://example.com/v1"
    )
    assert str(model) == "OpenCode Go: opencode-go/test"


def test_openai_async_chat_str():
    model = OpenCodeGoAsyncChat(
        model_id="opencode-go/test", model_name="test", api_base="https://example.com/v1"
    )
    assert str(model) == "OpenCode Go: opencode-go/test"
