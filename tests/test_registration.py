from unittest.mock import MagicMock, patch

import llm

from llm_opencode import (
    DocsScrapeError,
    OpenCodeGoAnthropicAsyncChat,
    OpenCodeGoAnthropicChat,
    OpenCodeGoAsyncChat,
    OpenCodeGoChat,
    OpenCodeGoResponsesAsyncChat,
    OpenCodeGoResponsesChat,
    register_models,
)

CHAT_ENDPOINT = "https://example-alt.test/inference/go/v1/chat/completions"
RESPONSES_ENDPOINT = "https://example-alt.test/inference/go/v1/responses"
ANTHROPIC_ENDPOINT = "https://example-alt.test/inference/go/v1/messages"


def _snapshot(*entries):
    return {"scraped_at": 1, "warnings": [], "models": list(entries)}


def _entry(model_id, endpoint, package=""):
    return {
        "model_id": model_id,
        "model_name": "",
        "matched": bool(endpoint),
        "endpoint": endpoint,
        "ai_sdk_package": package,
    }


def test_model_registration():
    models = llm.get_models()
    model_ids = [m.model_id for m in models]
    opencode_models = [m for m in model_ids if m.startswith("opencode-go/")]
    assert len(opencode_models) > 0


@patch("llm.get_key", return_value=None)
def test_register_models_no_key(mock_get_key):
    register = MagicMock()
    register_models(register)
    register.assert_not_called()


@patch("llm_opencode.fetch_unified_models")
@patch("llm.get_key", return_value="sk-test")
def test_register_models_snapshot_failure_registers_nothing(
    mock_get_key, mock_fetch, user_dir
):
    mock_fetch.side_effect = DocsScrapeError("boom")

    register = MagicMock()
    register_models(register)

    register.assert_not_called()


@patch("llm_opencode.fetch_unified_models")
@patch("llm.get_key", return_value="sk-test")
def test_register_models_empty_snapshot(mock_get_key, mock_fetch, user_dir):
    mock_fetch.return_value = _snapshot()

    register = MagicMock()
    register_models(register)

    register.assert_not_called()


@patch("llm_opencode.fetch_unified_models")
@patch("llm.get_key", return_value="sk-test")
def test_register_models_routes_by_endpoint(mock_get_key, mock_fetch, user_dir):
    mock_fetch.return_value = _snapshot(
        _entry("glm-5", CHAT_ENDPOINT, "@ai-sdk/openai-compatible"),
        _entry("minimax-m3", ANTHROPIC_ENDPOINT, "@ai-sdk/anthropic"),
        _entry("gpt-6-luna", RESPONSES_ENDPOINT, "@ai-sdk/openai"),
    )

    register = MagicMock()
    register_models(register)

    assert register.call_count == 3

    chat_args = register.call_args_list[0][0]
    assert isinstance(chat_args[0], OpenCodeGoChat)
    assert isinstance(chat_args[1], OpenCodeGoAsyncChat)
    assert chat_args[0].model_id == "opencode-go/glm-5"
    assert chat_args[0].api_base == "https://example-alt.test/inference/go/v1"

    anthropic_args = register.call_args_list[1][0]
    assert isinstance(anthropic_args[0], OpenCodeGoAnthropicChat)
    assert isinstance(anthropic_args[1], OpenCodeGoAnthropicAsyncChat)
    assert anthropic_args[0].model_id == "opencode-go/minimax-m3"
    assert anthropic_args[0].api_base == "https://example-alt.test/inference/go"

    responses_args = register.call_args_list[2][0]
    assert isinstance(responses_args[0], OpenCodeGoResponsesChat)
    assert isinstance(responses_args[1], OpenCodeGoResponsesAsyncChat)
    assert responses_args[0].model_id == "opencode-go/gpt-6-luna"
    assert responses_args[0].api_base == "https://example-alt.test/inference/go/v1"


@patch("llm_opencode.fetch_unified_models")
@patch("llm.get_key", return_value="sk-test")
def test_register_models_skips_no_docs_row_and_google_style(
    mock_get_key, mock_fetch, user_dir
):
    mock_fetch.return_value = _snapshot(
        _entry("omen-alpha", "", ""),
        _entry(
            "gemini-4-argon-contributor",
            "https://example-alt.test/inference/go/v1/models/gemini-4-argon-contributor",
            "@ai-sdk/google",
        ),
        _entry("glm-5", CHAT_ENDPOINT, "@ai-sdk/openai-compatible"),
    )

    register = MagicMock()
    register_models(register)

    assert register.call_count == 1
    assert register.call_args[0][0].model_id == "opencode-go/glm-5"


@patch("llm_opencode.fetch_unified_models")
@patch("llm.get_key", return_value="sk-test")
def test_register_models_package_mismatch_warns_but_registers(
    mock_get_key, mock_fetch, user_dir
):
    mock_fetch.return_value = _snapshot(
        _entry("glm-5", CHAT_ENDPOINT, "@ai-sdk/google")
    )

    register = MagicMock()
    register_models(register)

    assert register.call_count == 1
    assert register.call_args[0][0].model_id == "opencode-go/glm-5"


@patch("llm_opencode.fetch_unified_models")
@patch("llm.get_key", return_value="sk-test")
def test_register_models_variation_key_registers_inherited_route(
    mock_get_key, mock_fetch, user_dir
):
    mock_fetch.return_value = _snapshot(
        _entry("grok-4.7#le-200k-tokens", RESPONSES_ENDPOINT, "@ai-sdk/openai")
    )

    register = MagicMock()
    register_models(register)

    assert register.call_count == 1
    assert register.call_args[0][0].model_id == "opencode-go/grok-4.7#le-200k-tokens"


@patch("llm_opencode.fetch_unified_models")
@patch("llm.get_key", return_value="sk-test")
def test_register_models_novel_model_ids_need_no_hardcoded_list(
    mock_get_key, mock_fetch, user_dir
):
    mock_fetch.return_value = _snapshot(
        _entry("brand-new-model-99", CHAT_ENDPOINT, "")
    )

    register = MagicMock()
    register_models(register)

    assert register.call_count == 1
    assert register.call_args[0][0].model_id == "opencode-go/brand-new-model-99"


def test_openai_chat_str():
    model = OpenCodeGoChat(
        model_id="opencode-go/test", model_name="test", api_base="https://example.com/v1"
    )
    assert str(model) == "OpenCode Go: opencode-go/test"


def test_openai_async_chat_str():
    model = OpenCodeGoAsyncChat(
        model_id="opencode-go/test", model_name="test", api_base="https://example.com/v1"
    )
    assert str(model) == "OpenCode Go: opencode-go/test"
