from unittest.mock import MagicMock, patch

import llm

from llm_opencode import (
    OpenCodeGoAnthropicAsyncChat,
    OpenCodeGoAnthropicChat,
    OpenCodeGoAsyncChat,
    OpenCodeGoChat,
    register_models,
)


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
