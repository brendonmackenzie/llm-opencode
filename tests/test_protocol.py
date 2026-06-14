import os
from unittest.mock import patch

import llm
import pytest

from llm_opencode import _get_protocol


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


def test_get_protocol():
    assert _get_protocol("glm-5") == "openai"
    assert _get_protocol("glm-5.1") == "openai"
    assert _get_protocol("deepseek-v4-flash") == "openai"
    assert _get_protocol("kimi-k2.5") == "openai"
    assert _get_protocol("minimax-m3") == "anthropic"
    assert _get_protocol("minimax-m2.7") == "anthropic"
    assert _get_protocol("qwen3.7-max") == "anthropic"
    assert _get_protocol("unknown-model") == "openai"
