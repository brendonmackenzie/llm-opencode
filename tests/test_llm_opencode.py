import os

import llm
import pytest
from click.testing import CliRunner
from llm.cli import cli


def _get_key():
    return os.environ.get("OPENCODE_KEY", "sk-test")


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
    from llm_opencode import _get_protocol

    assert _get_protocol("glm-5") == "openai"
    assert _get_protocol("glm-5.1") == "openai"
    assert _get_protocol("deepseek-v4-flash") == "openai"
    assert _get_protocol("kimi-k2.5") == "openai"
    assert _get_protocol("minimax-m3") == "anthropic"
    assert _get_protocol("minimax-m2.7") == "anthropic"
    assert _get_protocol("qwen3.7-max") == "anthropic"
    assert _get_protocol("unknown-model") == "openai"