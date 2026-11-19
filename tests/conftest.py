import pathlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_opencode import OpenCodeGoAnthropicAsyncChat, OpenCodeGoAnthropicChat


@pytest.fixture
def user_dir(tmp_path, monkeypatch):
    """Isolate the plugin's user dir so tests never read the machine's
    real snapshot files."""
    monkeypatch.setattr(
        "llm_opencode.llm.user_dir", lambda: pathlib.Path(tmp_path)
    )
    return tmp_path


@pytest.fixture
def anthropic_sync_model():
    return OpenCodeGoAnthropicChat(model_id="opencode-go/minimax-m3")


@pytest.fixture
def anthropic_async_model():
    return OpenCodeGoAnthropicAsyncChat(model_id="opencode-go/minimax-m3")


@pytest.fixture
def mocked_sync_anthropic_cls():
    with patch("llm_opencode.Anthropic") as mock_cls:
        yield mock_cls


@pytest.fixture
def mocked_sync_anthropic_client(mocked_sync_anthropic_cls):
    mock_client = MagicMock()
    mocked_sync_anthropic_cls.return_value = mock_client
    yield mock_client


@pytest.fixture
def mocked_async_anthropic_cls():
    with patch("llm_opencode.AsyncAnthropic") as mock_cls:
        yield mock_cls


@pytest.fixture
def mocked_async_anthropic_client(mocked_async_anthropic_cls):
    mock_client = AsyncMock()
    mocked_async_anthropic_cls.return_value = mock_client
    yield mock_client


@pytest.fixture
def anthropic_response():
    return MagicMock()


@pytest.fixture
def make_opencode_models():
    """Factory for thin mock opencode models lists.

    Returns a list of model definition dicts (each with an ``id`` key) so
    bootstrap tests don't hit the real models endpoint when registration
    runs with no snapshot on disk.
    """
    def _make(*model_ids):
        return [{"id": model_id} for model_id in model_ids]
    return _make
