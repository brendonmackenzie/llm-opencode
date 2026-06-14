import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from llm_opencode import OpenCodeGoAnthropicAsyncChat, OpenCodeGoAnthropicChat


def _filter_headers(headers_to_redact):
    def before_record_request(request):
        for key in request.headers:
            if key.lower() in headers_to_redact:
                request.headers[key] = "REDACTED"
        return request
    return before_record_request


@pytest.fixture(scope="session")
def vcr_config():
    return {
        "before_record_request": _filter_headers(["authorization", "x-api-key"]),
    }


@pytest.fixture
def anthropic_sync_model():
    return OpenCodeGoAnthropicChat(model_id="opencode-go/minimax-m3")


@pytest.fixture
def anthropic_async_model():
    return OpenCodeGoAnthropicAsyncChat(model_id="opencode-go/minimax-m3")


@pytest.fixture
def mocked_sync_anthropic_client():
    with patch("llm_opencode.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        yield mock_client


@pytest.fixture
def mocked_async_anthropic_client():
    with patch("llm_opencode.AsyncAnthropic") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        yield mock_client


@pytest.fixture
def anthropic_response():
    return MagicMock()


@pytest.fixture
def make_opencode_models():
    """Factory for creating mock opencode models lists for VCR tests.

    Returns a list of model definition dicts (each with an ``id`` key) so the
    VCR-marked tests don't trigger a real HTTP GET to the models list endpoint
    when the local cache file is missing.
    """
    def _make(*model_ids):
        return [{"id": model_id} for model_id in model_ids]
    return _make
