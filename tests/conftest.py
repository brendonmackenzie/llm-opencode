import pytest
from unittest.mock import AsyncMock, MagicMock


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


@pytest.fixture
def make_message():
    def _make(text="Hello", input_tokens=10, output_tokens=5, content=None):
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = text

        usage = MagicMock()
        usage.input_tokens = input_tokens
        usage.output_tokens = output_tokens

        message = MagicMock()
        message.content = content if content is not None else [text_block]
        message.usage = usage
        message.model_dump.return_value = {"id": "msg_123"}
        return message
    return _make


@pytest.fixture
def make_prompt():
    def _make(prompt_text="Say hello", system=None, max_tokens=None, temperature=None):
        prompt = MagicMock()
        prompt.prompt = prompt_text
        prompt.system = system
        prompt.options.max_tokens = max_tokens
        prompt.options.temperature = temperature
        return prompt
    return _make


@pytest.fixture
def make_sync_stream():
    def _make(text_chunks=("\n\nHello", " world"), input_tokens=10, output_tokens=5):
        final_usage = MagicMock()
        final_usage.input_tokens = input_tokens
        final_usage.output_tokens = output_tokens

        final_message = MagicMock()
        final_message.usage = final_usage
        final_message.model_dump.return_value = {"id": "msg_123"}

        stream_obj = MagicMock()
        stream_obj.text_stream = text_chunks
        stream_obj.get_final_message.return_value = final_message

        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=stream_obj)
        ctx.__exit__ = MagicMock(return_value=False)
        return ctx, stream_obj, final_message
    return _make


@pytest.fixture
def make_async_stream():
    def _make(text_chunks=("\n\nHello", " world"), input_tokens=10, output_tokens=5):
        final_usage = MagicMock()
        final_usage.input_tokens = input_tokens
        final_usage.output_tokens = output_tokens

        final_message = MagicMock()
        final_message.usage = final_usage
        final_message.model_dump.return_value = {"id": "msg_123"}

        stream_obj = MagicMock()
        stream_obj.text_stream = AsyncIteratorWrapper(text_chunks)
        stream_obj.get_final_message = AsyncMock(return_value=final_message)

        class AsyncStreamContext:
            async def __aenter__(self):
                return stream_obj

            async def __aexit__(self, *args):
                pass

        ctx = AsyncStreamContext()
        return ctx, stream_obj, final_message
    return _make


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
