import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import llm
import pytest

from llm_opencode import (
    OpenCodeGoAsyncChat,
    OpenCodeGoChat,
    OpenCodeGoResponsesAsyncChat,
    OpenCodeGoResponsesChat,
)
from tests._fakes import Conversation, _Options, make_responses_completion


def _make_model(async_=False):
    cls = OpenCodeGoAsyncChat if async_ else OpenCodeGoChat
    return cls(
        model_id="opencode-go/deepseek-v4-flash",
        model_name="deepseek-v4-flash",
        api_base="https://opencode.ai/inference/go/openai/v1",
    )


def _make_responses_model(async_=False):
    cls = OpenCodeGoResponsesAsyncChat if async_ else OpenCodeGoResponsesChat
    return cls(
        model_id="opencode-go/gpt-6-luna",
        model_name="gpt-6-luna",
        api_base="https://opencode.ai/inference/go/openai/v1",
    )


def _make_prompt(model):
    return llm.Prompt(
        "Say hello",
        model=model,
        options=_Options(max_tokens=None, temperature=None),
    )


def _texts(chunks):
    return [chunk.chunk if hasattr(chunk, "chunk") else chunk for chunk in chunks]


def _completion(content="Hello"):
    completion = MagicMock()
    completion.usage.model_dump.return_value = {
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "total_tokens": 2,
    }
    completion.model_dump.return_value = {"id": "chatcmpl-test"}
    completion.choices = [
        SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None))
    ]
    return completion


def test_openai_session_header_from_conversation():
    model = _make_model()
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _completion()
    conv = Conversation(id="my-session-uuid")
    with patch("openai.OpenAI", return_value=mock_client):
        chunks = list(
            model.execute(_make_prompt(model), False, MagicMock(), conv, "sk-test")
        )
    assert _texts(chunks) == ["Hello"]
    kwargs = mock_client.chat.completions.create.call_args[1]
    assert kwargs["extra_headers"] == {"x-opencode-session": "my-session-uuid"}


def test_openai_session_header_stream():
    model = _make_model()
    mock_client = MagicMock()
    chunk = SimpleNamespace(
        usage=None,
        id=None,
        object=None,
        model=None,
        created=None,
        index=None,
        choices=[
            SimpleNamespace(
                logprobs=None,
                finish_reason=None,
                delta=SimpleNamespace(role="assistant", content="Hi", tool_calls=[]),
            )
        ],
    )
    mock_client.chat.completions.create.return_value = iter([chunk])
    conv = Conversation(id="my-session-uuid")
    with patch("openai.OpenAI", return_value=mock_client):
        chunks = list(
            model.execute(_make_prompt(model), True, MagicMock(), conv, "sk-test")
        )
    assert _texts(chunks) == ["Hi"]
    kwargs = mock_client.chat.completions.create.call_args[1]
    assert kwargs["extra_headers"] == {"x-opencode-session": "my-session-uuid"}


def test_openai_missing_conversation_generates_uuid():
    model = _make_model()
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _completion()
    with patch("openai.OpenAI", return_value=mock_client):
        list(model.execute(_make_prompt(model), False, MagicMock(), None, "sk-test"))
    header = mock_client.chat.completions.create.call_args[1]["extra_headers"][
        "x-opencode-session"
    ]
    uuid.UUID(header)


@pytest.mark.asyncio
async def test_openai_session_header_from_conversation_async():
    model = _make_model(async_=True)
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=_completion())
    conv = Conversation(id="my-session-uuid")
    with patch("openai.AsyncOpenAI", return_value=mock_client):
        chunks = [
            chunk
            async for chunk in model.execute(
                _make_prompt(model), False, MagicMock(), conv, "sk-test"
            )
        ]
    assert _texts(chunks) == ["Hello"]
    kwargs = mock_client.chat.completions.create.call_args[1]
    assert kwargs["extra_headers"] == {"x-opencode-session": "my-session-uuid"}


@pytest.mark.asyncio
async def test_openai_missing_conversation_generates_uuid_async():
    model = _make_model(async_=True)
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=_completion())
    with patch("openai.AsyncOpenAI", return_value=mock_client):
        async for _ in model.execute(
            _make_prompt(model), False, MagicMock(), None, "sk-test"
        ):
            pass
    header = mock_client.chat.completions.create.call_args[1]["extra_headers"][
        "x-opencode-session"
    ]
    uuid.UUID(header)


@pytest.mark.asyncio
async def test_async_interleaved_conversations_keep_separate_sessions():
    model = _make_model(async_=True)
    mock_client = MagicMock()

    async def _create(**kwargs):
        return _completion(kwargs["extra_headers"]["x-opencode-session"])

    mock_client.chat.completions.create = AsyncMock(side_effect=_create)

    async def run(conv_id):
        conv = Conversation(id=conv_id)
        return [
            chunk
            async for chunk in model.execute(
                _make_prompt(model), False, MagicMock(), conv, "sk-test"
            )
        ]

    with patch("openai.AsyncOpenAI", return_value=mock_client):
        results = await asyncio.gather(run("session-aaa"), run("session-bbb"))

    assert _texts(results[0]) == ["session-aaa"]
    assert _texts(results[1]) == ["session-bbb"]


def test_responses_session_header_from_conversation():
    model = _make_responses_model()
    mock_client = MagicMock()
    mock_client.responses.create.return_value = make_responses_completion()
    captured_client_kwargs = {}

    def fake_openai(**kwargs):
        captured_client_kwargs.update(kwargs)
        return mock_client

    conv = Conversation(id="my-session-uuid")
    with patch("openai.OpenAI", new=fake_openai):
        chunks = list(
            model.execute(_make_prompt(model), False, MagicMock(), conv, "sk-test")
        )
    assert captured_client_kwargs["default_headers"] == {
        "x-opencode-session": "my-session-uuid"
    }


async def test_responses_session_header_from_conversation_async():
    model = _make_responses_model(async_=True)
    captured_client_kwargs = {}

    def fake_async_openai(**kwargs):
        captured_client_kwargs.update(kwargs)
        client = MagicMock()
        client.responses.create = AsyncMock(return_value=make_responses_completion())
        return client

    conv = Conversation(id="my-session-uuid")
    with patch("openai.AsyncOpenAI", new=fake_async_openai):
        chunks = [
            chunk
            async for chunk in model.execute(
                _make_prompt(model), False, MagicMock(), conv, "sk-test"
            )
        ]
    assert chunks == []
    assert captured_client_kwargs["default_headers"] == {
        "x-opencode-session": "my-session-uuid"
    }


def test_responses_missing_conversation_generates_uuid():
    model = _make_responses_model()
    captured_client_kwargs = {}

    def fake_openai(**kwargs):
        captured_client_kwargs.update(kwargs)
        client = MagicMock()
        client.responses.create.return_value = make_responses_completion()
        return client

    with patch("openai.OpenAI", new=fake_openai):
        list(model.execute(_make_prompt(model), False, MagicMock(), None, "sk-test"))
    header = captured_client_kwargs["default_headers"]["x-opencode-session"]
    uuid.UUID(header)


async def test_responses_missing_conversation_generates_uuid_async():
    model = _make_responses_model(async_=True)
    captured_client_kwargs = {}

    def fake_async_openai(**kwargs):
        captured_client_kwargs.update(kwargs)
        client = MagicMock()
        client.responses.create = AsyncMock(
            return_value=make_responses_completion()
        )
        return client

    with patch("openai.AsyncOpenAI", new=fake_async_openai):
        chunks = [
            chunk
            async for chunk in model.execute(
                _make_prompt(model), False, MagicMock(), None, "sk-test"
            )
        ]
    assert chunks == []
    header = captured_client_kwargs["default_headers"]["x-opencode-session"]
    uuid.UUID(header)


async def test_async_responses_interleaved_conversations_keep_separate_sessions():
    model = _make_responses_model(async_=True)
    captured = []

    def fake_async_openai(**kwargs):
        session = (kwargs.get("default_headers") or {}).get("x-opencode-session")
        captured.append(session)
        client = MagicMock()
        client.responses.create = AsyncMock(
            return_value=make_responses_completion(session)
        )
        return client

    async def run(conv_id):
        conv = Conversation(id=conv_id)
        return [
            chunk
            async for chunk in model.execute(
                _make_prompt(model), False, MagicMock(), conv, "sk-test"
            )
        ]

    with patch("openai.AsyncOpenAI", new=fake_async_openai):
        await asyncio.gather(run("session-aaa"), run("session-bbb"))

    assert sorted(captured) == ["session-aaa", "session-bbb"]


def test_responses_get_client_without_session_uses_plain_client():
    model = _make_responses_model()
    with patch("openai.OpenAI") as mock_openai:
        model.get_client("sk-test")
    assert "default_headers" not in mock_openai.call_args[1]


async def test_responses_get_client_without_session_uses_plain_client_async():
    model = _make_responses_model(async_=True)
    with patch("openai.AsyncOpenAI") as mock_openai:
        model.get_client("sk-test", async_=True)
    assert "default_headers" not in mock_openai.call_args[1]


def test_openai_build_kwargs_without_session_header():
    model = _make_model()
    kwargs = model.build_kwargs(_make_prompt(model), False)
    assert "extra_headers" not in kwargs


async def test_openai_build_kwargs_without_session_header_async():
    model = _make_model(async_=True)
    kwargs = model.build_kwargs(_make_prompt(model), False)
    assert "extra_headers" not in kwargs
