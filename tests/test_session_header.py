import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_opencode import OpenCodeGoAsyncChat, OpenCodeGoChat
from tests._fakes import Conversation, make_prompt


def _make_model(async_=False):
    cls = OpenCodeGoAsyncChat if async_ else OpenCodeGoChat
    return cls(
        model_id="opencode-go/deepseek-v4-flash",
        model_name="deepseek-v4-flash",
        api_base="https://opencode.ai/zen/go/v1",
    )


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
            model.execute(make_prompt(), False, MagicMock(), conv, "sk-test")
        )
    assert chunks == ["Hello"]
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
        chunks = list(model.execute(make_prompt(), True, MagicMock(), conv, "sk-test"))
    assert chunks == ["Hi"]
    kwargs = mock_client.chat.completions.create.call_args[1]
    assert kwargs["extra_headers"] == {"x-opencode-session": "my-session-uuid"}


def test_openai_missing_conversation_generates_uuid():
    model = _make_model()
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _completion()
    with patch("openai.OpenAI", return_value=mock_client):
        list(model.execute(make_prompt(), False, MagicMock(), None, "sk-test"))
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
                make_prompt(), False, MagicMock(), conv, "sk-test"
            )
        ]
    assert chunks == ["Hello"]
    kwargs = mock_client.chat.completions.create.call_args[1]
    assert kwargs["extra_headers"] == {"x-opencode-session": "my-session-uuid"}


@pytest.mark.asyncio
async def test_openai_missing_conversation_generates_uuid_async():
    model = _make_model(async_=True)
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=_completion())
    with patch("openai.AsyncOpenAI", return_value=mock_client):
        async for _ in model.execute(make_prompt(), False, MagicMock(), None, "sk-test"):
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
                make_prompt(), False, MagicMock(), conv, "sk-test"
            )
        ]

    with patch("openai.AsyncOpenAI", return_value=mock_client):
        results = await asyncio.gather(run("session-aaa"), run("session-bbb"))

    assert results[0] == ["session-aaa"]
    assert results[1] == ["session-bbb"]
