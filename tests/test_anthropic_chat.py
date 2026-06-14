from unittest.mock import MagicMock

import pytest
from anthropic import APIConnectionError, AuthenticationError

from llm_opencode import OpenCodeGoAnthropicAsyncChat, OpenCodeGoAnthropicChat
from tests._fakes import (
    Conversation,
    PrevResponse,
    Prompt,
    make_async_stream,
    make_message,
    make_prompt,
    make_sync_stream,
)


def test_anthropic_chat_str():
    model = OpenCodeGoAnthropicChat(model_id="opencode-go/minimax-m3")
    assert str(model) == "OpenCode Go: opencode-go/minimax-m3"


def test_anthropic_async_chat_str():
    model = OpenCodeGoAnthropicAsyncChat(model_id="opencode-go/minimax-m3")
    assert str(model) == "OpenCode Go: opencode-go/minimax-m3"


def test_anthropic_chat_custom_model_id():
    model = OpenCodeGoAnthropicChat(
        model_id="opencode-go/minimax-m3", anthropic_model_id="custom-model"
    )
    assert model.anthropic_model_id == "custom-model"


def test_anthropic_chat_custom_model_id_async():
    model = OpenCodeGoAnthropicAsyncChat(
        model_id="opencode-go/minimax-m3", anthropic_model_id="custom-model"
    )
    assert model.anthropic_model_id == "custom-model"


def test_anthropic_chat_default_model_id():
    model = OpenCodeGoAnthropicChat(model_id="opencode-go/minimax-m3")
    assert model.anthropic_model_id == "minimax-m3"


def test_anthropic_chat_default_model_id_async():
    model = OpenCodeGoAnthropicAsyncChat(model_id="opencode-go/minimax-m3")
    assert model.anthropic_model_id == "minimax-m3"


def test_build_messages_with_conversation(anthropic_sync_model):
    conv = Conversation(
        responses=[
            PrevResponse(prompt=Prompt(prompt="Previous question"), text="Previous answer")
        ]
    )
    messages = anthropic_sync_model._build_messages(
        Prompt(prompt="Current question"), conv
    )
    assert messages == [
        {"role": "user", "content": "Previous question"},
        {"role": "assistant", "content": "Previous answer"},
        {"role": "user", "content": "Current question"},
    ]


def test_build_messages_with_empty_user_content(anthropic_sync_model):
    conv = Conversation(
        responses=[PrevResponse(prompt=Prompt(prompt=""), text="Answer")]
    )
    messages = anthropic_sync_model._build_messages(
        Prompt(prompt="New question"), conv
    )
    assert messages == [
        {"role": "assistant", "content": "Answer"},
        {"role": "user", "content": "New question"},
    ]


def test_build_messages_no_conversation(anthropic_sync_model):
    messages = anthropic_sync_model._build_messages(Prompt(prompt="Hello"), None)
    assert messages == [{"role": "user", "content": "Hello"}]


def test_build_messages_multi_turn_conversation(anthropic_sync_model):
    conv = Conversation(
        responses=[
            PrevResponse(prompt=Prompt(prompt="First question"), text="First answer"),
            PrevResponse(prompt=Prompt(prompt="Second question"), text="Second answer"),
        ]
    )
    messages = anthropic_sync_model._build_messages(
        Prompt(prompt="Third question"), conv
    )
    assert messages == [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Second question"},
        {"role": "assistant", "content": "Second answer"},
        {"role": "user", "content": "Third question"},
    ]


def test_build_messages_with_none_prompt(anthropic_sync_model):
    conv = Conversation(
        responses=[
            PrevResponse(prompt=Prompt(prompt="Previous question"), text="Previous answer")
        ]
    )
    messages = anthropic_sync_model._build_messages(Prompt(prompt=None), conv)
    assert messages == [
        {"role": "user", "content": "Previous question"},
        {"role": "assistant", "content": "Previous answer"},
    ]


def test_anthropic_prompt_no_stream(
    anthropic_sync_model,
    mocked_sync_anthropic_client,
    anthropic_response,
):
    mocked_sync_anthropic_client.messages.create.return_value = make_message()
    chunks = list(
        anthropic_sync_model.execute(
            make_prompt(),
            stream=False,
            response=anthropic_response,
            conversation=None,
            key="sk-test",
        )
    )
    assert chunks == ["Hello"]
    anthropic_response.set_usage.assert_called_once_with(input=10, output=5)
    mocked_sync_anthropic_client.messages.create.assert_called_once()


@pytest.mark.asyncio
async def test_anthropic_prompt_no_stream_async(
    anthropic_async_model,
    mocked_async_anthropic_client,
    anthropic_response,
):
    mocked_async_anthropic_client.messages.create.return_value = make_message()
    chunks = []
    async for chunk in anthropic_async_model.execute(
        make_prompt(),
        stream=False,
        response=anthropic_response,
        conversation=None,
        key="sk-test",
    ):
        chunks.append(chunk)
    assert chunks == ["Hello"]
    anthropic_response.set_usage.assert_called_once_with(input=10, output=5)
    mocked_async_anthropic_client.messages.create.assert_awaited_once()


def test_anthropic_prompt_with_system(
    anthropic_sync_model,
    mocked_sync_anthropic_client,
    anthropic_response,
):
    mocked_sync_anthropic_client.messages.create.return_value = make_message()
    list(
        anthropic_sync_model.execute(
            make_prompt(system="You are a helpful assistant"),
            stream=False,
            response=anthropic_response,
            conversation=None,
            key="sk-test",
        )
    )
    call_kwargs = mocked_sync_anthropic_client.messages.create.call_args[1]
    assert call_kwargs["system"] == "You are a helpful assistant"


@pytest.mark.asyncio
async def test_anthropic_prompt_with_system_async(
    anthropic_async_model,
    mocked_async_anthropic_client,
    anthropic_response,
):
    mocked_async_anthropic_client.messages.create.return_value = make_message()
    async for _ in anthropic_async_model.execute(
        make_prompt(system="You are a helpful assistant"),
        stream=False,
        response=anthropic_response,
        conversation=None,
        key="sk-test",
    ):
        pass
    call_kwargs = mocked_async_anthropic_client.messages.create.call_args[1]
    assert call_kwargs["system"] == "You are a helpful assistant"


def test_anthropic_prompt_with_temperature(
    anthropic_sync_model,
    mocked_sync_anthropic_client,
    anthropic_response,
):
    mocked_sync_anthropic_client.messages.create.return_value = make_message()
    list(
        anthropic_sync_model.execute(
            make_prompt(max_tokens=2048, temperature=0.5),
            stream=False,
            response=anthropic_response,
            conversation=None,
            key="sk-test",
        )
    )
    call_kwargs = mocked_sync_anthropic_client.messages.create.call_args[1]
    assert call_kwargs["max_tokens"] == 2048
    assert call_kwargs["temperature"] == 0.5


@pytest.mark.asyncio
async def test_anthropic_prompt_with_temperature_async(
    anthropic_async_model,
    mocked_async_anthropic_client,
    anthropic_response,
):
    mocked_async_anthropic_client.messages.create.return_value = make_message()
    async for _ in anthropic_async_model.execute(
        make_prompt(max_tokens=2048, temperature=0.5),
        stream=False,
        response=anthropic_response,
        conversation=None,
        key="sk-test",
    ):
        pass
    call_kwargs = mocked_async_anthropic_client.messages.create.call_args[1]
    assert call_kwargs["max_tokens"] == 2048
    assert call_kwargs["temperature"] == 0.5


def test_anthropic_prompt_stream(
    anthropic_sync_model,
    mocked_sync_anthropic_client,
    anthropic_response,
):
    mocked_sync_anthropic_client.messages.stream.return_value = make_sync_stream()
    chunks = list(
        anthropic_sync_model.execute(
            make_prompt(),
            stream=True,
            response=anthropic_response,
            conversation=None,
            key="sk-test",
        )
    )
    assert chunks == ["Hello", " world"]
    anthropic_response.set_usage.assert_called_once_with(input=10, output=5)


@pytest.mark.asyncio
async def test_anthropic_prompt_stream_async(
    anthropic_async_model,
    mocked_async_anthropic_client,
    anthropic_response,
):
    mocked_async_anthropic_client.messages.stream = MagicMock(
        return_value=make_async_stream()
    )
    chunks = []
    async for chunk in anthropic_async_model.execute(
        make_prompt(),
        stream=True,
        response=anthropic_response,
        conversation=None,
        key="sk-test",
    ):
        chunks.append(chunk)
    assert chunks == ["Hello", " world"]
    anthropic_response.set_usage.assert_called_once_with(input=10, output=5)


def test_anthropic_prompt_empty_content(
    anthropic_sync_model,
    mocked_sync_anthropic_client,
    anthropic_response,
):
    mocked_sync_anthropic_client.messages.create.return_value = make_message(content=[])
    chunks = list(
        anthropic_sync_model.execute(
            make_prompt(),
            stream=False,
            response=anthropic_response,
            conversation=None,
            key="sk-test",
        )
    )
    assert chunks == []
    anthropic_response.set_usage.assert_called_once_with(input=10, output=5)


@pytest.mark.asyncio
async def test_anthropic_prompt_empty_content_async(
    anthropic_async_model,
    mocked_async_anthropic_client,
    anthropic_response,
):
    mocked_async_anthropic_client.messages.create.return_value = make_message(content=[])
    chunks = []
    async for chunk in anthropic_async_model.execute(
        make_prompt(),
        stream=False,
        response=anthropic_response,
        conversation=None,
        key="sk-test",
    ):
        chunks.append(chunk)
    assert chunks == []
    anthropic_response.set_usage.assert_called_once_with(input=10, output=5)


def test_anthropic_prompt_stream_only_whitespace(
    anthropic_sync_model,
    mocked_sync_anthropic_client,
    anthropic_response,
):
    mocked_sync_anthropic_client.messages.stream.return_value = make_sync_stream(
        text_chunks=("\n\n", "   ", "\t")
    )
    chunks = list(
        anthropic_sync_model.execute(
            make_prompt(),
            stream=True,
            response=anthropic_response,
            conversation=None,
            key="sk-test",
        )
    )
    assert chunks == []
    anthropic_response.set_usage.assert_called_once_with(input=10, output=5)


@pytest.mark.asyncio
async def test_anthropic_prompt_stream_only_whitespace_async(
    anthropic_async_model,
    mocked_async_anthropic_client,
    anthropic_response,
):
    mocked_async_anthropic_client.messages.stream = MagicMock(
        return_value=make_async_stream(text_chunks=("\n\n", "   ", "\t"))
    )
    chunks = []
    async for chunk in anthropic_async_model.execute(
        make_prompt(),
        stream=True,
        response=anthropic_response,
        conversation=None,
        key="sk-test",
    ):
        chunks.append(chunk)
    assert chunks == []
    anthropic_response.set_usage.assert_called_once_with(input=10, output=5)


def test_anthropic_prompt_empty_user(
    anthropic_sync_model,
    mocked_sync_anthropic_client,
    anthropic_response,
):
    mocked_sync_anthropic_client.messages.create.return_value = make_message()
    list(
        anthropic_sync_model.execute(
            make_prompt(prompt_text=""),
            stream=False,
            response=anthropic_response,
            conversation=None,
            key="sk-test",
        )
    )
    call_kwargs = mocked_sync_anthropic_client.messages.create.call_args[1]
    assert call_kwargs["messages"] == []


@pytest.mark.asyncio
async def test_anthropic_prompt_empty_user_async(
    anthropic_async_model,
    mocked_async_anthropic_client,
    anthropic_response,
):
    mocked_async_anthropic_client.messages.create.return_value = make_message()
    async for _ in anthropic_async_model.execute(
        make_prompt(prompt_text=""),
        stream=False,
        response=anthropic_response,
        conversation=None,
        key="sk-test",
    ):
        pass
    call_kwargs = mocked_async_anthropic_client.messages.create.call_args[1]
    assert call_kwargs["messages"] == []


def test_anthropic_execute_raises_on_auth_error(
    anthropic_sync_model,
    mocked_sync_anthropic_client,
    anthropic_response,
):
    mocked_sync_anthropic_client.messages.create.side_effect = AuthenticationError(
        message="invalid x-api-key", response=MagicMock(), body=None
    )
    with pytest.raises(AuthenticationError):
        list(
            anthropic_sync_model.execute(
                make_prompt(),
                stream=False,
                response=anthropic_response,
                conversation=None,
                key="sk-test",
            )
        )


@pytest.mark.asyncio
async def test_anthropic_execute_raises_on_auth_error_async(
    anthropic_async_model,
    mocked_async_anthropic_client,
    anthropic_response,
):
    mocked_async_anthropic_client.messages.create.side_effect = AuthenticationError(
        message="invalid x-api-key", response=MagicMock(), body=None
    )
    with pytest.raises(AuthenticationError):
        async for _ in anthropic_async_model.execute(
            make_prompt(),
            stream=False,
            response=anthropic_response,
            conversation=None,
            key="sk-test",
        ):
            pass


def test_anthropic_execute_raises_on_network_error(
    anthropic_sync_model,
    mocked_sync_anthropic_client,
    anthropic_response,
):
    mocked_sync_anthropic_client.messages.create.side_effect = APIConnectionError(
        request=MagicMock()
    )
    with pytest.raises(APIConnectionError):
        list(
            anthropic_sync_model.execute(
                make_prompt(),
                stream=False,
                response=anthropic_response,
                conversation=None,
                key="sk-test",
            )
        )


@pytest.mark.asyncio
async def test_anthropic_execute_raises_on_network_error_async(
    anthropic_async_model,
    mocked_async_anthropic_client,
    anthropic_response,
):
    mocked_async_anthropic_client.messages.create.side_effect = APIConnectionError(
        request=MagicMock()
    )
    with pytest.raises(APIConnectionError):
        async for _ in anthropic_async_model.execute(
            make_prompt(),
            stream=False,
            response=anthropic_response,
            conversation=None,
            key="sk-test",
        ):
            pass


def test_anthropic_execute_stream_raises_on_auth_error(
    anthropic_sync_model,
    mocked_sync_anthropic_client,
    anthropic_response,
):
    mocked_sync_anthropic_client.messages.stream.side_effect = AuthenticationError(
        message="invalid x-api-key", response=MagicMock(), body=None
    )
    with pytest.raises(AuthenticationError):
        list(
            anthropic_sync_model.execute(
                make_prompt(),
                stream=True,
                response=anthropic_response,
                conversation=None,
                key="sk-test",
            )
        )
