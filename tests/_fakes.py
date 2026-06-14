from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional
from unittest.mock import AsyncMock


@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class Usage:
    input_tokens: int = 10
    output_tokens: int = 5


@dataclass
class Message:
    content: list
    usage: Usage

    def model_dump(self):
        return {"id": "msg_123"}


def make_message(text="Hello", input_tokens=10, output_tokens=5, content=None):
    if content is None:
        content = [TextBlock(text=text)] if text else []
    return Message(
        content=content,
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


@dataclass
class Prompt:
    prompt: Optional[str] = None
    system: Optional[str] = None
    options: SimpleNamespace = field(
        default_factory=lambda: SimpleNamespace(max_tokens=None, temperature=None)
    )


def make_prompt(prompt_text="Say hello", system=None, max_tokens=None, temperature=None):
    return Prompt(
        prompt=prompt_text,
        system=system,
        options=SimpleNamespace(max_tokens=max_tokens, temperature=temperature),
    )


@dataclass
class PrevResponse:
    prompt: Prompt
    text: str

    def text_or_raise(self):
        return self.text


@dataclass
class Conversation:
    responses: list = field(default_factory=list)


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


@dataclass
class Stream:
    text_stream: object
    final_message: Message

    def get_final_message(self):
        return self.final_message


@dataclass
class StreamContext:
    stream: Stream

    def __enter__(self):
        return self.stream

    def __exit__(self, *args):
        return False


@dataclass
class AsyncStreamContext:
    stream: Stream

    async def __aenter__(self):
        return self.stream

    async def __aexit__(self, *args):
        pass


def make_sync_stream(text_chunks=("\n\nHello", " world"), input_tokens=10, output_tokens=5):
    return StreamContext(
        stream=Stream(
            text_stream=text_chunks,
            final_message=make_message(
                input_tokens=input_tokens, output_tokens=output_tokens
            ),
        )
    )


def make_async_stream(text_chunks=("\n\nHello", " world"), input_tokens=10, output_tokens=5):
    final_message = make_message(
        input_tokens=input_tokens, output_tokens=output_tokens
    )
    stream = Stream(
        text_stream=AsyncIteratorWrapper(text_chunks),
        final_message=final_message,
    )
    stream.get_final_message = AsyncMock(return_value=final_message)
    return AsyncStreamContext(stream=stream)
