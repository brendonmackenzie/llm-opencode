import contextvars
import uuid

import llm
from anthropic import Anthropic, AsyncAnthropic
from llm.default_plugins.openai_models import (
    AsyncChat,
    AsyncResponses,
    Chat,
    Responses,
)
from pydantic import Field

from .cli import (
    format_unified_csv,
    format_unified_json,
    format_unified_pricing,
    format_unified_table,
    register_commands,
    sort_models,
)
from .docs_scraper import (
    DOCS_PAGE_URL,
    DocsScrapeError,
    fetch_docs_page,
    parse_docs_tables,
    parse_html_tables,
)

from .routing import (
    BASE_URL_ANTHROPIC,
    BASE_URL_OPENAI,
    SKIP_GOOGLE_STYLE,
    SKIP_NO_DOCS_ROW,
    SKIP_UNRECOGNIZED_PATH,
    analyze_routes,
    corroboration_warnings,
    derive_route,
    register_models,
    unregistered_routes,
)
from .unified import (
    MODELS_URL,
    UNIFIED_MODELS_CACHE_TIMEOUT,
    DownloadError,
    fetch_cached_json,
    fetch_unified_models,
    get_opencode_models,
    merge_model_data,
)

__all__ = [
    "BASE_URL_ANTHROPIC",
    "BASE_URL_OPENAI",
    "DOCS_PAGE_URL",
    "DocsScrapeError",
    "DownloadError",
    "MODELS_URL",
    "SKIP_GOOGLE_STYLE",
    "SKIP_NO_DOCS_ROW",
    "SKIP_UNRECOGNIZED_PATH",
    "UNIFIED_MODELS_CACHE_TIMEOUT",
    "OpenCodeGoAnthropicAsyncChat",
    "OpenCodeGoAnthropicChat",
    "OpenCodeGoAsyncChat",
    "OpenCodeGoChat",
    "OpenCodeGoResponsesAsyncChat",
    "OpenCodeGoResponsesChat",
    "analyze_routes",
    "corroboration_warnings",
    "derive_route",
    "fetch_cached_json",
    "fetch_docs_page",
    "fetch_unified_models",
    "format_unified_csv",
    "format_unified_json",
    "format_unified_pricing",
    "format_unified_table",
    "get_opencode_models",
    "merge_model_data",
    "parse_docs_tables",
    "parse_html_tables",
    "register_commands",
    "register_models",
    "sort_models",
    "unregistered_routes",
]

_SESSION_ID: contextvars.ContextVar = contextvars.ContextVar(
    "opencode_session_id", default=None
)


def _resolve_session_id(conversation):
    if conversation is not None and conversation.id:
        return conversation.id
    return str(uuid.uuid4())


class OpenCodeGoChat(Chat):
    needs_key = "opencode"
    key_env_var = "OPENCODE_KEY"

    def __str__(self):
        return f"OpenCode Go: {self.model_id}"

    def build_kwargs(self, prompt, stream):
        kwargs = super().build_kwargs(prompt, stream)
        session_id = _SESSION_ID.get()
        if session_id:
            kwargs["extra_headers"] = {"x-opencode-session": session_id}
        return kwargs

    def execute(self, prompt, stream, response, conversation=None, key=None):
        token = _SESSION_ID.set(_resolve_session_id(conversation))
        try:
            yield from super().execute(prompt, stream, response, conversation, key)
        finally:
            _SESSION_ID.reset(token)


class OpenCodeGoAsyncChat(AsyncChat):
    needs_key = "opencode"
    key_env_var = "OPENCODE_KEY"

    def __str__(self):
        return f"OpenCode Go: {self.model_id}"

    def build_kwargs(self, prompt, stream):
        kwargs = super().build_kwargs(prompt, stream)
        session_id = _SESSION_ID.get()
        if session_id:
            kwargs["extra_headers"] = {"x-opencode-session": session_id}
        return kwargs

    async def execute(self, prompt, stream, response, conversation=None, key=None):
        token = _SESSION_ID.set(_resolve_session_id(conversation))
        try:
            async for chunk in super().execute(prompt, stream, response, conversation, key):
                yield chunk
        finally:
            _SESSION_ID.reset(token)


class OpenCodeGoResponsesChat(Responses):
    needs_key = "opencode"
    key_env_var = "OPENCODE_KEY"

    def __str__(self):
        return f"OpenCode Go: {self.model_id}"

    def get_client(self, key, *, async_=False):
        session_id = _SESSION_ID.get()
        if not session_id:
            return super().get_client(key, async_=async_)
        original_headers = self.headers
        try:
            self.headers = {
                **(original_headers or {}),
                "x-opencode-session": session_id,
            }
            return super().get_client(key, async_=async_)
        finally:
            self.headers = original_headers

    def execute(self, prompt, stream, response, conversation=None, key=None):
        token = _SESSION_ID.set(_resolve_session_id(conversation))
        try:
            yield from super().execute(prompt, stream, response, conversation, key)
        finally:
            _SESSION_ID.reset(token)


class OpenCodeGoResponsesAsyncChat(AsyncResponses):
    needs_key = "opencode"
    key_env_var = "OPENCODE_KEY"

    def __str__(self):
        return f"OpenCode Go: {self.model_id}"

    def get_client(self, key, *, async_=False):
        session_id = _SESSION_ID.get()
        if not session_id:
            return super().get_client(key, async_=async_)
        original_headers = self.headers
        try:
            self.headers = {
                **(original_headers or {}),
                "x-opencode-session": session_id,
            }
            return super().get_client(key, async_=async_)
        finally:
            self.headers = original_headers

    async def execute(self, prompt, stream, response, conversation=None, key=None):
        token = _SESSION_ID.set(_resolve_session_id(conversation))
        try:
            async for chunk in super().execute(prompt, stream, response, conversation, key):
                yield chunk
        finally:
            _SESSION_ID.reset(token)


class _OpenCodeGoAnthropicChatBase:
    needs_key = "opencode"
    key_env_var = "OPENCODE_KEY"
    can_stream = True

    def __init__(self, model_id, anthropic_model_id=None, api_base=None):
        self.model_id = model_id
        self.anthropic_model_id = anthropic_model_id or model_id.replace(
            "opencode-go/", ""
        )
        self.api_base = api_base or BASE_URL_ANTHROPIC
        self.attachment_types = set()

    class Options(llm.Options):
        max_tokens: int | None = Field(
            description="Maximum number of tokens to generate",
            default=None,
        )
        temperature: float | None = Field(
            description="Temperature (0.0-1.0)",
            default=None,
        )

    def __str__(self):
        return f"OpenCode Go: {self.model_id}"

    def _build_messages(self, prompt, conversation):
        messages = []
        if conversation:
            for prev_response in conversation.responses:
                user_content = prev_response.prompt.prompt
                if user_content:
                    messages.append({"role": "user", "content": user_content})
                messages.append(
                    {"role": "assistant", "content": prev_response.text_or_raise()}
                )
        if prompt.prompt:
            messages.append({"role": "user", "content": prompt.prompt})
        return messages

    def _build_kwargs(self, prompt, conversation):
        kwargs = {
            "model": self.anthropic_model_id,
            "messages": self._build_messages(prompt, conversation),
        }
        if prompt.system:
            kwargs["system"] = prompt.system
        kwargs["max_tokens"] = prompt.options.max_tokens or 4096
        if prompt.options.temperature is not None:
            kwargs["temperature"] = prompt.options.temperature
        return kwargs

    def _iter_text_blocks(self, message):
        if message.content:
            for block in message.content:
                if block.type == "text":
                    yield block.text

    def _apply_response(self, response, message):
        response.response_json = message.model_dump()
        response.set_usage(
            input=message.usage.input_tokens,
            output=message.usage.output_tokens,
        )


class OpenCodeGoAnthropicChat(_OpenCodeGoAnthropicChatBase, llm.KeyModel):
    def execute(self, prompt, stream, response, conversation, key):
        session_id = _resolve_session_id(conversation)
        client = Anthropic(
            api_key=self.get_key(key),
            base_url=self.api_base,
            default_headers={"x-opencode-session": session_id},
        )
        kwargs = self._build_kwargs(prompt, conversation)

        if stream:
            with client.messages.stream(**kwargs) as stream_obj:
                started = False
                for text in stream_obj.text_stream:
                    if not started:
                        if text.strip():
                            started = True
                            yield text.lstrip()
                        continue
                    yield text
            final_message = stream_obj.get_final_message()
            self._apply_response(response, final_message)
        else:
            message = client.messages.create(**kwargs)
            yield from self._iter_text_blocks(message)
            self._apply_response(response, message)


class OpenCodeGoAnthropicAsyncChat(_OpenCodeGoAnthropicChatBase, llm.AsyncKeyModel):
    async def execute(self, prompt, stream, response, conversation, key):
        session_id = _resolve_session_id(conversation)
        client = AsyncAnthropic(
            api_key=self.get_key(key),
            base_url=self.api_base,
            default_headers={"x-opencode-session": session_id},
        )
        kwargs = self._build_kwargs(prompt, conversation)

        if stream:
            async with client.messages.stream(**kwargs) as stream_obj:
                started = False
                async for text in stream_obj.text_stream:
                    if not started:
                        if text.strip():
                            started = True
                            yield text.lstrip()
                        continue
                    yield text
            final_message = await stream_obj.get_final_message()
            self._apply_response(response, final_message)
        else:
            message = await client.messages.create(**kwargs)
            for text in self._iter_text_blocks(message):
                yield text
            self._apply_response(response, message)
