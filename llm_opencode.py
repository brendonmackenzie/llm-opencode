import click
import json
import llm
import time
from anthropic import Anthropic, AsyncAnthropic
from llm.default_plugins.openai_models import Chat, AsyncChat
from pathlib import Path
from pydantic import Field
from typing import Optional


OPENAI_PROTOCOL_MODELS = {
    "glm-5",
    "glm-5.1",
    "kimi-k2.5",
    "kimi-k2.6",
    "mimo-v2",
    "mimo-v2-pro",
    "mimo-v2.5",
    "mimo-v2.5-pro",
    "mimo-v2-omni",
    "deepseek-v4-pro",
    "deepseek-v4-flash",
}

ANTHROPIC_PROTOCOL_MODELS = {
    "minimax-m3",
    "minimax-m2.7",
    "minimax-m2.5",
    "qwen3.5-plus",
    "qwen3.6-plus",
    "qwen3.7-plus",
    "qwen3.7-max",
}

UNKNOWN_MODELS_DEFAULT = "openai"

BASE_URL_OPENAI = "https://opencode.ai/zen/go/v1"
BASE_URL_ANTHROPIC = "https://opencode.ai/zen/go"

MODELS_URL = "https://opencode.ai/zen/go/v1/models"


class OpenCodeGoChat(Chat):
    needs_key = "opencode"
    key_env_var = "OPENCODE_KEY"

    def __str__(self):
        return f"OpenCode Go: {self.model_id}"


class OpenCodeGoAsyncChat(AsyncChat):
    needs_key = "opencode"
    key_env_var = "OPENCODE_KEY"

    def __str__(self):
        return f"OpenCode Go: {self.model_id}"


class OpenCodeGoAnthropicChat(llm.KeyModel):
    needs_key = "opencode"
    key_env_var = "OPENCODE_KEY"
    can_stream = True

    def __init__(self, model_id, anthropic_model_id=None):
        self.model_id = model_id
        self.anthropic_model_id = anthropic_model_id or model_id.replace(
            "opencode-go/", ""
        )
        self.attachment_types = set()

    class Options(llm.Options):
        max_tokens: Optional[int] = Field(
            description="Maximum number of tokens to generate",
            default=None,
        )
        temperature: Optional[float] = Field(
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

    def execute(self, prompt, stream, response, conversation, key):
        client = Anthropic(
            api_key=self.get_key(key),
            base_url=BASE_URL_ANTHROPIC,
        )

        kwargs = {
            "model": self.anthropic_model_id,
            "messages": self._build_messages(prompt, conversation),
        }

        if prompt.system:
            kwargs["system"] = prompt.system

        max_tokens = prompt.options.max_tokens or 4096
        kwargs["max_tokens"] = max_tokens

        if prompt.options.temperature is not None:
            kwargs["temperature"] = prompt.options.temperature

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
            response.response_json = final_message.model_dump()
            response.set_usage(
                input=final_message.usage.input_tokens,
                output=final_message.usage.output_tokens,
            )
        else:
            message = client.messages.create(**kwargs)
            if message.content:
                for block in message.content:
                    if block.type == "text":
                        yield block.text
            response.response_json = message.model_dump()
            response.set_usage(
                input=message.usage.input_tokens,
                output=message.usage.output_tokens,
            )


class OpenCodeGoAnthropicAsyncChat(llm.AsyncKeyModel):
    needs_key = "opencode"
    key_env_var = "OPENCODE_KEY"
    can_stream = True

    def __init__(self, model_id, anthropic_model_id=None):
        self.model_id = model_id
        self.anthropic_model_id = anthropic_model_id or model_id.replace(
            "opencode-go/", ""
        )
        self.attachment_types = set()

    class Options(llm.Options):
        max_tokens: Optional[int] = Field(
            description="Maximum number of tokens to generate",
            default=None,
        )
        temperature: Optional[float] = Field(
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

    async def execute(self, prompt, stream, response, conversation, key):
        client = AsyncAnthropic(
            api_key=self.get_key(key),
            base_url=BASE_URL_ANTHROPIC,
        )

        kwargs = {
            "model": self.anthropic_model_id,
            "messages": self._build_messages(prompt, conversation),
        }

        if prompt.system:
            kwargs["system"] = prompt.system

        max_tokens = prompt.options.max_tokens or 4096
        kwargs["max_tokens"] = max_tokens

        if prompt.options.temperature is not None:
            kwargs["temperature"] = prompt.options.temperature

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
            response.response_json = final_message.model_dump()
            response.set_usage(
                input=final_message.usage.input_tokens,
                output=final_message.usage.output_tokens,
            )
        else:
            message = await client.messages.create(**kwargs)
            if message.content:
                for block in message.content:
                    if block.type == "text":
                        yield block.text
            response.response_json = message.model_dump()
            response.set_usage(
                input=message.usage.input_tokens,
                output=message.usage.output_tokens,
            )


def get_opencode_models():
    models = fetch_cached_json(
        url=MODELS_URL,
        path=llm.user_dir() / "opencode_models.json",
        cache_timeout=3600,
    )["data"]
    return models


def _get_protocol(model_id):
    if model_id in ANTHROPIC_PROTOCOL_MODELS:
        return "anthropic"
    return "openai"


@llm.hookimpl
def register_models(register):
    key = llm.get_key("", "opencode", "OPENCODE_KEY")
    if not key:
        return
    for model_definition in get_opencode_models():
        model_id = model_definition["id"]
        llm_model_id = f"opencode-go/{model_id}"
        protocol = _get_protocol(model_id)

        if protocol == "openai":
            register(
                OpenCodeGoChat(
                    model_id=llm_model_id,
                    model_name=model_id,
                    api_base=BASE_URL_OPENAI,
                ),
                OpenCodeGoAsyncChat(
                    model_id=llm_model_id,
                    model_name=model_id,
                    api_base=BASE_URL_OPENAI,
                ),
            )
        elif protocol == "anthropic":
            register(
                OpenCodeGoAnthropicChat(model_id=llm_model_id),
                OpenCodeGoAnthropicAsyncChat(model_id=llm_model_id),
            )


class DownloadError(Exception):
    pass


def fetch_cached_json(url, path, cache_timeout):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.is_file():
        mod_time = path.stat().st_mtime
        if time.time() - mod_time < cache_timeout:
            with open(path, "r") as file:
                return json.load(file)

    try:
        import httpx

        response = httpx.get(url, follow_redirects=True)
        response.raise_for_status()
        with open(path, "w") as file:
            json.dump(response.json(), file)
        return response.json()
    except httpx.HTTPError:
        if path.is_file():
            with open(path, "r") as file:
                return json.load(file)
        else:
            raise DownloadError(
                f"Failed to download data and no cache is available at {path}"
            )


@llm.hookimpl
def register_commands(cli):
    @cli.group()
    def opencode():
        "Commands relating to the llm-opencode plugin"

    @opencode.command()
    @click.option("json_", "--json", is_flag=True, help="Output as JSON")
    def models(json_):
        "List OpenCode Go models"
        all_models = get_opencode_models()
        if json_:
            click.echo(json.dumps(all_models, indent=2))
        else:
            for model in all_models:
                model_id = model["id"]
                protocol = _get_protocol(model_id)
                endpoint = (
                    "/v1/chat/completions"
                    if protocol == "openai"
                    else "/v1/messages"
                )
                click.echo(f"- id: {model_id}")
                click.echo(f"  protocol: {protocol}")
                click.echo(f"  endpoint: https://opencode.ai/zen/go{endpoint}")
                click.echo()