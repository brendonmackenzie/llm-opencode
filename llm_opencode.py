import csv
import io
import json
import re
import time
from html.parser import HTMLParser
from pathlib import Path

import click
import llm
from anthropic import Anthropic, AsyncAnthropic
from llm.default_plugins.openai_models import AsyncChat, Chat
from pydantic import Field

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

DOCS_PAGE_URL = "https://opencode.ai/docs/go/"

UNIFIED_MODELS_CACHE_TIMEOUT = 86400  # 24 hours


class DocsScrapeError(Exception):
    pass


class _DocsTableParser(HTMLParser):
    """Parse HTML into a list of tables, each with ``headers`` and ``rows``."""

    def __init__(self):
        super().__init__()
        self.tables = []
        self._table_stack = []
        self._row = None
        self._cell = None
        self._in_headers = False

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._table_stack.append({"headers": [], "rows": []})
        elif tag == "thead":
            self._in_headers = True
        elif tag == "tbody":
            self._in_headers = False
        elif tag == "tr" and self._table_stack and self._row is None:
            self._row = []
        elif tag in ("th", "td") and self._row is not None and self._cell is None:
            self._cell = []

    def handle_endtag(self, tag):
        if tag == "table":
            if self._table_stack:
                self.tables.append(self._table_stack.pop())
        elif tag == "thead":
            self._in_headers = False
        elif tag == "tr":
            if self._row is not None and self._table_stack:
                table = self._table_stack[-1]
                if self._in_headers:
                    table["headers"].extend(self._row)
                else:
                    table["rows"].append(self._row)
            self._row = None
        elif tag in ("th", "td") and self._cell is not None:
            if self._row is not None:
                self._row.append("".join(self._cell).strip())
            self._cell = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def parse_html_tables(html):
    """Parse HTML into a list of tables with ``headers`` and ``rows``."""
    parser = _DocsTableParser()
    parser.feed(html)
    parser.close()
    return parser.tables


def fetch_docs_page():
    """Fetch the OpenCode Go docs page HTML."""
    try:
        import httpx

        response = httpx.get(DOCS_PAGE_URL, follow_redirects=True)
        response.raise_for_status()
        return response.text
    except httpx.HTTPError as error:
        raise DocsScrapeError(f"Failed to fetch docs page: {error}") from error


def _normalize_name(name):
    """Normalize a model display name for matching against Model ID rows."""
    return re.sub(r"[\s_-]+", " ", name.lower().strip())


def _parse_price(value):
    """Parse '$2.00' or '15' into a float, or None for '-' or empty."""
    if not value or value == "-":
        return None
    cleaned = re.sub(r"[^\d.]", "", value)
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def parse_docs_tables(tables):
    """Parse the docs page tables into model data keyed by model ID.

    Tables are identified by their headers, not their position:
    - a table with a "Model ID" column maps display names to model IDs
    - a table with "Input"/"Cached Read" columns holds pricing
    - a table with "requests per" columns holds rate limits
    - a table with "Model training"/"Data retention" columns holds policies
    """
    docs_data = {}
    name_to_id = {}

    for table in tables:
        headers = [h.lower() for h in table["headers"]]
        if "model id" in headers:
            for row in table["rows"]:
                if len(row) >= 2:
                    name_to_id[_normalize_name(row[0])] = row[1]

    for table in tables:
        headers = [h.lower() for h in table["headers"]]
        for row in table["rows"]:
            if not row:
                continue
            model_id = name_to_id.get(_normalize_name(row[0]), row[0])
            entry = docs_data.setdefault(model_id, {})
            entry["model_name"] = row[0]
            if "model id" in headers:
                if len(row) >= 4:
                    entry["endpoint"] = row[2]
                    entry["ai_sdk_package"] = row[3]
            elif "cached read" in headers:
                if len(row) >= 6:
                    entry["pricing"] = {
                        "input": _parse_price(row[1]),
                        "output": _parse_price(row[2]),
                        "cached_read": _parse_price(row[3]),
                        "cached_write": _parse_price(row[4]),
                        "usage": _parse_price(row[5]),
                        "currency": "USD",
                    }
            elif any("requests per" in header for header in headers):
                if len(row) >= 4:
                    entry["rate_limits"] = {
                        "per_5_hours": row[1].replace(",", ""),
                        "per_week": row[2].replace(",", ""),
                        "per_month": row[3].replace(",", ""),
                    }
            elif (
                "model training" in headers or "data retention" in headers
            ) and len(row) >= 3:
                entry["model_training"] = row[1]
                entry["data_retention"] = row[2]

    for model_id, entry in list(docs_data.items()):
        variant_name = entry.get("model_name", "")
        if "(" in variant_name:
            base_name = variant_name.split("(")[0].strip()
            base_id = name_to_id.get(_normalize_name(base_name))
            base_entry = docs_data.get(base_id) if base_id else None
            if base_entry and base_id != model_id:
                for field in (
                    "endpoint",
                    "ai_sdk_package",
                    "model_training",
                    "data_retention",
                    "rate_limits",
                ):
                    if field not in entry and field in base_entry:
                        entry[field] = base_entry[field]

    return docs_data


_DEFAULT_PRICING = {
    "input": None,
    "output": None,
    "cached_read": None,
    "cached_write": None,
    "usage": None,
    "currency": "USD",
}

_DEFAULT_RATE_LIMITS = {"per_5_hours": "", "per_week": "", "per_month": ""}


def _unified_entry(model_id, api_model, docs_entry):
    return {
        "model_id": model_id,
        "model_name": docs_entry.get("model_name", ""),
        "provider": api_model.get("owned_by", ""),
        "created": api_model.get("created"),
        "endpoint": docs_entry.get("endpoint", ""),
        "ai_sdk_package": docs_entry.get("ai_sdk_package", ""),
        "model_training": docs_entry.get("model_training", ""),
        "data_retention": docs_entry.get("data_retention", ""),
        "pricing": docs_entry.get("pricing", dict(_DEFAULT_PRICING)),
        "rate_limits": docs_entry.get("rate_limits", dict(_DEFAULT_RATE_LIMITS)),
    }


def merge_model_data(api_models, docs_data):
    """Merge API model data with docs table data into a unified list."""
    unified = []
    for api_model in api_models:
        model_id = api_model.get("id", "")
        docs_entry = docs_data.get(model_id, {})
        unified.append(_unified_entry(model_id, api_model, docs_entry))

    api_ids = {m.get("id", "") for m in api_models}
    for model_id, docs_entry in docs_data.items():
        if model_id not in api_ids:
            unified.append(_unified_entry(model_id, {}, docs_entry))

    return unified


def fetch_unified_models():
    """Fetch API and docs model data, merged into a unified list.

    Results are cached for 24 hours. Raises DocsScrapeError if the docs
    page cannot be fetched or parsed.
    """
    cache_path = llm.user_dir() / "opencode_unified_models.json"
    if (
        cache_path.is_file()
        and time.time() - cache_path.stat().st_mtime < UNIFIED_MODELS_CACHE_TIMEOUT
    ):
            try:
                with open(cache_path) as file:
                    return json.load(file)
            except (json.JSONDecodeError, OSError):
                pass

    api_models = get_opencode_models()
    tables = parse_html_tables(fetch_docs_page())
    if not tables:
        raise DocsScrapeError("No tables found on the OpenCode Go docs page")
    docs_data = parse_docs_tables(tables)
    unified = merge_model_data(api_models, docs_data)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as file:
        json.dump(unified, file)

    return unified


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


class _OpenCodeGoAnthropicChatBase:
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
        client = Anthropic(
            api_key=self.get_key(key),
            base_url=BASE_URL_ANTHROPIC,
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
        client = AsyncAnthropic(
            api_key=self.get_key(key),
            base_url=BASE_URL_ANTHROPIC,
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
                    supports_schema=True,
                ),
                OpenCodeGoAsyncChat(
                    model_id=llm_model_id,
                    model_name=model_id,
                    api_base=BASE_URL_OPENAI,
                    supports_schema=True,
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
            try:
                with open(path, "r") as file:
                    return json.load(file)
            except (json.JSONDecodeError, OSError):
                pass

    try:
        import httpx

        response = httpx.get(url, follow_redirects=True)
        response.raise_for_status()
        with open(path, "w") as file:
            json.dump(response.json(), file)
        return response.json()
    except httpx.HTTPError:
        if path.is_file():
            try:
                with open(path, "r") as file:
                    return json.load(file)
            except (json.JSONDecodeError, OSError):
                raise DownloadError(
                    f"Failed to download data and no valid cache is available at {path}"
                )
        else:
            raise DownloadError(
                f"Failed to download data and no cache is available at {path}"
            )


@llm.hookimpl
def register_commands(cli):
    @cli.group(invoke_without_command=True)
    @click.pass_context
    def opencode(ctx):
        "Commands relating to the llm-opencode plugin"
        if ctx.invoked_subcommand is None:
            ctx.invoke(models, json_=False)

    @opencode.group(invoke_without_command=True)
    @click.option("json_", "--json", is_flag=True, help="Output as JSON")
    @click.pass_context
    def models(ctx, json_):
        "List OpenCode Go models"
        if ctx.invoked_subcommand is not None:
            return
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

    @models.command()
    @click.option(
        "--format",
        "-f",
        "format_",
        type=click.Choice(["table", "csv", "pricing", "json"]),
        default="table",
        help="Output format",
    )
    @click.option("--search", help="Search models by name or ID")
    @click.option("--provider", help="Filter by provider")
    @click.option("--model", "model_id", help="Filter by exact model ID")
    @click.option(
        "--sort-by",
        type=click.Choice(["requests", "input", "output", "usage", "name", "model_id"]),
        default="requests",
        help="Sort order (pricing format always sorts by input price)",
    )
    @click.option("-o", "--output", type=click.Path(), help="Write output to file")
    def detail(format_, sort_by, search, provider, model_id, output):
        "Show detailed model info including pricing, rate limits, and policies"
        try:
            unified = fetch_unified_models()
        except DocsScrapeError:
            click.echo(
                "OpenCode Go page scraping failed. Please click here for more details:"
            )
            click.echo(DOCS_PAGE_URL)
            raise click.exceptions.Exit(1)

        if model_id:
            query = model_id.lower()
            unified = [m for m in unified if m.get("model_id", "").lower() == query]
        if provider:
            query = provider.lower()
            unified = [m for m in unified if query in m.get("provider", "").lower()]
        if search:
            query = search.lower()
            unified = [
                m
                for m in unified
                if query in m.get("model_id", "").lower()
                or query in m.get("model_name", "").lower()
            ]

        formatters = {
            "table": format_unified_table,
            "csv": format_unified_csv,
            "pricing": format_unified_pricing,
            "json": format_unified_json,
        }
        if format_ != "pricing":
            unified = sort_models(unified, sort_by)
        result = formatters[format_](unified)

        if output:
            with open(output, "w", encoding="utf-8") as file:
                file.write(result)
        else:
            click.echo(result)


def _parse_rate_limit(value):
    """Parse a rate limit like '1,080' into an int, or 0 when missing/invalid."""
    if not value:
        return 0
    try:
        return int(value.replace(",", ""))
    except (ValueError, TypeError):
        return 0


def _requests_per_month_key(model):
    """Sort key: highest requests per month first, then cheapest input."""
    rate_limits = model.get("rate_limits", {})
    per_month = _parse_rate_limit(rate_limits.get("per_month", ""))
    input_price = model.get("pricing", {}).get("input")
    input_sort = input_price if input_price is not None else float("inf")
    if per_month > 0:
        return (False, -per_month, input_sort)
    return (True, 0, input_sort)


def _price_key(model, field):
    """Sort key for a pricing field: ascending, missing values pinned to the end."""
    value = model.get("pricing", {}).get(field)
    if value is not None:
        return (False, value)
    return (True, 0.0)


def _name_key(model):
    name = model.get("model_name") or ""
    return (not name, name.lower())


def _model_id_key(model):
    model_id = model.get("model_id") or ""
    return (not model_id, model_id.lower())


def sort_models(models, sort_by="requests"):
    """Sort unified models by the given key (default: requests per month, highest first)."""
    keys = {
        "requests": _requests_per_month_key,
        "input": lambda m: _price_key(m, "input"),
        "output": lambda m: _price_key(m, "output"),
        "usage": lambda m: _price_key(m, "usage"),
        "name": _name_key,
        "model_id": _model_id_key,
    }
    return sorted(models, key=keys[sort_by])


def format_unified_json(models):
    """Format unified models as JSON with metadata."""
    meta = {
        "sources": [MODELS_URL, DOCS_PAGE_URL],
    }
    return json.dumps({"count": len(models), "models": models, "meta": meta}, indent=2)


def format_unified_table(models):
    """Format unified models as a pretty pricing table."""
    if not models:
        return "No models found."

    def _fmt_price(value):
        return f"${value:.2f}" if value is not None else "—"

    def _fmt_requests_per_month(value):
        return f"{value:,}" if value > 0 else "—"

    rows = []
    for model in models:
        pricing = model.get("pricing", {})
        rate_limits = model.get("rate_limits", {})
        per_month = _parse_rate_limit(rate_limits.get("per_month", ""))
        rows.append(
            [
                model.get("model_id", "")[:25],
                model.get("model_name", "")[:20],
                _fmt_requests_per_month(per_month),
                _fmt_price(pricing.get("input")),
                _fmt_price(pricing.get("output")),
                _fmt_price(pricing.get("cached_read")),
                _fmt_price(pricing.get("cached_write")),
                _fmt_price(pricing.get("usage")),
                model.get("data_retention", "")[:10] or "—",
            ]
        )

    headers = [
        "Model ID", "Name", "Req/Month", "Input", "Output",
        "Cache R", "Cache W", "Usage", "Retention",
    ]
    widths = [
        max(len(header), max((len(row[i]) for row in rows), default=0))
        for i, header in enumerate(headers)
    ]

    lines = []
    lines.append("  ".join(header.ljust(width) for header, width in zip(headers, widths)))
    lines.append("  ".join("─" * width for width in widths))
    for row in rows:
        lines.append("  ".join(cell.ljust(width) for cell, width in zip(row, widths)))
    return "\n".join(lines)


def format_unified_csv(models):
    """Format unified models as CSV."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "model_id",
            "model_name",
            "provider",
            "input_price",
            "output_price",
            "cached_read",
            "cached_write",
            "usage",
            "endpoint",
            "ai_sdk_package",
            "data_retention",
            "model_training",
            "rate_per_5h",
            "rate_per_week",
            "rate_per_month",
        ]
    )
    for model in models:
        pricing = model.get("pricing", {})
        rate_limits = model.get("rate_limits", {})
        writer.writerow(
            [
                model.get("model_id", ""),
                model.get("model_name", ""),
                model.get("provider", ""),
                pricing.get("input", ""),
                pricing.get("output", ""),
                pricing.get("cached_read", ""),
                pricing.get("cached_write", ""),
                pricing.get("usage", ""),
                model.get("endpoint", ""),
                model.get("ai_sdk_package", ""),
                model.get("data_retention", ""),
                model.get("model_training", ""),
                rate_limits.get("per_5_hours", ""),
                rate_limits.get("per_week", ""),
                rate_limits.get("per_month", ""),
            ]
        )
    return buf.getvalue()


def format_unified_pricing(models):
    """Format unified models as a pricing comparison sorted by input cost."""
    priced = [m for m in models if m.get("pricing", {}).get("input") is not None]
    priced.sort(key=lambda m: m["pricing"]["input"])

    if not priced:
        return "No pricing data found."

    lines = ["Pricing Comparison (per 1M tokens)", "═" * 70]
    for model in priced:
        pricing = model["pricing"]
        name = model.get("model_id") or model.get("model_name") or "?"
        lines.append(f"\n  {name}")
        lines.append(f"    Input:        ${pricing.get('input') or 0:.2f}")
        lines.append(f"    Output:       ${pricing.get('output') or 0:.2f}")
        if pricing.get("cached_read") is not None:
            lines.append(f"    Cached Read:  ${pricing['cached_read']:.2f}")
        else:
            lines.append("    Cached Read:  —")
        if pricing.get("cached_write") is not None:
            lines.append(f"    Cached Write: ${pricing['cached_write']:.2f}")
        else:
            lines.append("    Cached Write: —")
        lines.append(f"    Usage:        ${pricing.get('usage') or 0:.2f}")
        lines.append(f"    Retention:    {model.get('data_retention', '—')}")
    return "\n".join(lines)