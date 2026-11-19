import csv
import io
import json
import time
from datetime import datetime, timezone

import click
import llm

from .routing import analyze_routes, unregistered_routes
from .unified import (
    MODELS_URL,
    SNAPSHOT_REFRESH_TIMEOUT,
    DownloadError,
    snapshot_age,
)

DEFAULT_PLAN = "go"
PLANS = ("go", "go-plus", "go-teams")

PROTOCOL_UNKNOWN = "unknown"


@llm.hookimpl
def register_commands(cli):
    @cli.group(invoke_without_command=True)
    @click.pass_context
    def opencode(ctx):
        "Commands relating to the llm-opencode plugin"
        if ctx.invoked_subcommand is None:
            click.echo(ctx.get_help())

    @opencode.command()
    @click.option(
        "--format",
        "-f",
        "format_",
        type=click.Choice(["table", "csv", "pricing", "json"]),
        default="table",
        help="Output format",
    )
    @click.option("--search", help="Search models by name or ID")
    @click.option("--model", "model_id", help="Filter by exact model ID")
    @click.option(
        "--schema-capable",
        is_flag=True,
        help="Only models that accept llm --schema structured output",
    )
    @click.option(
        "--plan",
        type=click.Choice([*PLANS, "all"]),
        default=DEFAULT_PLAN,
        help=(
            "Plan to show monthly usage for "
            "(JSON always carries every plan and ignores this filter)"
        ),
    )
    @click.option(
        "--sort-by",
        type=click.Choice(
            ["usage", "input", "output", "name", "model_id"]
        ),
        default="usage",
        help="Sort order (pricing format always sorts by input price)",
    )
    @click.option("-o", "--output", type=click.Path(), help="Write output to file")
    def detail(format_, sort_by, search, model_id, plan, schema_capable, output):
        "Show detailed model info including pricing, usage, and protocols"
        from llm_opencode import fetch_models_snapshot

        try:
            snapshot = fetch_models_snapshot()
        except DownloadError:
            click.echo(
                "OpenCode Go models API unavailable. Please try again later:"
            )
            click.echo(MODELS_URL)
            raise click.exceptions.Exit(1)

        fetched_at = snapshot.get("fetched_at")
        stale_since = _stale_since(fetched_at)
        unified = snapshot.get("models", [])
        if model_id:
            query = model_id.lower()
            unified = [m for m in unified if m.get("model_id", "").lower() == query]
        if search:
            query = search.lower()
            unified = [
                m
                for m in unified
                if query in m.get("model_id", "").lower()
                or query in m.get("model_name", "").lower()
            ]

        routes = analyze_routes(unified)
        if schema_capable:
            routes = [route for route in routes if route.get("supports_schema")]
        formatters = {
            "table": format_unified_table,
            "csv": format_unified_csv,
            "pricing": format_unified_pricing,
        }
        if format_ == "json":
            # JSON is unaffected by --plan: it always carries every plan, so
            # its order is defined by the default plan alone.
            routes = sort_models(routes, sort_by)
            result = format_unified_json(
                routes,
                warnings=snapshot.get("warnings", []),
                fetched_at=fetched_at,
                unregistered=unregistered_routes(routes),
                stale_since=stale_since,
            )
        else:
            if format_ != "pricing":
                routes = sort_models(routes, sort_by, plan=plan)
            result = formatters[format_](routes, plan=plan)
            if format_ == "table":
                blocks = []
                banner = _stale_banner(fetched_at)
                if banner:
                    blocks.append(banner)
                warnings = snapshot.get("warnings", [])
                if warnings:
                    blocks.append(
                        "Warnings:\n" + "\n".join(f"  {w}" for w in warnings)
                    )
                if blocks:
                    result = "\n\n".join([*blocks, result])

        if output:
            with open(output, "w", encoding="utf-8") as file:
                file.write(result)
        else:
            click.echo(result)


def _show_plans(plan):
    """The plan columns to render: every plan in ``all`` mode, else the one."""
    return PLANS if plan == "all" else (plan,)


def _row_usage(model, plan_column):
    """The plan's monthly usage allowance in dollars, or ``None``.

    Strict per-plan lookup: an allowance the API didn't price for this
    plan renders a dash, never another plan's number borrowed.
    """
    usage = model.get("monthly_usage")
    if isinstance(usage, dict):
        return usage.get(plan_column)
    return None


def _usage_key(model, plan=DEFAULT_PLAN):
    """Sort key: highest monthly usage first, then cheapest input.

    In ``all`` mode the most generous plan allowance represents the model
    (an ordering heuristic only — rendered rows stay strictly per-plan).
    """
    if plan == "all":
        usage = model.get("monthly_usage")
        value = max(usage.values(), default=None) if isinstance(usage, dict) else None
    else:
        value = _row_usage(model, plan)
    pricing = model.get("pricing") or {}
    input_price = pricing.get("input")
    input_sort = input_price if input_price is not None else float("inf")
    if value is not None:
        return (False, -value, input_sort)
    return (True, 0.0, input_sort)


def _price_key(model, field):
    """Sort key for a pricing field: ascending, missing values pinned to the end."""
    pricing = model.get("pricing") or {}
    value = pricing.get(field)
    if value is not None:
        return (False, value)
    return (True, 0.0)


def _name_key(model):
    name = model.get("model_name") or ""
    return (not name, name.lower())


def _model_id_key(model):
    model_id = model.get("model_id") or ""
    return (not model_id, model_id.lower())


def sort_models(models, sort_by="usage", plan=DEFAULT_PLAN):
    """Sort unified models by the given key (default: monthly usage, highest first)."""
    keys = {
        "usage": lambda m: _usage_key(m, plan),
        "input": lambda m: _price_key(m, "input"),
        "output": lambda m: _price_key(m, "output"),
        "name": _name_key,
        "model_id": _model_id_key,
    }
    return sorted(models, key=keys[sort_by])


def _stale_since(fetched_at):
    """ISO timestamp of the snapshot's age origin, only when it passed the
    24-hour refresh timeout (``None`` when the snapshot is fresh or has no
    usable ``fetched_at``)."""
    age = snapshot_age(fetched_at)
    if age is None or age <= SNAPSHOT_REFRESH_TIMEOUT:
        return None
    return datetime.fromtimestamp(float(fetched_at), tz=timezone.utc).isoformat()


def _stale_banner(fetched_at):
    """One-line staleness warning for table output; ``None`` when fresh."""
    stale_since = _stale_since(fetched_at)
    if stale_since is None:
        return None
    age_hours = (time.time() - float(fetched_at)) / 3600
    return (
        f"⚠ Snapshot is stale: {age_hours:.1f} hours old (last fetched "
        f"{stale_since}). The models API was unreachable at the last refresh "
        "attempt — showing the saved snapshot."
    )


def format_unified_json(
    models, warnings=None, fetched_at=None, unregistered=None, stale_since=None
):
    """Format unified models as JSON with metadata and an unregistered section."""
    meta = {
        "sources": [MODELS_URL],
        "fetched_at": fetched_at,
        "warnings": warnings or [],
    }
    if stale_since is not None:
        meta["stale_since"] = stale_since
    return json.dumps(
        {
            "count": len(models),
            "models": models,
            "unregistered": unregistered or [],
            "meta": meta,
        },
        indent=2,
    )


def _fmt_price(value):
    """Dollar render: ``.2f``, except below a cent where four decimals keep
    the API's real sub-cent cache prices visible ($0.003, not $0.00)."""
    if value is None:
        return "—"
    if value == 0:
        return "$0.00"
    if value < 0.01:
        return "$" + f"{value:.4f}".rstrip("0")
    return f"${value:.2f}"


def _fmt_cached_write(value):
    """TTL-keyed cache-write prices render as their cheapest tier + TTL."""
    if value is None:
        return "—"
    if isinstance(value, dict):
        if not value:
            return "—"
        ttl, price = min(value.items(), key=lambda item: item[1])
        return f"${price:g} ({ttl})"
    return _fmt_price(value)


def _fmt_usage(value):
    return f"${value:g}" if value is not None else "—"


def _fmt_tokens(value):
    if not value:
        return "—"
    if value >= 1_000_000:
        text = f"{value / 1_000_000:.1f}".rstrip("0").rstrip(".")
        return f"{text}M"
    if value >= 1_000:
        return f"{round(value / 1000)}K"
    return str(value)


def _fmt_flag(value):
    if value is None:
        return "—"
    return "yes" if value else "no"


def _cost_notes(model):
    """True when the API carries context-tiered prices or time-window
    multipliers the headline prices don't reflect."""
    overrides = model.get("cost_overrides") or {}
    if any(overrides.get(field) for field in ("input", "output")):
        return True
    return bool(model.get("cost_multipliers"))


def _render_table(headers, rows):
    widths = [
        max(len(header), max((len(str(row[i])) for row in rows), default=0))
        for i, header in enumerate(headers)
    ]
    lines = ["  ".join(header.ljust(width) for header, width in zip(headers, widths))]
    lines.append("  ".join("─" * width for width in widths))
    for row in rows:
        lines.append("  ".join(str(cell).ljust(width) for cell, width in zip(row, widths)))
    return lines


def format_unified_table(models, plan=DEFAULT_PLAN):
    """Format unified models as a pretty pricing table."""
    if not models:
        return "No models found."

    show_plans = _show_plans(plan)
    headers = [
        "Model ID", "Name", "Protocol", "Usage/Mo", "Input", "Output",
        "Cache R", "Cache W", "Ctx", "Schema", "Training", "ZDR",
    ]
    if plan == "all":
        headers.insert(0, "Plan")
    rows = []
    for model in models:
        protocol = model.get("protocol") or PROTOCOL_UNKNOWN
        pricing = model.get("pricing") or {}
        limits = model.get("limits") or {}
        for current_plan in show_plans:
            row = []
            if plan == "all":
                row.append(current_plan)
            row.extend(
                [
                    model.get("model_id", "")[:25],
                    model.get("model_name", "")[:20],
                    protocol,
                    _fmt_usage(_row_usage(model, current_plan)),
                    _fmt_price(pricing.get("input")),
                    _fmt_price(pricing.get("output")),
                    _fmt_price(pricing.get("cached_read")),
                    _fmt_cached_write(pricing.get("cached_write")),
                    _fmt_tokens(limits.get("context")),
                    _fmt_flag(model.get("supports_schema")),
                    _fmt_flag(model.get("training")),
                    _fmt_flag(model.get("zdr")),
                ]
            )
            rows.append(row)

    lines = _render_table(headers, rows)

    skipped = [model for model in models if model.get("registered") is False]
    if skipped:
        lines.append("")
        lines.append("Unregistered:")
        for model in skipped:
            model_id = str(model.get("model_id") or "?")
            reason = model.get("skip_reason") or ""
            lines.append(f"  {model_id} — {reason}")
    return "\n".join(lines)


CSV_HEADERS = [
    "model_id",
    "model_name",
    "canonical_id",
    "provider",
    "family",
    "plan",
    "protocol",
    "recommended_api",
    "supports_schema",
    "input_price",
    "output_price",
    "cached_read",
    "cached_write",
    "monthly_usage",
    "training",
    "zdr",
    "limited_regions",
    "featured",
    "open_weights",
    "knowledge",
    "release_date",
    "last_updated",
    "context_limit",
    "output_limit",
    "endpoint",
    "ai_sdk_package",
    "description",
]


def format_unified_csv(models, plan=DEFAULT_PLAN):
    """Format unified models as CSV (one row per model-plan combination)."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(CSV_HEADERS)
    for model in models:
        show_plans = _show_plans(plan)
        protocol = model.get("protocol") or ""
        pricing = model.get("pricing") or {}
        limits = model.get("limits") or {}
        cached_write = pricing.get("cached_write")
        for current_plan in show_plans:
            writer.writerow(
                [
                    model.get("model_id", ""),
                    model.get("model_name", ""),
                    model.get("canonical_id", ""),
                    model.get("provider", ""),
                    model.get("family", ""),
                    current_plan,
                    protocol,
                    model.get("recommended_api", ""),
                    model.get("supports_schema", ""),
                    pricing.get("input", ""),
                    pricing.get("output", ""),
                    pricing.get("cached_read", ""),
                    json.dumps(cached_write) if isinstance(cached_write, dict) else cached_write,
                    _row_usage(model, current_plan) or "",
                    model.get("training", ""),
                    model.get("zdr", ""),
                    model.get("limited_regions", ""),
                    model.get("featured", ""),
                    model.get("open_weights", ""),
                    model.get("knowledge", ""),
                    model.get("release_date", ""),
                    model.get("last_updated", ""),
                    limits.get("context", ""),
                    limits.get("output", ""),
                    model.get("endpoint", ""),
                    model.get("ai_sdk_package", ""),
                    model.get("description", ""),
                ]
            )
    return buf.getvalue()


def format_unified_pricing(models, plan=DEFAULT_PLAN):
    """Format unified models as a pricing comparison sorted by input cost."""
    priced = []
    show_plans = _show_plans(plan)
    for model in models:
        name = model.get("model_id") or model.get("model_name") or "?"
        for current_plan in show_plans:
            pricing = model.get("pricing") or {}
            if pricing.get("input") is None:
                continue
            label = f"{name} ({current_plan})" if plan == "all" else name
            priced.append(
                (label, model, pricing, _row_usage(model, current_plan))
            )
    priced.sort(key=lambda entry: entry[2]["input"])

    if not priced:
        return "No pricing data found."

    lines = ["Pricing Comparison (per 1M tokens)", "═" * 70]
    for name, model, pricing, usage in priced:
        lines.append(f"\n  {name}")
        lines.append(f"    Input:        {_fmt_price(pricing.get('input'))}")
        lines.append(f"    Output:       {_fmt_price(pricing.get('output'))}")
        lines.append(f"    Cached Read:  {_fmt_price(pricing.get('cached_read'))}")
        lines.append(f"    Cached Write: {_fmt_cached_write(pricing.get('cached_write'))}")
        if usage is not None:
            lines.append(f"    Included use: {_fmt_usage(usage)}/mo")
        if _cost_notes(model):
            lines.append(
                "    Note: context-tiered prices or time-window multipliers "
                "apply — see --format json"
            )
    return "\n".join(lines)
