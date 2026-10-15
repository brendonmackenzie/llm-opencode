import csv
import io
import json

import click
import llm

from .docs_scraper import DOCS_PAGE_URL, DocsScrapeError
from .routing import analyze_routes, corroboration_warnings, unregistered_routes
from .unified import MODELS_URL

DEFAULT_PLAN = "go"


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
        from llm_opencode import get_opencode_models

        all_models = get_opencode_models()
        if json_:
            click.echo(json.dumps(all_models, indent=2))
        else:
            for model in all_models:
                click.echo(f"- id: {model['id']}")
                click.echo("  protocol: unknown (no scrape data)")
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
        from llm_opencode import fetch_unified_models

        try:
            snapshot = fetch_unified_models()
        except DocsScrapeError:
            click.echo(
                "OpenCode Go page scraping failed. Please click here for more details:"
            )
            click.echo(DOCS_PAGE_URL)
            raise click.exceptions.Exit(1)

        unified = snapshot.get("models", [])
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
        }
        if format_ == "json":
            unified = sort_models(unified, sort_by)
            routes = analyze_routes(unified)
            result = format_unified_json(
                routes,
                warnings=[
                    *snapshot.get("warnings", []),
                    *corroboration_warnings(routes),
                ],
                scraped_at=snapshot.get("scraped_at"),
                unregistered=unregistered_routes(routes),
            )
        else:
            if format_ != "pricing":
                unified = sort_models(unified, sort_by)
            result = formatters[format_](unified)

        if output:
            with open(output, "w", encoding="utf-8") as file:
                file.write(result)
        else:
            click.echo(result)


def _plan_pricing(model, plan=DEFAULT_PLAN):
    pricing = model.get("pricing", {})
    if isinstance(pricing, dict):
        block = pricing.get(plan) or next(iter(pricing.values()), None)
        if isinstance(block, dict):
            return block
    return {}


def _plan_rate_limits(model, plan=DEFAULT_PLAN):
    rate_limits = model.get("rate_limits", {})
    if isinstance(rate_limits, dict):
        block = rate_limits.get(plan) or next(iter(rate_limits.values()), None)
        if isinstance(block, dict):
            return block
    return {}


def _parse_rate_limit(value):
    """Parse a rate limit like '1,080' into an int, or 0 when missing/invalid."""
    if not value:
        return 0
    try:
        return int(str(value).replace(",", ""))
    except (ValueError, TypeError):
        return 0


def _requests_per_month_key(model):
    """Sort key: highest requests per month first, then cheapest input."""
    per_month = _parse_rate_limit(_plan_rate_limits(model).get("per_month"))
    input_price = _plan_pricing(model).get("input")
    input_sort = input_price if input_price is not None else float("inf")
    if per_month > 0:
        return (False, -per_month, input_sort)
    return (True, 0, input_sort)


def _price_key(model, field):
    """Sort key for a pricing field: ascending, missing values pinned to the end."""
    value = _plan_pricing(model).get(field)
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
        "usage": lambda m: _price_key(m, "monthly_limit"),
        "name": _name_key,
        "model_id": _model_id_key,
    }
    return sorted(models, key=keys[sort_by])


def format_unified_json(
    models, warnings=None, scraped_at=None, unregistered=None
):
    """Format unified models as JSON with metadata and an unregistered section."""
    meta = {
        "sources": [MODELS_URL, DOCS_PAGE_URL],
        "scraped_at": scraped_at,
        "warnings": warnings or [],
    }
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
    return f"${value:.2f}" if value is not None else "—"


def _fmt_requests_per_month(value):
    return f"{value:,}" if value > 0 else "—"


def format_unified_table(models):
    """Format unified models as a pretty pricing table."""
    if not models:
        return "No models found."

    rows = []
    for model in models:
        pricing = _plan_pricing(model)
        per_month = _parse_rate_limit(_plan_rate_limits(model).get("per_month"))
        rows.append(
            [
                model.get("model_id", "")[:25],
                model.get("model_name", "")[:20],
                _fmt_requests_per_month(per_month),
                _fmt_price(pricing.get("input")),
                _fmt_price(pricing.get("output")),
                _fmt_price(pricing.get("cached_read")),
                _fmt_price(pricing.get("cached_write")),
                _fmt_price(pricing.get("monthly_limit")),
                model.get("data_retention", "")[:10] or "—",
            ]
        )

    headers = [
        "Model ID", "Name", "Req/Month", "Input", "Output",
        "Cache R", "Cache W", "Limit", "Retention",
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
            "matched",
            "input_price",
            "output_price",
            "cached_read",
            "cached_write",
            "monthly_limit",
            "limit_note",
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
        pricing = _plan_pricing(model)
        rate_limits = _plan_rate_limits(model)
        writer.writerow(
            [
                model.get("model_id", ""),
                model.get("model_name", ""),
                model.get("provider", ""),
                model.get("matched", ""),
                pricing.get("input", ""),
                pricing.get("output", ""),
                pricing.get("cached_read", ""),
                pricing.get("cached_write", ""),
                pricing.get("monthly_limit", ""),
                pricing.get("limit_note", ""),
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
    priced = [m for m in models if _plan_pricing(m).get("input") is not None]
    priced.sort(key=lambda m: _plan_pricing(m)["input"])

    if not priced:
        return "No pricing data found."

    lines = ["Pricing Comparison (per 1M tokens)", "═" * 70]
    for model in priced:
        pricing = _plan_pricing(model)
        name = model.get("model_id") or model.get("model_name") or "?"
        lines.append(f"\n  {name}")
        note = pricing.get("pricing_note")
        suffix = f" ({note})" if note else ""
        lines.append(f"    Input:        {_fmt_price(pricing.get('input'))}{suffix}")
        lines.append(f"    Output:       {_fmt_price(pricing.get('output'))}")
        lines.append(f"    Cached Read:  {_fmt_price(pricing.get('cached_read'))}")
        lines.append(f"    Cached Write: {_fmt_price(pricing.get('cached_write'))}")
        limit = pricing.get("monthly_limit")
        lines.append(f"    Monthly cap:  {_fmt_price(limit)}")
        lines.append(f"    Retention:    {model.get('data_retention', '—')}")
    return "\n".join(lines)
