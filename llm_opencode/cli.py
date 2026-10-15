import csv
import io
import json

import click
import llm

from .docs_scraper import DOCS_PAGE_URL, DocsScrapeError
from .routing import analyze_routes, corroboration_warnings, unregistered_routes
from .unified import MODELS_URL

DEFAULT_PLAN = "go"
PLANS = ("go", "go_plus")
PROTOCOL_UNKNOWN = "unknown"


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
        "--plan",
        type=click.Choice(["go", "go_plus", "all"]),
        default=DEFAULT_PLAN,
        help=(
            "Plan to show pricing and rate limits for "
            "(JSON always carries both plans; the filter only changes their order)"
        ),
    )
    @click.option(
        "--sort-by",
        type=click.Choice(
            ["requests", "input", "output", "monthly-limit", "name", "model_id"]
        ),
        default="requests",
        help="Sort order (pricing format always sorts by input price)",
    )
    @click.option("-o", "--output", type=click.Path(), help="Write output to file")
    def detail(format_, sort_by, search, model_id, plan, output):
        "Show detailed model info including pricing, rate limits, and protocols"
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
        if search:
            query = search.lower()
            unified = [
                m
                for m in unified
                if query in m.get("model_id", "").lower()
                or query in m.get("model_name", "").lower()
            ]

        routes = analyze_routes(unified)
        formatters = {
            "table": format_unified_table,
            "csv": format_unified_csv,
            "pricing": format_unified_pricing,
        }
        if format_ == "json":
            routes = sort_models(routes, sort_by, plan=plan)
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
                routes = sort_models(routes, sort_by, plan=plan)
            result = formatters[format_](routes, plan=plan)

        if output:
            with open(output, "w", encoding="utf-8") as file:
                file.write(result)
        else:
            click.echo(result)

    # Alias: `llm opencode models detail` resolves to the same command.
    models.add_command(detail)


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


def _plan_block(container, plan):
    """A strict plan block lookup: no first-value fallback, {} when absent."""
    if isinstance(container, dict):
        block = container.get(plan)
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


def _requests_per_month_key(model, plan=DEFAULT_PLAN):
    """Sort key: highest requests per month first, then cheapest input."""
    per_month = _parse_rate_limit(_plan_rate_limits(model, plan).get("per_month"))
    input_price = _plan_pricing(model, plan).get("input")
    input_sort = input_price if input_price is not None else float("inf")
    if per_month > 0:
        return (False, -per_month, input_sort)
    return (True, 0, input_sort)


def _price_key(model, field, plan=DEFAULT_PLAN):
    """Sort key for a pricing field: ascending, missing values pinned to the end."""
    value = _plan_pricing(model, plan).get(field)
    if value is not None:
        return (False, value)
    return (True, 0.0)


def _name_key(model):
    name = model.get("model_name") or ""
    return (not name, name.lower())


def _model_id_key(model):
    model_id = model.get("model_id") or ""
    return (not model_id, model_id.lower())


def sort_models(models, sort_by="requests", plan=DEFAULT_PLAN):
    """Sort unified models by the given key (default: requests per month, highest first)."""
    keys = {
        "requests": lambda m: _requests_per_month_key(m, plan),
        "input": lambda m: _price_key(m, "input", plan),
        "output": lambda m: _price_key(m, "output", plan),
        "monthly-limit": lambda m: _price_key(m, "monthly_limit", plan),
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


def _plan_selection(model, plan, current_plan):
    """Pricing/rate-limit blocks for a row, honoring the ``--plan`` filter.

    In ``all`` mode the plan blocks are looked up strictly — a missing plan
    renders dashes instead of silently borrowing the other plan's numbers.
    """
    if plan == "all":
        return _plan_block(model.get("pricing"), current_plan), _plan_block(
            model.get("rate_limits"), current_plan
        )
    return _plan_pricing(model, plan), _plan_rate_limits(model, plan)


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

    show_plans = PLANS if plan == "all" else (plan,)
    headers = [
        "Model ID", "Name", "Protocol", "Req/Month", "Input", "Output",
        "Cache R", "Cache W", "Monthly Cap", "Retention",
    ]
    if plan == "all":
        headers.insert(0, "Plan")
    rows = []
    for model in models:
        protocol = model.get("protocol") or PROTOCOL_UNKNOWN
        for current_plan in show_plans:
            pricing, rate_limits = _plan_selection(model, plan, current_plan)
            per_month = _parse_rate_limit(rate_limits.get("per_month"))
            row = []
            if plan == "all":
                row.append(current_plan)
            row.extend(
                [
                    model.get("model_id", "")[:25],
                    model.get("model_name", "")[:20],
                    protocol,
                    _fmt_requests_per_month(per_month),
                    _fmt_price(pricing.get("input")),
                    _fmt_price(pricing.get("output")),
                    _fmt_price(pricing.get("cached_read")),
                    _fmt_price(pricing.get("cached_write")),
                    _fmt_price(pricing.get("monthly_limit")),
                    model.get("data_retention", "")[:10] or "—",
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
    "plan",
    "protocol",
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


def format_unified_csv(models, plan=DEFAULT_PLAN):
    """Format unified models as CSV (one row per model-plan combination)."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(CSV_HEADERS)
    for model in models:
        show_plans = PLANS if plan == "all" else (plan,)
        protocol = model.get("protocol") or ""
        for current_plan in show_plans:
            pricing, rate_limits = _plan_selection(model, plan, current_plan)
            writer.writerow(
                [
                    model.get("model_id", ""),
                    model.get("model_name", ""),
                    current_plan,
                    protocol,
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


def format_unified_pricing(models, plan=DEFAULT_PLAN):
    """Format unified models as a pricing comparison sorted by input cost."""
    priced = []
    show_plans = PLANS if plan == "all" else (plan,)
    for model in models:
        name = model.get("model_id") or model.get("model_name") or "?"
        for current_plan in show_plans:
            if plan == "all":
                pricing = _plan_block(model.get("pricing"), current_plan)
            else:
                pricing = _plan_pricing(model, current_plan)
            if pricing.get("input") is None:
                continue
            label = f"{name} ({current_plan})" if plan == "all" else name
            priced.append((label, model, pricing))
    priced.sort(key=lambda entry: entry[2]["input"])

    if not priced:
        return "No pricing data found."

    lines = ["Pricing Comparison (per 1M tokens)", "═" * 70]
    for name, model, pricing in priced:
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
