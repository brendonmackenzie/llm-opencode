"""Scrape contract fixtures per spec §6: the 9-table page, plan panels, tier
variants, promo spans, Free/-/markup cells, unknown headers, zero-match."""

import pytest

from llm_opencode import (
    DocsScrapeError,
    merge_model_data,
    parse_docs_tables,
    parse_html_tables,
)


def _row(*cells):
    return "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"


def _table(headers, rows):
    thead = "<thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead>"
    tbody = "<tbody>" + "".join(rows) + "</tbody>"
    return f"<table>{thead}{tbody}</table>"


def _panel(panel, table_html, hidden=False):
    hidden_attr = " hidden" if hidden else ""
    return (
        f'<div role="tabpanel" id="go-pricing-panel-{panel}"{hidden_attr} '
        f'data-plan-panel="{panel}">{table_html}</div>'
    )


def _tabs(group, panels_html):
    return (
        f'<div class="docs-plan-tabs" data-plan-tabs data-sync-key="{group}"'
        f">{panels_html}</div>"
    )


def _plan_row(model, input_, output, cached_read, cached_write, limit):
    return _row(model, input_, output, cached_read, cached_write, limit)


def _rates_row(model, per_5h, per_week, per_month):
    return _row(model, per_5h, per_week, per_month)


PRICING_HEADERS = [
    "Model", "Input", "Output", "Cached Read", "Cached Write", "Monthly limit",
]
RATES_HEADERS = [
    "Model", "Requests per 5 hours", "Requests per week", "Requests per month",
]
ENDPOINT_HEADERS = ["Model", "Model ID", "Endpoint", "AI SDK Package"]
POLICY_HEADERS = ["Model", "Model training", "Data retention"]

PRICING_GO_ROWS = [
    _row("GLM-5.3-Flash", "$0.15", "$0.50", "$0.03", "-", "<strong>$60</strong>"),
    _row(
        "LongCat 2.5 Preview Free",
        "Free",
        "Free",
        "Free",
        "-",
        "<strong>Unlimited</strong><br/><small>limited time</small>",
    ),
    _row("MiniMax M3", "$0.30", "$1.20", "$0.06", "-", "<strong>$60</strong>"),
    _row(
        "Qwen3.7 Plus (≤ 256K tokens)",
        "$0.40", "$1.60", "$0.04", "$0.50", "<strong>$60</strong>",
    ),
    _row(
        "Qwen3.7 Plus (&gt; 256K tokens)",
        "$1.20", "$4.80", "$0.12", "$1.50", "<strong>$60</strong>",
    ),
    _row(
        "DeepSeek V4.1 Flash (Off-Peak)",
        "$0.15", "$0.60", "$0.003", "-", "<strong>$80</strong>",
    ),
    _row(
        "DeepSeek V4.1 Flash (Peak)",
        "$0.30", "$1.20", "$0.006", "-", "<strong>$80</strong>",
    ),
    _row(
        "Gemini 4 Argon Contributor",
        "$4.00", "$20.00", "$0.20", "-",
        "<del>$30</del> <strong>$60</strong><br/><small>2x · Limited time</small>",
    ),
    _row("Mystery New Model", "$9.99", "$19.99", "-", "-", "<strong>$60</strong>"),
]


def _pricing_go_plus_rows():
    return [
        _row("GLM-5.3-Flash", "$0.15", "$0.50", "$0.03", "-", "<strong>$180</strong>"),
        _row(
            "LongCat 2.5 Preview Free",
            "Free", "Free", "Free", "-",
            "<strong>Unlimited</strong><br/><small>limited time</small>",
        ),
        _row(
            "Qwen3.7 Plus (≤ 256K tokens)",
            "$0.40", "$1.60", "$0.04", "$0.50", "<strong>$180</strong>",
        ),
        _row(
            "Qwen3.7 Plus (&gt; 256K tokens)",
            "$1.20", "$4.80", "$0.12", "$1.50", "<strong>$180</strong>",
        ),
    ]


def _endpoint_table():
    return _table(
        ENDPOINT_HEADERS,
        [
            _row(
                "GLM-5.3-Flash", "glm-5.3-flash",
                "<code>https://opencode.ai/inference/go/v1/chat/completions</code>",
                "<code>@ai-sdk/openai-compatible</code>",
            ),
            _row(
                "LongCat 2.5 Preview Free", "longcat-2.5-preview-free",
                "<code>https://opencode.ai/inference/go/v1/chat/completions</code>",
                "<code>@ai-sdk/openai-compatible</code>",
            ),
            _row(
                "MiniMax M3", "minimax-m3",
                "<code>https://opencode.ai/inference/go/v1/messages</code>",
                "<code>@ai-sdk/anthropic</code>",
            ),
            _row(
                "Qwen3.7 Plus", "qwen3.7-plus",
                "<code>https://opencode.ai/inference/go/v1/messages</code>",
                "<code>@ai-sdk/anthropic</code>",
            ),
            _row(
                "DeepSeek V4.1 Flash", "deepseek-v4.1-flash",
                "<code>https://opencode.ai/inference/go/v1/chat/completions</code>",
                "<code>@ai-sdk/openai-compatible</code>",
            ),
            _row(
                "Gemini 4 Argon Contributor", "gemini-4-argon-contributor",
                "<code>https://opencode.ai/inference/go/v1/models/gemini-4-argon-contributor</code>",
                "<code>@ai-sdk/google</code>",
            ),
        ],
    )


def _policy_table():
    return _table(
        POLICY_HEADERS,
        [
            _row("GLM-5.3-Flash", "Not used", "30 days"),
            _row("Qwen3.7 Plus", "Not used", "30 days"),
            _row("MiniMax M3", "Not used", "30 days"),
        ],
    )


def _rates_go_table():
    return _table(
        RATES_HEADERS,
        [
            _rates_row("GLM-5.3-Flash", "6,320", "15,790", "31,580"),
            _rates_row("LongCat 2.5 Preview Free", "Unlimited", "Unlimited", "Unlimited"),
            _rates_row("Qwen3.7 Plus", "1,100", "2,750", "5,500"),
            _rates_row("MiniMax M3", "3,200", "8,000", "16,000"),
            _rates_row(
                "Gemini 4 Argon Contributor<br/><small>2x · Limited time</small>",
                "<del>220</del><br/><strong>440</strong>",
                "<del>540</del><br/><strong>1,080</strong>",
                "<del>1,060</del><br/><strong>2,120</strong>",
            ),
        ],
    )


def _rates_go_plus_table():
    return _table(
        RATES_HEADERS,
        [_rates_row("GLM-5.3-Flash", "18,960", "47,370", "94,740")],
    )


def full_page_html():
    """Live-shaped 9-table docs page: two PlanTabs groups plus three unmapped
    tables (plans overview, client support x2), endpoint and policy tables."""
    return (
        # 0: plans overview (recognized-but-unmapped)
        _table(
            ["Plan", "Price", "Included usage"],
            [
                _row("<strong>Go</strong>", "<strong>$10/month</strong>", "Lower-cost access"),
                _row("<strong>Go Plus</strong>", "<strong>$40/month</strong>", "Higher usage limits"),
            ],
        )
        # 1-2: client tables (recognized-but-unmapped)
        + _table(["Client", "Session support"], [_row("Hermes", "Sends the header")])
        + _table(["Client", "Status and tracking"], [_row("DeepSeek Harness", "Tracking: filed")])
        # 3-4: pricing PlanTabs (both panels fully in the DOM)
        + _tabs(
            "go-plan-pricing",
            _panel("go", _table(PRICING_HEADERS, PRICING_GO_ROWS))
            + _panel("go-plus", _table(PRICING_HEADERS, _pricing_go_plus_rows()), hidden=True),
        )
        # 5-6: request limits PlanTabs
        + _tabs(
            "go-plan-rates",
            _panel("go", _rates_go_table())
            + _panel("go-plus", _rates_go_plus_table(), hidden=True),
        )
        # 7: endpoint / model-ID table
        + _endpoint_table()
        # 8: privacy (policies) table
        + _policy_table()
    )


def _docs_data(html=None):
    tables = parse_html_tables(full_page_html() if html is None else html)
    return parse_docs_tables(tables)


# ─── plan panels: both plans kept, no silent overwrite ──────────────────────

def test_plan_keyed_pricing_from_both_panels():
    docs_data, warnings = _docs_data()

    assert docs_data["glm-5.3-flash"]["pricing"] == {
        "go": {
            "input": 0.15, "output": 0.5, "cached_read": 0.03, "cached_write": None,
            "monthly_limit": 60.0, "limit_note": None, "currency": "USD",
        },
        "go_plus": {
            "input": 0.15, "output": 0.5, "cached_read": 0.03, "cached_write": None,
            "monthly_limit": 180.0, "limit_note": None, "currency": "USD",
        },
    }
    assert not warnings


def test_plan_keyed_rate_limits_from_both_panels():
    docs_data, _ = _docs_data()
    assert docs_data["glm-5.3-flash"]["rate_limits"] == {
        "go": {"per_5_hours": "6320", "per_week": "15790", "per_month": "31580"},
        "go_plus": {"per_5_hours": "18960", "per_week": "47370", "per_month": "94740"},
    }


def test_plan_fallback_uses_table_order_without_panels():
    html = (
        "<table><thead><tr><th>Model</th><th>Input</th><th>Output</th>"
        "<th>Cached Read</th><th>Cached Write</th><th>Monthly limit</th></tr></thead>"
        "<tbody><tr><td>Grok 4.5</td><td>$2</td><td>$6</td><td>$0.30</td><td>-</td>"
        "<td><strong>$15</strong></td></tr></tbody></table>"
        "<table><thead><tr><th>Model</th><th>Input</th><th>Output</th>"
        "<th>Cached Read</th><th>Cached Write</th><th>Monthly limit</th></tr></thead>"
        "<tbody><tr><td>Grok 4.5</td><td>$2</td><td>$6</td><td>$0.30</td><td>-</td>"
        "<td><strong>$45</strong></td></tr></tbody></table>"
    )
    docs_data, warnings = _docs_data(html)
    pricing = docs_data["Grok 4.5"]["pricing"]
    assert pricing["go"]["monthly_limit"] == 15.0
    assert pricing["go_plus"]["monthly_limit"] == 45.0
    assert warnings == []


# ─── Free / dash / markup cells ──────────────────────────────────────────────

def test_free_parses_to_zero_dot_zero_with_free_note():
    docs_data, _ = _docs_data()
    pricing = docs_data["longcat-2.5-preview-free"]["pricing"]["go"]
    assert pricing["input"] == 0.0
    assert pricing["output"] == 0.0
    assert pricing["pricing_note"] == "free"


def test_unlimited_monthly_limit_is_none_with_note():
    docs_data, _ = _docs_data()
    pricing = docs_data["longcat-2.5-preview-free"]["pricing"]["go"]
    assert pricing["monthly_limit"] is None
    assert pricing["limit_note"] == "Unlimited limited time"


def test_dash_stays_none():
    docs_data, _ = _docs_data()
    pricing = docs_data["glm-5.3-flash"]["pricing"]["go"]
    assert pricing["cached_write"] is None


def test_monthly_limit_markup_parses_numeric_plus_note():
    docs_data, _ = _docs_data()
    glm = docs_data["glm-5.3-flash"]["pricing"]["go"]
    assert glm["monthly_limit"] == 60.0
    assert glm["limit_note"] is None


def test_superseded_del_value_loses_to_strong_new_value():
    docs_data, _ = _docs_data()
    gemini = docs_data["gemini-4-argon-contributor"]
    assert gemini["pricing"]["go"]["monthly_limit"] == 60.0
    assert gemini["pricing"]["go"]["limit_note"] == "2x · Limited time"
    assert gemini["rate_limits"]["go"] == {
        "per_5_hours": "440", "per_week": "1080", "per_month": "2120",
    }


# ─── promo spans: structured, never concatenated into the name ──────────────

def test_promo_span_captured_and_name_joined_cleanly():
    docs_data, _ = _docs_data()
    gemini = docs_data["gemini-4-argon-contributor"]
    assert gemini["promo"] == "2x · Limited time"
    assert gemini["model_name"] == "Gemini 4 Argon Contributor"
    assert gemini["matched"] is True
    assert gemini["endpoint"] == (
        "https://opencode.ai/inference/go/v1/models/gemini-4-argon-contributor"
    )


# ─── tier variants: distinct entries, no pricing inheritance ────────────────

def test_tier_variants_produce_distinct_slug_keys():
    docs_data, _ = _docs_data()
    assert set(docs_data) >= {
        "qwen3.7-plus#le-256k-tokens",
        "qwen3.7-plus#gt-256k-tokens",
        "deepseek-v4.1-flash#off-peak",
        "deepseek-v4.1-flash#peak",
    }


def test_variant_entries_carry_variant_metadata():
    docs_data, _ = _docs_data()
    variant = docs_data["qwen3.7-plus#le-256k-tokens"]
    assert variant["variant"] == "≤ 256K tokens"
    assert variant["model_name"] == "Qwen3.7 Plus"
    assert variant["matched"] is True


def test_variant_pricing_is_its_own_row_economics():
    docs_data, _ = _docs_data()
    cheap = docs_data["qwen3.7-plus#le-256k-tokens"]["pricing"]
    pricey = docs_data["qwen3.7-plus#gt-256k-tokens"]["pricing"]
    assert cheap["go"]["input"] == 0.40
    assert pricey["go"]["input"] == 1.20
    assert cheap["go_plus"]["monthly_limit"] == 180.0
    assert pricey["go_plus"]["monthly_limit"] == 180.0
    # Peak vs off-peak economics stay separate.
    off_peak = docs_data["deepseek-v4.1-flash#off-peak"]["pricing"]["go"]
    peak = docs_data["deepseek-v4.1-flash#peak"]["pricing"]["go"]
    assert (off_peak["input"], peak["input"]) == (0.15, 0.30)


def test_variants_never_inherit_pricing_and_base_is_unmasked():
    docs_data, _ = _docs_data()
    base = docs_data["qwen3.7-plus"]
    # The base display name has no pricing row of its own; variant economics
    # must not be masked onto it.
    assert "pricing" not in base
    assert base["matched"] is True


def test_variants_inherit_non_pricing_base_fields_when_absent():
    docs_data, _ = _docs_data()
    variant = docs_data["qwen3.7-plus#le-256k-tokens"]
    assert variant["endpoint"] == "https://opencode.ai/inference/go/v1/messages"
    assert variant["ai_sdk_package"] == "@ai-sdk/anthropic"
    assert variant["model_training"] == "Not used"
    assert variant["data_retention"] == "30 days"
    assert variant["rate_limits"]["go"]["per_month"] == "5500"


def test_variant_with_unjoined_base_is_orphan():
    html = (
        "<table><thead><tr><th>Model</th><th>Input</th><th>Output</th>"
        "<th>Cached Read</th><th>Cached Write</th><th>Monthly limit</th></tr></thead>"
        "<tbody>"
        "<tr><td>Foo (Peak)</td><td>$1</td><td>$2</td><td>-</td><td>-</td>"
        "<td><strong>$3</strong></td></tr>"
        "</tbody></table>"
    )
    docs_data, warnings = _docs_data(html)
    variant = docs_data["Foo (Peak)#peak"]
    assert variant["matched"] is False
    assert variant["variant"] == "Peak"
    assert variant["pricing"]["go"]["input"] == 1.0


# ─── orphan docs rows ────────────────────────────────────────────────────────

def test_orphan_docs_row_kept_flagged_unmatched():
    docs_data, _ = _docs_data()
    orphan = docs_data["Mystery New Model"]
    assert orphan["matched"] is False
    assert orphan["pricing"]["go"]["input"] == 9.99
    assert orphan["model_name"] == "Mystery New Model"


def test_matched_rows_join_by_display_name_across_hyphenation():
    html = (
        "<table><thead><tr><th>Model</th><th>Input</th><th>Output</th>"
        "<th>Cached Read</th><th>Cached Write</th><th>Monthly limit</th></tr></thead>"
        "<tbody><tr><td>MiMo V2.5</td><td>$0.14</td><td>$0.28</td><td>$0.0028</td><td>-</td>"
        "<td><strong>$60</strong></td></tr></tbody></table>"
        "<table><thead><tr><th>Model</th><th>Model ID</th><th>Endpoint</th>"
        "<th>AI SDK Package</th></tr></thead><tbody><tr><td>MiMo-V2.5</td>"
        "<td>mimo-v2.5</td><td>https://x/v1/chat/completions</td>"
        "<td>@ai-sdk/openai-compatible</td></tr></tbody></table>"
    )
    docs_data, _ = _docs_data(html)
    assert docs_data["mimo-v2.5"]["matched"] is True


# ─── merge: API ids vs docs rows ────────────────────────────────────────────

def test_merge_model_data_matches_flags():
    api_models = [
        {"id": "glm-5.3-flash", "object": "model", "created": 123, "owned_by": "opencode"},
        {"id": "omen-alpha", "object": "model", "created": 124, "owned_by": "opencode"},
    ]
    tables = parse_html_tables(full_page_html())
    docs_data, _ = parse_docs_tables(tables)
    unified = merge_model_data(api_models, docs_data)

    glm = next(m for m in unified if m["model_id"] == "glm-5.3-flash")
    assert glm["matched"] is True
    assert glm["provider"] == "opencode"
    assert glm["pricing"]["go"]["monthly_limit"] == 60.0

    # API model with no docs row: kept, flagged unmatched, no fabricated data.
    omen = next(m for m in unified if m["model_id"] == "omen-alpha")
    assert omen["matched"] is False
    assert omen["pricing"] == {}
    assert omen["rate_limits"] == {}

    # Docs-only entries (variants, orphans) also land in the output.
    keys = {m["model_id"] for m in unified}
    assert "qwen3.7-plus#le-256k-tokens" in keys
    assert "Mystery New Model" in keys


# ─── header-signature contract: warnings and hard failure ───────────────────

def test_header_drift_warns_without_mapping():
    html = (
        "<table><thead><tr><th>Model</th><th>Input</th><th>Output</th>"
        "<th>Cached Read</th><th>Monthly limit</th></tr></thead>"
        "<tbody><tr><td>Grok 4.5</td><td>$2</td><td>$6</td><td>$0.30</td>"
        "<td><strong>$15</strong></td></tr></tbody></table>"
        "<table><thead><tr><th>Model</th><th>Model ID</th><th>Endpoint</th>"
        "<th>AI SDK Package</th></tr></thead><tbody><tr><td>Grok</td><td>grok</td>"
        "<td>https://x</td><td>@ai-sdk/openai</td></tr></tbody></table>"
    )
    docs_data, warnings = _docs_data(html)
    assert set(docs_data) == {"grok"}
    assert len(warnings) == 1
    assert "header drift" in warnings[0]
    assert "pricing" in warnings[0]
    assert "missing" in warnings[0]


def test_extra_column_warns():
    html = (
        "<table><thead><tr><th>Model</th><th>Input</th><th>Output</th>"
        "<th>Cached Read</th><th>Cached Write</th><th>Monthly limit</th>"
        "<th>Fallbacks</th></tr></thead>"
        "<tbody><tr><td>Grok 4.5</td><td>$2</td><td>$6</td><td>$0.30</td><td>-</td>"
        "<td><strong>$15</strong></td><td>openai</td></tr></tbody></table>"
    )
    docs_data, warnings = _docs_data(html)
    assert docs_data["Grok 4.5"]["pricing"]["go"]["input"] == 2.0
    assert len(warnings) == 1
    assert "extra column" in warnings[0]
    assert "fallbacks" in warnings[0]


def test_unknown_header_shape_warns():
    html = (
        "<table><thead><tr><th>Topic</th><th>Warranty</th></tr></thead>"
        "<tbody><tr><td>Support</td><td>None</td></tr></tbody></table>"
        "<table><thead><tr><th>Model</th><th>Model ID</th><th>Endpoint</th>"
        "<th>AI SDK Package</th></tr></thead><tbody><tr><td>Grok</td><td>grok</td>"
        "<td>https://x</td><td>@ai-sdk/openai</td></tr></tbody></table>"
    )
    docs_data, warnings = _docs_data(html)
    assert "unrecognized table shape" in warnings[0]
    assert docs_data["grok"]["matched"] is True


def test_recognized_but_unmapped_tables_are_silent():
    docs_data, warnings = _docs_data()
    assert warnings == []


def test_zero_matching_tables_fails_loudly():
    html = (
        "<table><thead><tr><th>Plan</th><th>Price</th><th>Included usage</th></tr></thead>"
        "<tbody><tr><td>Go</td><td>$10/month</td><td>stuff</td></tr></tbody></table>"
    )
    tables = parse_html_tables(html)
    with pytest.raises(DocsScrapeError):
        parse_docs_tables(tables)


def test_no_tables_at_all_fails_loudly():
    with pytest.raises(DocsScrapeError):
        parse_docs_tables([])


# ─── malformed cells and rows degrade to facts, never guesses ───────────────

def test_malformed_cells_and_short_rows_degrade():
    html = (
        "<table><thead><tr><th>Model</th><th>Model ID</th><th>Endpoint</th>"
        "<th>AI SDK Package</th></tr></thead><tbody>"
        "<tr><td>MiMo-V2.5 (Peak)</td><td>mimo-v2.5</td><td></td>"
        "<td>@ai-sdk/openai-compatible</td></tr>"
        "<tr><td></td><td></td><td></td><td></td></tr>"
        "</tbody></table>"
        "<table><thead><tr><th>Model</th><th>Input</th><th>Output</th>"
        "<th>Cached Read</th><th>Cached Write</th><th>Monthly limit</th></tr></thead>"
        "<tbody>"
        "<tr><td>MiMo V2.5 (Peak)</td><td>$abc</td><td></td><td>Free</td><td>-</td>"
        "<td></td></tr>"
        "</tbody></table>"
        "<table><thead><tr><th>Model</th><th>Requests per 5 hours</th>"
        "<th>Requests per week</th><th>Requests per month</th></tr></thead>"
        "<tbody><tr><td>MiMo V2.5</td></tr></tbody></table>"
        "<table><thead><tr><th>Model</th><th>Model training</th>"
        "<th>Data retention</th></tr></thead><tbody>"
        "<tr><td>MiMo V2.5 (Peak)</td><td>Not used</td><td>30 days</td></tr>"
        "</tbody></table>"
    )
    docs_data, warnings = _docs_data(html)

    base = docs_data["mimo-v2.5"]
    assert base["endpoint"] == ""  # empty cell joins to nothing
    assert "pricing" not in base  # the only pricing row was the Peak variant's
    assert base["rate_limits"]["go"] == {"per_5_hours": "", "per_week": "", "per_month": ""}

    variant = docs_data["mimo-v2.5#peak"]
    assert variant["variant"] == "Peak"
    assert variant["pricing"]["go"] == {
        "input": None, "output": None, "cached_read": 0.0, "cached_write": None,
        "monthly_limit": None, "limit_note": None, "currency": "USD",
        "pricing_note": "free",
    }
    assert variant["model_training"] == "Not used"
    assert variant["data_retention"] == "30 days"
    assert variant["rate_limits"] == base["rate_limits"]


def test_variant_label_parsing_edges():
    from llm_opencode.docs_scraper import _slug, _split_variant

    assert _split_variant("(≤ 100K)") == ("(≤ 100K)", None)
    assert _split_variant("GPT 5.6 Luna (> 272K tokens)") == (
        "GPT 5.6 Luna", "> 272K tokens",
    )
    assert _slug("·") == "variant"
    assert _slug("≤ 256K tokens") == "le-256k-tokens"
    assert _slug("Off-Peak") == "off-peak"
