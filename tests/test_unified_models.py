import json
import os
from unittest.mock import patch

import pytest

from llm_opencode import (
    DocsScrapeError,
    fetch_docs_page,
    fetch_unified_models,
    merge_model_data,
    parse_docs_tables,
    parse_html_tables,
)


def _docs_tables():
    return [
        {
            "headers": [
                "Model",
                "requests per 5 hour",
                "requests per week",
                "requests per month",
            ],
            "rows": [
                ["Grok 4.5", "120", "300", "600"],
                ["GPT 5.6 Luna", "2,050", "5,100", "10,250"],
            ],
        },
        {
            "headers": ["Model", "Input", "Output", "Cached Read", "Cached Write", "Usage"],
            "rows": [
                ["Grok 4.5", "$2.00", "$6.00", "$0.30", "-", "$15"],
                ["GPT 5.6 Luna (≤ 272K tokens)", "$0.20", "$1.20", "$0.02", "$0.25", "$15"],
            ],
        },
        {
            "headers": ["Model", "Model ID", "Endpoint", "AI SDK Package"],
            "rows": [
                ["Grok 4.5", "grok-4.5", "https://opencode.ai/zen/go/v1/responses", "@ai-sdk/openai"],
                ["GPT 5.6 Luna", "gpt-5.6-luna", "https://opencode.ai/zen/go/v1/responses", "@ai-sdk/openai"],
            ],
        },
        {
            "headers": ["Model", "Model training", "Data retention"],
            "rows": [
                ["Grok 4.5", "Not used", "30 days"],
                ["GPT 5.6 Luna", "Not used", "30 days"],
            ],
        },
    ]


def _docs_html():
    rows = []
    for table in _docs_tables():
        thead = "".join(f"<th>{h}</th>" for h in table["headers"])
        body = ""
        for row in table["rows"]:
            body += "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        rows.append(
            f"<table><thead><tr>{thead}</tr></thead><tbody>{body}</tbody></table>"
        )
    return "".join(rows)


# ─── fetch_docs_page ─────────────────────────────────────────────────────────

def test_fetch_docs_page_returns_text():
    with patch("httpx.get") as mock_get:
        mock_response = mock_get.return_value
        mock_response.text = "<html>docs</html>"
        mock_response.raise_for_status.return_value = None

        result = fetch_docs_page()

    assert result == "<html>docs</html>"
    mock_get.assert_called_once()


def test_fetch_docs_page_http_error_raises_docs_scrape_error():
    import httpx

    with (
        patch("httpx.get", side_effect=httpx.HTTPError("boom")),
        pytest.raises(DocsScrapeError),
    ):
        fetch_docs_page()


# ─── parse_docs_tables ───────────────────────────────────────────────────────

def test_parse_docs_tables_builds_model_data():
    docs_data = parse_docs_tables(_docs_tables())

    assert set(docs_data) == {"grok-4.5", "gpt-5.6-luna", "GPT 5.6 Luna (≤ 272K tokens)"}

    grok = docs_data["grok-4.5"]
    assert grok["model_name"] == "Grok 4.5"
    assert grok["rate_limits"] == {
        "per_5_hours": "120",
        "per_week": "300",
        "per_month": "600",
    }
    assert grok["pricing"] == {
        "input": 2.0,
        "output": 6.0,
        "cached_read": 0.3,
        "cached_write": None,
        "usage": 15.0,
        "currency": "USD",
    }
    assert grok["endpoint"] == "https://opencode.ai/zen/go/v1/responses"
    assert grok["ai_sdk_package"] == "@ai-sdk/openai"
    assert grok["model_training"] == "Not used"
    assert grok["data_retention"] == "30 days"


def test_parse_docs_tables_variant_inherits_base_fields():
    docs_data = parse_docs_tables(_docs_tables())

    variant = docs_data["GPT 5.6 Luna (≤ 272K tokens)"]
    assert variant["pricing"]["input"] == 0.2
    assert variant["endpoint"] == "https://opencode.ai/zen/go/v1/responses"
    assert variant["ai_sdk_package"] == "@ai-sdk/openai"
    assert variant["model_training"] == "Not used"
    assert variant["data_retention"] == "30 days"
    assert variant["rate_limits"] == {
        "per_5_hours": "2050",
        "per_week": "5100",
        "per_month": "10250",
    }


def test_parse_docs_tables_no_tables():
    assert parse_docs_tables([]) == {}


def test_parse_docs_tables_dash_price_parses_to_none():
    tables = [
        {
            "headers": ["Model", "Input", "Output", "Cached Read", "Cached Write", "Usage"],
            "rows": [["Grok 4.5", "-", "$6.00", "$0.30", "-", "$15"]],
        },
        {
            "headers": ["Model", "Model ID", "Endpoint", "AI SDK Package"],
            "rows": [["Grok 4.5", "grok-4.5", "https://x/v1/responses", "@ai-sdk/openai"]],
        },
    ]
    docs_data = parse_docs_tables(tables)
    pricing = docs_data["grok-4.5"]["pricing"]
    assert pricing["input"] is None
    assert pricing["cached_write"] is None
    assert pricing["output"] == 6.0


def test_parse_docs_tables_matches_hyphenated_names():
    tables = [
        {
            "headers": ["Model", "Input", "Output", "Cached Read", "Cached Write", "Usage"],
            "rows": [["MiMo V2.5", "$0.14", "$0.28", "$0.0028", "-", "$60"]],
        },
        {
            "headers": ["Model", "Model ID", "Endpoint", "AI SDK Package"],
            "rows": [["MiMo-V2.5", "mimo-v2.5", "https://x/v1/chat/completion", "@ai-sdk/openai-compatible"]],
        },
    ]
    docs_data = parse_docs_tables(tables)
    assert docs_data["mimo-v2.5"]["pricing"]["input"] == 0.14


def test_parse_docs_tables_through_html_parser():
    docs_data = parse_docs_tables(parse_html_tables(_docs_html()))
    assert "grok-4.5" in docs_data
    assert docs_data["grok-4.5"]["rate_limits"]["per_week"] == "300"


def test_parse_docs_tables_tolerates_malformed_rows():
    tables = [
        {
            "headers": ["Model", "Model ID", "Endpoint", "AI SDK Package"],
            "rows": [["short"]],
        },
        {
            "headers": ["Model", "Input", "Output", "Cached Read", "Cached Write", "Usage"],
            "rows": [[], ["Grok 4.5", "a", "b", "c", "d"]],
        },
        {
            "headers": ["Model", "requests per 5 hour", "requests per week", "requests per month"],
            "rows": [["Grok 4.5", "a", "b"]],
        },
        {
            "headers": ["Model", "Model training", "Data retention"],
            "rows": [["Grok 4.5", "a"]],
        },
        {
            "headers": ["Something", "Else"],
            "rows": [["Grok 4.5", "x"]],
        },
    ]
    assert parse_docs_tables(tables) == {
        "short": {"model_name": "short"},
        "Grok 4.5": {"model_name": "Grok 4.5"},
    }


def test_parse_docs_tables_variant_without_base_in_docs():
    tables = [
        {
            "headers": ["Model", "Model ID", "Endpoint", "AI SDK Package"],
            "rows": [["Bar", "bar", "https://x", "@sdk"]],
        },
        {
            "headers": ["Model", "Input", "Output", "Cached Read", "Cached Write", "Usage"],
            "rows": [["Foo (variant)", "$1", "$2", "-", "-", "$3"]],
        },
    ]
    docs_data = parse_docs_tables(tables)
    variant = docs_data["Foo (variant)"]
    assert "endpoint" not in variant
    assert variant["pricing"]["input"] == 1.0


def test_parse_docs_tables_variant_keeps_own_fields():
    tables = [
        {
            "headers": ["Model", "Model ID", "Endpoint", "AI SDK Package"],
            "rows": [
                ["Foo", "foo", "https://base", "@sdk-base"],
                ["Foo (variant)", "foo-v", "https://variant", "@sdk-v"],
            ],
        },
        {
            "headers": ["Model", "Model training", "Data retention"],
            "rows": [["Foo", "Trained", "90 days"]],
        },
    ]
    docs_data = parse_docs_tables(tables)
    variant = docs_data["foo-v"]
    assert variant["endpoint"] == "https://variant"
    assert variant["ai_sdk_package"] == "@sdk-v"
    assert variant["model_training"] == "Trained"
    assert variant["data_retention"] == "90 days"


def test_parse_docs_tables_unparseable_price_is_none():
    tables = [
        {
            "headers": ["Model", "Input", "Output", "Cached Read", "Cached Write", "Usage"],
            "rows": [["Grok", "$abc", "$2", "$3", "$4", "$5"]],
        },
        {
            "headers": ["Model", "Model ID", "Endpoint", "AI SDK Package"],
            "rows": [["Grok", "grok", "https://x", "@sdk"]],
        },
    ]
    docs_data = parse_docs_tables(tables)
    assert docs_data["grok"]["pricing"]["input"] is None


# ─── merge_model_data ────────────────────────────────────────────────────────

def test_merge_model_data_combines_api_and_docs():
    api_models = [
        {"id": "grok-4.5", "object": "model", "created": 123, "owned_by": "opencode"},
    ]
    docs_data = {
        "grok-4.5": {
            "model_name": "Grok 4.5",
            "rate_limits": {"per_5_hours": "120", "per_week": "300", "per_month": "600"},
            "pricing": {
                "input": 2.0, "output": 6.0, "cached_read": 0.3,
                "cached_write": None, "usage": 15.0, "currency": "USD",
            },
            "endpoint": "https://opencode.ai/zen/go/v1/responses",
            "ai_sdk_package": "@ai-sdk/openai",
            "model_training": "Not used",
            "data_retention": "30 days",
        }
    }

    unified = merge_model_data(api_models, docs_data)

    assert len(unified) == 1
    entry = unified[0]
    assert entry["model_id"] == "grok-4.5"
    assert entry["model_name"] == "Grok 4.5"
    assert entry["provider"] == "opencode"
    assert entry["created"] == 123
    assert entry["endpoint"] == "https://opencode.ai/zen/go/v1/responses"
    assert entry["ai_sdk_package"] == "@ai-sdk/openai"
    assert entry["model_training"] == "Not used"
    assert entry["data_retention"] == "30 days"
    assert entry["pricing"] == {
        "input": 2.0, "output": 6.0, "cached_read": 0.3,
        "cached_write": None, "usage": 15.0, "currency": "USD",
    }
    assert entry["rate_limits"] == {"per_5_hours": "120", "per_week": "300", "per_month": "600"}


def test_merge_model_data_api_model_without_docs_gets_defaults():
    api_models = [{"id": "unknown-model", "object": "model", "created": 1, "owned_by": "opencode"}]

    unified = merge_model_data(api_models, {})

    assert len(unified) == 1
    entry = unified[0]
    assert entry["model_name"] == ""
    assert entry["pricing"] == {
        "input": None, "output": None, "cached_read": None,
        "cached_write": None, "usage": None, "currency": "USD",
    }
    assert entry["rate_limits"] == {"per_5_hours": "", "per_week": "", "per_month": ""}
    assert entry["endpoint"] == ""
    assert entry["ai_sdk_package"] == ""
    assert entry["model_training"] == ""
    assert entry["data_retention"] == ""


def test_merge_model_data_docs_only_models_included():
    api_models = [{"id": "grok-4.5", "object": "model", "created": 123, "owned_by": "opencode"}]
    docs_data = {
        "upcoming-model": {
            "model_name": "Upcoming Model",
            "pricing": {"input": 1.0, "output": 2.0, "cached_read": None,
                        "cached_write": None, "usage": None, "currency": "USD"},
            "rate_limits": {"per_5_hours": "10", "per_week": "20", "per_month": "30"},
        }
    }

    unified = merge_model_data(api_models, docs_data)

    assert [m["model_id"] for m in unified] == ["grok-4.5", "upcoming-model"]
    upcoming = unified[1]
    assert upcoming["provider"] == ""
    assert upcoming["created"] is None
    assert upcoming["pricing"]["input"] == 1.0


# ─── fetch_unified_models ────────────────────────────────────────────────────

def test_fetch_unified_models_uses_fresh_cache(tmp_path):
    cache_data = [{"model_id": "grok-4.5"}]
    cache_file = tmp_path / "opencode_unified_models.json"
    cache_file.write_text(json.dumps(cache_data))

    with (
        patch("llm_opencode.llm.user_dir", return_value=tmp_path),
        patch("llm_opencode.get_opencode_models") as mock_get,
        patch("llm_opencode.fetch_docs_page") as mock_docs,
    ):
        result = fetch_unified_models()

    assert result == cache_data
    mock_get.assert_not_called()
    mock_docs.assert_not_called()


def test_fetch_unified_models_fetches_merges_and_caches(tmp_path):
    cache_file = tmp_path / "opencode_unified_models.json"

    with (
        patch("llm_opencode.llm.user_dir", return_value=tmp_path),
        patch(
            "llm_opencode.get_opencode_models",
            return_value=[{"id": "grok-4.5", "object": "model", "created": 123, "owned_by": "opencode"}],
        ),
        patch("llm_opencode.fetch_docs_page", return_value=_docs_html()),
    ):
        result = fetch_unified_models()

    assert [m["model_id"] for m in result] == [
        "grok-4.5",
        "gpt-5.6-luna",
        "GPT 5.6 Luna (≤ 272K tokens)",
    ]
    grok = result[0]
    assert grok["provider"] == "opencode"
    assert grok["pricing"]["input"] == 2.0
    assert grok["rate_limits"]["per_week"] == "300"

    variant = result[2]
    assert variant["model_id"] == "GPT 5.6 Luna (≤ 272K tokens)"
    assert variant["rate_limits"]["per_month"] == "10250"

    cached = json.loads(cache_file.read_text())
    assert cached == result


def test_fetch_unified_models_stale_cache_refetches(tmp_path):
    cache_file = tmp_path / "opencode_unified_models.json"
    cache_file.write_text(json.dumps([{"model_id": "old"}]))
    os.utime(cache_file, (0, 0))

    with (
        patch("llm_opencode.llm.user_dir", return_value=tmp_path),
        patch(
            "llm_opencode.get_opencode_models",
            return_value=[{"id": "grok-4.5", "object": "model", "created": 123, "owned_by": "opencode"}],
        ),
        patch("llm_opencode.fetch_docs_page", return_value=_docs_html()),
    ):
        result = fetch_unified_models()

    assert [m["model_id"] for m in result] == [
        "grok-4.5",
        "gpt-5.6-luna",
        "GPT 5.6 Luna (≤ 272K tokens)",
    ]
    assert json.loads(cache_file.read_text()) == result


def test_fetch_unified_models_fresh_corrupt_cache_refetches(tmp_path):
    cache_file = tmp_path / "opencode_unified_models.json"
    cache_file.write_text("not valid json")

    with (
        patch("llm_opencode.llm.user_dir", return_value=tmp_path),
        patch(
            "llm_opencode.get_opencode_models",
            return_value=[{"id": "grok-4.5", "object": "model", "created": 123, "owned_by": "opencode"}],
        ),
        patch("llm_opencode.fetch_docs_page", return_value=_docs_html()),
    ):
        result = fetch_unified_models()

    assert result[0]["model_id"] == "grok-4.5"
    assert json.loads(cache_file.read_text()) == result


def test_fetch_unified_models_docs_failure_raises(tmp_path):
    with (
        patch("llm_opencode.llm.user_dir", return_value=tmp_path),
        patch("llm_opencode.get_opencode_models"),
        patch("llm_opencode.fetch_docs_page", side_effect=DocsScrapeError("boom")),
        pytest.raises(DocsScrapeError),
    ):
        fetch_unified_models()


def test_fetch_unified_models_no_tables_raises(tmp_path):
    with (
        patch("llm_opencode.llm.user_dir", return_value=tmp_path),
        patch("llm_opencode.get_opencode_models"),
        patch("llm_opencode.fetch_docs_page", return_value="<html><body>changed</body></html>"),
        pytest.raises(DocsScrapeError),
    ):
        fetch_unified_models()
