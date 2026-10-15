import json
import os
from unittest.mock import patch

import pytest

from llm_opencode import (
    DocsScrapeError,
    fetch_docs_page,
    fetch_unified_models,
    merge_model_data,
)


# ─── fetch_docs_page ─────────────────────────────────────────────────────────

def test_fetch_docs_page_returns_text():
    with patch("httpx2.get") as mock_get:
        mock_response = mock_get.return_value
        mock_response.text = "<html>docs</html>"
        mock_response.raise_for_status.return_value = None

        result = fetch_docs_page()

    assert result == "<html>docs</html>"
    mock_get.assert_called_once()


def test_fetch_docs_page_http_error_raises_docs_scrape_error():
    import httpx2

    with (
        patch("httpx2.get", side_effect=httpx2.HTTPError("boom")),
        pytest.raises(DocsScrapeError),
    ):
        fetch_docs_page()


# ─── merge_model_data ────────────────────────────────────────────────────────

def _docs_entry():
    return {
        "model_id": "grok-4.5",
        "model_name": "Grok 4.5",
        "matched": True,
        "variant": "≤ 256K tokens",
        "promo": "2x · Limited time",
        "endpoint": "https://opencode.ai/inference/go/v1/responses",
        "ai_sdk_package": "@ai-sdk/openai",
        "model_training": "Not used",
        "data_retention": "30 days",
        "pricing": {
            "go": {"monthly_limit": 15.0},
            "go_plus": {"monthly_limit": 45.0},
        },
        "rate_limits": {
            "go": {"per_month": "600"},
            "go_plus": {"per_month": "3600"},
        },
    }


def test_merge_model_data_combines_api_and_docs():
    api_models = [
        {"id": "grok-4.5", "object": "model", "created": 123, "owned_by": "opencode"},
    ]

    unified = merge_model_data(api_models, {"grok-4.5": _docs_entry()})

    assert len(unified) == 1
    entry = unified[0]
    assert entry["model_id"] == "grok-4.5"
    assert entry["model_name"] == "Grok 4.5"
    assert entry["provider"] == "opencode"
    assert entry["created"] == 123
    assert entry["matched"] is True
    assert entry["variant"] == "≤ 256K tokens"
    assert entry["promo"] == "2x · Limited time"
    assert entry["endpoint"] == "https://opencode.ai/inference/go/v1/responses"
    assert entry["ai_sdk_package"] == "@ai-sdk/openai"
    assert entry["model_training"] == "Not used"
    assert entry["data_retention"] == "30 days"
    assert entry["pricing"]["go"]["monthly_limit"] == 15.0
    assert entry["pricing"]["go_plus"]["monthly_limit"] == 45.0
    assert entry["rate_limits"]["go"]["per_month"] == "600"


def test_merge_model_data_plain_entry_omits_variant_promo_keys():
    api_models = [{"id": "grok-4.5"}]
    docs_entry = _docs_entry()
    del docs_entry["variant"], docs_entry["promo"]

    unified = merge_model_data(api_models, {"grok-4.5": docs_entry})

    entry = unified[0]
    assert "variant" not in entry
    assert "promo" not in entry


def test_merge_model_data_api_model_without_docs_gets_defaults():
    api_models = [{"id": "unknown-model", "object": "model", "created": 1, "owned_by": "opencode"}]

    unified = merge_model_data(api_models, {})

    assert len(unified) == 1
    entry = unified[0]
    assert entry["model_name"] == ""
    assert entry["matched"] is False
    assert entry["pricing"] == {}
    assert entry["rate_limits"] == {}
    assert entry["endpoint"] == ""
    assert entry["ai_sdk_package"] == ""
    assert entry["model_training"] == ""
    assert entry["data_retention"] == ""


def test_merge_model_data_docs_only_models_included():
    api_models = [{"id": "grok-4.5", "object": "model", "created": 123, "owned_by": "opencode"}]
    docs_data = {
        "upcoming-model": {
            "model_id": "upcoming-model",
            "model_name": "Upcoming Model",
            "matched": True,
            "pricing": {"go": {"input": 1.0}},
            "rate_limits": {"go": {"per_month": "30"}},
        }
    }

    unified = merge_model_data(api_models, docs_data)

    assert [m["model_id"] for m in unified] == ["grok-4.5", "upcoming-model"]
    upcoming = unified[1]
    assert upcoming["provider"] == ""
    assert upcoming["created"] is None
    assert upcoming["pricing"]["go"]["input"] == 1.0


# ─── fetch_unified_models (snapshot payload) ────────────────────────────────

DOCS_HTML = (
    "<table><thead><tr><th>Model</th><th>Input</th><th>Output</th>"
    "<th>Cached Read</th><th>Cached Write</th><th>Monthly limit</th></tr></thead><tbody>"
    "<tr><td>Grok 4.5</td><td>$2.00</td><td>$6.00</td><td>$0.30</td><td>-</td>"
    "<td><strong>$15</strong></td></tr>"
    "<tr><td>Mystery New Model</td><td>$9.99</td><td>$19.99</td><td>-</td><td>-</td>"
    "<td><strong>$60</strong></td></tr></tbody></table>"
    "<table><thead><tr><th>Model</th><th>Model ID</th><th>Endpoint</th>"
    "<th>AI SDK Package</th></tr></thead><tbody>"
    "<tr><td>Grok 4.5</td><td>grok-4.5</td><td><code>https://x/v1/responses</code></td>"
    "<td><code>@ai-sdk/openai</code></td></tr></tbody></table>"
)


def _api_models():
    return [
        {"id": "grok-4.5", "object": "model", "created": 123, "owned_by": "opencode"},
        {"id": "omen-alpha", "object": "model", "created": 9, "owned_by": "opencode"},
    ]


def test_fetch_unified_models_uses_fresh_snapshot_cache(tmp_path):
    snapshot = {"scraped_at": 42, "warnings": [], "models": [{"model_id": "grok-4.5"}]}
    cache_file = tmp_path / "opencode_unified_models.json"
    cache_file.write_text(json.dumps(snapshot))

    with (
        patch("llm_opencode.llm.user_dir", return_value=tmp_path),
        patch("llm_opencode.get_opencode_models") as mock_get,
        patch("llm_opencode.fetch_docs_page") as mock_docs,
    ):
        result = fetch_unified_models()

    assert result == snapshot
    mock_get.assert_not_called()
    mock_docs.assert_not_called()


def test_fetch_unified_models_fetches_merges_and_caches_snapshot(tmp_path):
    cache_file = tmp_path / "opencode_unified_models.json"

    with (
        patch("llm_opencode.llm.user_dir", return_value=tmp_path),
        patch("llm_opencode.get_opencode_models", return_value=_api_models()),
        patch("llm_opencode.fetch_docs_page", return_value=DOCS_HTML),
    ):
        result = fetch_unified_models()

    assert set(result) == {"scraped_at", "warnings", "models"}
    assert isinstance(result["scraped_at"], int)
    assert result["warnings"] == []
    assert [m["model_id"] for m in result["models"]] == [
        "grok-4.5",
        "omen-alpha",
        "Mystery New Model",
    ]
    grok = result["models"][0]
    assert grok["matched"] is True
    assert grok["pricing"]["go"]["input"] == 2.0
    assert grok["pricing"]["go"]["monthly_limit"] == 15.0
    omen = result["models"][1]
    assert omen["matched"] is False
    assert omen["pricing"] == {}
    orphan = result["models"][2]
    assert orphan["matched"] is False
    # Go Plus values do not exist without a Go Plus panel: no silent defaults.
    assert "go_plus" not in grok["pricing"]

    cached = json.loads(cache_file.read_text())
    assert cached == result


def test_fetch_unified_models_caches_scrape_warnings(tmp_path):
    warning_html = (
        "<table><thead><tr><th>Topic</th><th>Warranty</th></tr></thead>"
        "<tbody><tr><td>Support</td><td>None</td></tr></tbody></table>"
        + DOCS_HTML
    )
    with (
        patch("llm_opencode.llm.user_dir", return_value=tmp_path),
        patch("llm_opencode.get_opencode_models", return_value=_api_models()),
        patch("llm_opencode.fetch_docs_page", return_value=warning_html),
    ):
        result = fetch_unified_models()

    assert any("unrecognized table shape" in w for w in result["warnings"])
    assert json.loads((tmp_path / "opencode_unified_models.json").read_text()) == result


def test_fetch_unified_models_stale_cache_refetches(tmp_path):
    cache_file = tmp_path / "opencode_unified_models.json"
    cache_file.write_text(json.dumps({"scraped_at": 1, "warnings": [], "models": [{"model_id": "old"}]}))
    os.utime(cache_file, (0, 0))

    with (
        patch("llm_opencode.llm.user_dir", return_value=tmp_path),
        patch("llm_opencode.get_opencode_models", return_value=_api_models()),
        patch("llm_opencode.fetch_docs_page", return_value=DOCS_HTML),
    ):
        result = fetch_unified_models()

    assert result["models"][0]["model_id"] == "grok-4.5"
    assert json.loads(cache_file.read_text()) == result


def test_fetch_unified_models_fresh_corrupt_cache_refetches(tmp_path):
    cache_file = tmp_path / "opencode_unified_models.json"
    cache_file.write_text("not valid json")

    with (
        patch("llm_opencode.llm.user_dir", return_value=tmp_path),
        patch("llm_opencode.get_opencode_models", return_value=_api_models()),
        patch("llm_opencode.fetch_docs_page", return_value=DOCS_HTML),
    ):
        result = fetch_unified_models()

    assert result["models"][0]["model_id"] == "grok-4.5"
    assert json.loads(cache_file.read_text()) == result


def test_fetch_unified_models_docs_failure_raises(tmp_path):
    with (
        patch("llm_opencode.llm.user_dir", return_value=tmp_path),
        patch("llm_opencode.get_opencode_models"),
        patch("llm_opencode.fetch_docs_page", side_effect=DocsScrapeError("boom")),
        pytest.raises(DocsScrapeError),
    ):
        fetch_unified_models()


def test_fetch_unified_models_no_contract_tables_raises(tmp_path):
    with (
        patch("llm_opencode.llm.user_dir", return_value=tmp_path),
        patch("llm_opencode.get_opencode_models"),
        patch("llm_opencode.fetch_docs_page", return_value="<html><body>changed</body></html>"),
        pytest.raises(DocsScrapeError),
    ):
        fetch_unified_models()
