import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner
from llm.cli import cli

from llm_opencode import (
    DocsScrapeError,
    DownloadError,
    fetch_unified_models,
    read_snapshot,
    register_models,
)

REFRESH_TIMEOUT = 86400  # 24 hours


def _entry(
    model_id="grok-4.5",
    endpoint="https://opencode.ai/inference/go/v1/chat/completions",
):
    entry = {
        "model_id": model_id,
        "model_name": model_id,
        "matched": True,
        "ai_sdk_package": "@ai-sdk/openai-compatible",
    }
    if endpoint is not None:
        entry["endpoint"] = endpoint
    return entry


def _write_snapshot(
    user_dir, scraped_at, models=None, warnings=None
):
    path = Path(user_dir) / "opencode_unified_models.json"
    path.write_text(
        json.dumps(
            {
                "scraped_at": scraped_at,
                "warnings": warnings or [],
                "models": [_entry()] if models is None else models,
            }
        )
    )
    return path


def _stale_since_iso(scraped_at):
    return datetime.fromtimestamp(scraped_at, tz=timezone.utc).isoformat()


DOCS_HTML = (
    "<table><thead><tr><th>Model</th><th>Input</th><th>Output</th>"
    "<th>Cached Read</th><th>Cached Write</th><th>Monthly limit</th></tr></thead><tbody>"
    "<tr><td>Grok 4.5</td><td>$2.00</td><td>$6.00</td><td>$0.30</td><td>-</td>"
    "<td><strong>$15</strong></td></tr></tbody></table>"
    "<table><thead><tr><th>Model</th><th>Model ID</th><th>Endpoint</th>"
    "<th>AI SDK Package</th></tr></thead><tbody>"
    "<tr><td>Grok 4.5</td><td>grok-4.5</td><td><code>https://x/v1/chat/completions</code></td>"
    "<td><code>@ai-sdk/openai-compatible</code></td></tr></tbody></table>"
)


# ─── expired-but-present snapshot: served with its age, never discarded ─────


def test_expired_snapshot_served_with_age(user_dir):
    scraped_at = int(time.time()) - 26 * 3600
    _write_snapshot(user_dir, scraped_at)

    with (
        patch("llm_opencode.get_opencode_models") as mock_api,
        patch("llm_opencode.fetch_docs_page", side_effect=DocsScrapeError("boom")),
    ):
        result = fetch_unified_models()

    assert result["scraped_at"] == scraped_at
    assert result["models"] == [_entry()]
    assert (user_dir / "opencode_unified_models.json").read_text() == json.dumps(result)
    mock_api.assert_called_once()
    assert mock_api.call_count == 1


def test_expiry_gates_refresh_attempts_only(user_dir):
    scraped_at = int(time.time()) - 26 * 3600
    _write_snapshot(user_dir, scraped_at)

    with (
        patch("llm_opencode.get_opencode_models", return_value=[]),
        patch("llm_opencode.fetch_docs_page", return_value=DOCS_HTML) as mock_docs,
    ):
        result = fetch_unified_models()

    mock_docs.assert_called_once()
    assert result["scraped_at"] > scraped_at
    assert json.loads(
        (user_dir / "opencode_unified_models.json").read_text()
    )["scraped_at"] == result["scraped_at"]


def test_fresh_snapshot_served_without_a_refresh_attempt(user_dir):
    _write_snapshot(user_dir, int(time.time()))

    with (
        patch("llm_opencode.get_opencode_models") as mock_api,
        patch("llm_opencode.fetch_docs_page") as mock_docs,
    ):
        result = fetch_unified_models()

    mock_api.assert_not_called()
    mock_docs.assert_not_called()


def test_fresh_scrape_failure_over_live_data_serves_last_good_and_marks_staleness(user_dir):
    scraped_at = int(time.time()) - 30 * 3600
    _write_snapshot(user_dir, scraped_at)

    with (
        patch("llm_opencode.get_opencode_models", return_value=[]),
        patch("llm_opencode.fetch_docs_page", side_effect=DocsScrapeError("boom")),
    ):
        result = fetch_unified_models()

    assert result["scraped_at"] == scraped_at
    scrape_age = time.time() - result["scraped_at"]
    assert scrape_age > REFRESH_TIMEOUT


# ─── registration: snapshot-only, one-time bootstrap ─────────────────────────


def test_registration_bootstraps_missing_snapshot(user_dir):
    assert read_snapshot() is None

    with (
        patch("llm_opencode.get_opencode_models", return_value=[]),
        patch("llm_opencode.fetch_docs_page", return_value=DOCS_HTML),
        patch("llm.get_key", return_value="sk-test"),
    ):
        register = MagicMock()
        register_models(register)

    assert register.call_count == 1
    assert register.call_args[0][0].model_id == "opencode-go/grok-4.5"
    persisted = read_snapshot()
    assert persisted is not None
    assert persisted["models"]


def test_registration_bootstrap_failure_registers_nothing_without_crash(user_dir):
    assert read_snapshot() is None

    with (
        patch("llm_opencode.get_opencode_models", return_value=[]),
        patch("llm_opencode.fetch_docs_page", side_effect=DocsScrapeError("boom")),
        patch("llm.get_key", return_value="sk-test"),
    ):
        register = MagicMock()
        register_models(register)

    register.assert_not_called()
    assert read_snapshot() is None


def test_registration_bootstrap_download_failure_registers_nothing_without_crash(user_dir):
    assert read_snapshot() is None

    with (
        patch("llm_opencode.get_opencode_models", side_effect=DownloadError("no net")),
        patch("llm.get_key", return_value="sk-test"),
    ):
        register = MagicMock()
        register_models(register)

    register.assert_not_called()
    assert read_snapshot() is None


def test_registration_serves_stale_snapshot_silently(user_dir, capsys):
    scraped_at = int(time.time()) - 50 * 3600
    _write_snapshot(user_dir, scraped_at)

    with (
        patch("llm_opencode.get_opencode_models") as mock_api,
        patch("llm_opencode.fetch_docs_page") as mock_docs,
        patch("llm.get_key", return_value="sk-test"),
    ):
        register = MagicMock()
        register_models(register)

    mock_api.assert_not_called()
    mock_docs.assert_not_called()
    assert register.call_count == 1
    assert register.call_args[0][0].model_id == "opencode-go/grok-4.5"
    assert capsys.readouterr().out == ""


# ─── detail: staleness surfaces only here ────────────────────────────────────


def _invoke_detail_with_snapshot(snapshot, *args):
    runner = CliRunner()
    with patch("llm_opencode.fetch_unified_models", return_value=snapshot):
        return runner.invoke(cli, ["opencode", "detail", *args])


def test_detail_serves_last_good_snapshot_when_fresh_scrape_fails(user_dir):
    scraped_at = int(time.time()) - 30 * 3600
    _write_snapshot(user_dir, scraped_at)

    with (
        patch("llm_opencode.get_opencode_models", return_value=[]),
        patch("llm_opencode.fetch_docs_page", side_effect=DocsScrapeError("boom")),
    ):
        runner = CliRunner()
        result = runner.invoke(cli, ["opencode", "detail"])

    assert result.exit_code == 0, result.output
    assert "grok-4.5" in result.output
    assert "OpenCode Go page scraping failed" not in result.output
    assert "Snapshot is stale" in result.output
    assert _stale_since_iso(scraped_at) in result.output


def test_detail_without_snapshot_failure_keeps_exit_1_docs_url_path(user_dir):
    assert read_snapshot() is None

    with (
        patch("llm_opencode.get_opencode_models", return_value=[]),
        patch("llm_opencode.fetch_docs_page", side_effect=DocsScrapeError("boom")),
    ):
        runner = CliRunner()
        result = runner.invoke(cli, ["opencode", "detail"])

    assert result.exit_code == 1
    assert "OpenCode Go page scraping failed" in result.output
    assert "https://opencode.ai/docs/go/" in result.output
    assert read_snapshot() is None


def test_detail_table_banner_appears_only_when_stale():
    fresh_snapshot = {
        "scraped_at": int(time.time()) - 3600,
        "warnings": [],
        "models": [_entry()],
    }
    stale_scraped_at = int(time.time()) - 30 * 3600
    stale_snapshot = {
        "scraped_at": stale_scraped_at,
        "warnings": [],
        "models": [_entry()],
    }

    fresh = _invoke_detail_with_snapshot(fresh_snapshot)
    assert fresh.exit_code == 0, fresh.output
    assert "Snapshot is stale" not in fresh.output

    stale = _invoke_detail_with_snapshot(stale_snapshot)
    assert stale.exit_code == 0, stale.output
    assert "Snapshot is stale" in stale.output
    assert _stale_since_iso(stale_scraped_at) in stale.output


def test_detail_json_stale_since_present_only_when_stale():
    stale_scraped_at = int(time.time()) - 26 * 3600
    fresh_snapshot = {
        "scraped_at": int(time.time()) - 3600,
        "warnings": [],
        "models": [_entry()],
    }
    stale_snapshot = {
        "scraped_at": stale_scraped_at,
        "warnings": [],
        "models": [_entry()],
    }

    fresh = _invoke_detail_with_snapshot(fresh_snapshot, "--format", "json")
    assert fresh.exit_code == 0, fresh.output
    fresh_meta = json.loads(fresh.output)["meta"]
    assert "stale_since" not in fresh_meta

    stale = _invoke_detail_with_snapshot(stale_snapshot, "--format", "json")
    assert stale.exit_code == 0, stale.output
    stale_meta = json.loads(stale.output)["meta"]
    assert stale_meta["stale_since"] == _stale_since_iso(stale_scraped_at)


def test_unregistered_marker_is_honest_under_staleness():
    stale_snapshot = {
        "scraped_at": int(time.time()) - 30 * 3600,
        "warnings": [],
        "models": [{"model_id": "omen-alpha"}],
    }

    result = _invoke_detail_with_snapshot(stale_snapshot)

    assert result.exit_code == 0, result.output
    assert "Snapshot is stale" in result.output
    assert "omen-alpha" in result.output
    assert "unknown" in result.output


def test_csv_and_pricing_formats_carry_no_banner():
    stale_snapshot = {
        "scraped_at": int(time.time()) - 30 * 3600,
        "warnings": [],
        "models": [_entry()],
    }

    csv_result = _invoke_detail_with_snapshot(stale_snapshot, "--format", "csv")
    pricing_result = _invoke_detail_with_snapshot(stale_snapshot, "--format", "pricing")

    assert "Snapshot is stale" not in csv_result.output
    assert "Snapshot is stale" not in pricing_result.output


# ─── plain listing: snapshot-only, never scraping ────────────────────────────


def test_listing_reads_snapshot_only_never_scrapes(user_dir):
    _write_snapshot(
        user_dir,
        int(time.time()) - 30 * 3600,
        models=[_entry(), {"model_id": "omen-alpha"}],
    )

    with (
        patch("llm_opencode.get_opencode_models") as mock_api,
        patch("llm_opencode.fetch_docs_page") as mock_docs,
    ):
        runner = CliRunner()
        result = runner.invoke(cli, ["opencode", "models"])

    assert result.exit_code == 0, result.output
    assert "- id: grok-4.5" in result.output
    assert "protocol: openai" in result.output
    assert "omen-alpha" in result.output
    assert "unknown (no scrape data)" in result.output
    mock_api.assert_not_called()
    mock_docs.assert_not_called()


def test_listing_without_snapshot_reads_api_cache_without_refresh(user_dir):
    api_cache = user_dir / "opencode_models.json"
    api_cache.write_text(json.dumps({"data": [{"id": "glm-5"}]}))
    os.utime(api_cache, (0, 0))  # expired on disk: still no refresh, no fetch

    with (
        patch("llm_opencode.get_opencode_models") as mock_api,
        patch("llm_opencode.fetch_docs_page") as mock_docs,
    ):
        runner = CliRunner()
        result = runner.invoke(cli, ["opencode", "models"])

    assert result.exit_code == 0, result.output
    assert "- id: glm-5" in result.output
    assert "protocol: unknown (no scrape data)" in result.output
    mock_api.assert_not_called()
    mock_docs.assert_not_called()


def test_listing_fresh_install_prints_hint_and_hits_nothing(user_dir):
    with (
        patch("httpx2.get", side_effect=AssertionError("listing hit the network")),
        patch("llm_opencode.get_opencode_models") as mock_api,
        patch("llm_opencode.fetch_docs_page") as mock_docs,
    ):
        runner = CliRunner()
        result = runner.invoke(cli, ["opencode", "models"])

    assert result.exit_code == 0, result.output
    assert "llm opencode detail" in result.output
    mock_api.assert_not_called()
    mock_docs.assert_not_called()


def test_stale_listing_json_carries_snapshot_protocols(user_dir):
    _write_snapshot(user_dir, int(time.time()) - 30 * 3600)

    with (
        patch("llm_opencode.get_opencode_models") as mock_api,
        patch("llm_opencode.fetch_docs_page") as mock_docs,
    ):
        runner = CliRunner()
        result = runner.invoke(cli, ["opencode", "models", "--json"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data == [{"id": "grok-4.5", "protocol": "openai"}]
    mock_api.assert_not_called()
    mock_docs.assert_not_called()


# ─── docs guidance for the deleted-snapshot state ────────────────────────────


def test_readme_documents_deleted_snapshot_state():
    readme = Path(__file__).parent.parent / "README.md"
    text = readme.read_text()
    assert "opencode_unified_models.json" in text
    assert "llm opencode detail" in text
    assert ".bootstrap" in text or "repopulat" in text.lower()
