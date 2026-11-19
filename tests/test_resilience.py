import json
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner
from llm.cli import cli

from llm_opencode import (
    DownloadError,
    fetch_models_snapshot,
    read_snapshot,
    register_models,
)
from tests._payload import api_model, entry, snapshot

REFRESH_TIMEOUT = 86400  # 24 hours

SNAPSHOT_FILE = "opencode_api_snapshot.json"


def _write_snapshot(user_dir, fetched_at, models=None, warnings=None):
    path = Path(user_dir) / SNAPSHOT_FILE
    path.write_text(
        json.dumps(
            {
                "fetched_at": fetched_at,
                "warnings": warnings or [],
                "models": [entry()] if models is None else models,
            }
        )
    )
    return path


def _stale_since_iso(fetched_at):
    return datetime.fromtimestamp(fetched_at, tz=timezone.utc).isoformat()


# ─── expired-but-present snapshot: served with its age, never discarded ─────


def test_expired_snapshot_served_with_age(user_dir):
    fetched_at = int(time.time()) - 26 * 3600
    _write_snapshot(user_dir, fetched_at)

    with patch(
        "llm_opencode.get_opencode_models",
        side_effect=DownloadError("no net"),
    ) as mock_api:
        result = fetch_models_snapshot()

    assert result["fetched_at"] == fetched_at
    assert result["models"] == [entry()]
    assert (user_dir / SNAPSHOT_FILE).read_text() == json.dumps(result)
    assert mock_api.call_count == 1


def test_expiry_gates_refresh_attempts_only(user_dir):
    fetched_at = int(time.time()) - 26 * 3600
    _write_snapshot(user_dir, fetched_at)

    with patch(
        "llm_opencode.get_opencode_models",
        return_value=[api_model()],
    ) as mock_api:
        result = fetch_models_snapshot()

    mock_api.assert_called_once()
    assert result["fetched_at"] > fetched_at
    assert json.loads((user_dir / SNAPSHOT_FILE).read_text())["fetched_at"] == (
        result["fetched_at"]
    )


def test_fresh_snapshot_served_without_a_refresh_attempt(user_dir):
    _write_snapshot(user_dir, int(time.time()))

    with patch("llm_opencode.get_opencode_models") as mock_api:
        result = fetch_models_snapshot()

    mock_api.assert_not_called()
    assert result["models"] == [entry()]


def test_fresh_fetch_failure_over_live_data_serves_last_good_and_marks_staleness(user_dir):
    fetched_at = int(time.time()) - 30 * 3600
    _write_snapshot(user_dir, fetched_at)

    with patch(
        "llm_opencode.get_opencode_models",
        side_effect=DownloadError("no net"),
    ):
        result = fetch_models_snapshot()

    assert result["fetched_at"] == fetched_at
    snapshot_age = time.time() - result["fetched_at"]
    assert snapshot_age > REFRESH_TIMEOUT


# ─── registration: snapshot-only, one-time bootstrap ─────────────────────────


def test_registration_bootstraps_missing_snapshot(user_dir):
    assert read_snapshot() is None

    with (
        patch("llm_opencode.get_opencode_models", return_value=[api_model()]),
        patch("llm.get_key", return_value="sk-test"),
    ):
        register = MagicMock()
        register_models(register)

    assert register.call_count == 1
    assert register.call_args[0][0].model_id == "opencode-go/glm-5.3"
    persisted = read_snapshot()
    assert persisted is not None
    assert persisted["models"]


def test_registration_bootstrap_failure_registers_nothing_without_crash(user_dir):
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
    fetched_at = int(time.time()) - 50 * 3600
    _write_snapshot(user_dir, fetched_at)

    with (
        patch("llm_opencode.get_opencode_models") as mock_api,
        patch("llm.get_key", return_value="sk-test"),
    ):
        register = MagicMock()
        register_models(register)

    mock_api.assert_not_called()
    assert register.call_count == 1
    assert register.call_args[0][0].model_id == "opencode-go/glm-5.3"
    assert capsys.readouterr().out == ""


# ─── detail: staleness surfaces only here ────────────────────────────────────


def _invoke_detail_with_snapshot(payload, *args):
    runner = CliRunner()
    with patch("llm_opencode.fetch_models_snapshot", return_value=payload):
        return runner.invoke(cli, ["opencode", "detail", *args])


def test_detail_serves_last_good_snapshot_when_fresh_fetch_fails(user_dir):
    fetched_at = int(time.time()) - 30 * 3600
    _write_snapshot(user_dir, fetched_at)

    with patch(
        "llm_opencode.get_opencode_models",
        side_effect=DownloadError("no net"),
    ):
        runner = CliRunner()
        result = runner.invoke(cli, ["opencode", "detail"])

    assert result.exit_code == 0, result.output
    assert "glm-5.3" in result.output
    assert "models API unavailable" not in result.output
    assert "Snapshot is stale" in result.output
    assert _stale_since_iso(fetched_at) in result.output


def test_detail_without_snapshot_fetch_failure_keeps_exit_1_api_url_path(user_dir):
    assert read_snapshot() is None

    with patch(
        "llm_opencode.get_opencode_models",
        side_effect=DownloadError("no net"),
    ):
        runner = CliRunner()
        result = runner.invoke(cli, ["opencode", "detail"])

    assert result.exit_code == 1
    assert "models API unavailable" in result.output
    assert "https://opencode.ai/inference/go/v1/models" in result.output
    assert read_snapshot() is None


def test_detail_table_banner_appears_only_when_stale():
    fresh_payload = snapshot(entry(), fetched_at=int(time.time()) - 3600)
    stale_fetched_at = int(time.time()) - 30 * 3600
    stale_payload = snapshot(entry(), fetched_at=stale_fetched_at)

    fresh = _invoke_detail_with_snapshot(fresh_payload)
    assert fresh.exit_code == 0, fresh.output
    assert "Snapshot is stale" not in fresh.output

    stale = _invoke_detail_with_snapshot(stale_payload)
    assert stale.exit_code == 0, stale.output
    assert "Snapshot is stale" in stale.output
    assert _stale_since_iso(stale_fetched_at) in stale.output


def test_detail_json_stale_since_present_only_when_stale():
    stale_fetched_at = int(time.time()) - 26 * 3600
    fresh_payload = snapshot(entry(), fetched_at=int(time.time()) - 3600)
    stale_payload = snapshot(entry(), fetched_at=stale_fetched_at)

    fresh = _invoke_detail_with_snapshot(fresh_payload, "--format", "json")
    assert fresh.exit_code == 0, fresh.output
    fresh_meta = json.loads(fresh.output)["meta"]
    assert "stale_since" not in fresh_meta

    stale = _invoke_detail_with_snapshot(stale_payload, "--format", "json")
    assert stale.exit_code == 0, stale.output
    stale_meta = json.loads(stale.output)["meta"]
    assert stale_meta["stale_since"] == _stale_since_iso(stale_fetched_at)


def test_unregistered_marker_is_honest_under_staleness():
    stale_payload = snapshot(
        entry(api_model(model_id="omen-alpha", api={}, recommended_api="")),
        fetched_at=int(time.time()) - 30 * 3600,
    )

    result = _invoke_detail_with_snapshot(stale_payload)

    assert result.exit_code == 0, result.output
    assert "Snapshot is stale" in result.output
    assert "omen-alpha" in result.output
    assert "unrecognized recommended_api" in result.output


def test_csv_and_pricing_formats_carry_no_banner():
    stale_payload = snapshot(entry(), fetched_at=int(time.time()) - 30 * 3600)

    csv_result = _invoke_detail_with_snapshot(stale_payload, "--format", "csv")
    pricing_result = _invoke_detail_with_snapshot(stale_payload, "--format", "pricing")

    assert "Snapshot is stale" not in csv_result.output
    assert "Snapshot is stale" not in pricing_result.output


# ─── docs guidance for the deleted-snapshot state ────────────────────────────


def test_readme_documents_deleted_snapshot_state():
    readme = Path(__file__).parent.parent / "README.md"
    text = readme.read_text()
    assert SNAPSHOT_FILE in text
    assert "llm opencode detail" in text
    assert "repopulat" in text.lower()
