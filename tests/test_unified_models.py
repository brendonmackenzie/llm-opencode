import json
import time
from unittest.mock import patch

import pytest

from llm_opencode import (
    DownloadError,
    api_model_to_entry,
    fetch_models_snapshot,
    read_snapshot,
    refresh_snapshot,
)
from tests._payload import (
    anthropic_api_model,
    api_model,
    free_api_model,
    google_api_model,
    responses_api_model,
)

# ─── api_model_to_entry ──────────────────────────────────────────────────────


def test_entry_maps_chat_completions_model():
    entry = api_model_to_entry(api_model())

    assert entry["model_id"] == "glm-5.3"
    assert entry["model_name"] == "GLM-5.3"
    assert entry["canonical_id"] == "zhipuai/glm-5.3"
    assert entry["provider"] == "opencode"
    assert entry["created"] == 1795050567
    assert entry["family"] == "glm"
    assert entry["recommended_api"] == "chat-completions"
    assert entry["endpoint"] == "https://opencode.ai/inference/go/openai/v1"
    assert entry["ai_sdk_package"] == "@ai-sdk/openai-compatible"
    assert entry["pricing"] == {
        "input": 1.4,
        "output": 4.4,
        "cached_read": 0.26,
        "cached_write": None,
    }
    assert entry["monthly_usage"] == {"go": 500, "go-plus": 1000, "go-teams": 505}
    assert entry["training"] is False
    assert entry["zdr"] is True
    assert entry["limited_regions"] is False
    assert entry["featured"] is True
    assert entry["limits"] == {"context": 1000000, "output": 131072}
    assert entry["capabilities"]["structured_output"] is True
    assert entry["open_weights"] is True
    assert entry["release_date"] == "2026-08-14"


def test_entry_maps_anthropic_model_with_scalar_cache_write():
    entry = api_model_to_entry(anthropic_api_model())

    assert entry["recommended_api"] == "messages"
    assert entry["endpoint"] == "https://opencode.ai/inference/go/anthropic/v1"
    assert entry["ai_sdk_package"] == "@ai-sdk/anthropic"
    assert entry["pricing"]["cached_write"] == 0.125
    assert entry["cost_overrides"]["input"][0]["price"] == 0.5
    assert entry["cost_overrides"]["output"][0]["price"] == 2.5


def test_entry_preserves_ttl_keyed_cache_write():
    entry = api_model_to_entry(responses_api_model())

    assert entry["recommended_api"] == "responses"
    assert entry["pricing"]["cached_write"] == {"30m": 0.125, "12h": 0.25}


def test_entry_free_model_has_empty_usage_and_zero_prices():
    entry = api_model_to_entry(free_api_model())

    assert entry["monthly_usage"] == {}
    assert entry["pricing"]["input"] == 0
    assert entry["pricing"]["output"] == 0
    assert entry["pricing"]["cached_read"] == 0


def test_entry_google_model_carries_its_route():
    entry = api_model_to_entry(google_api_model())

    assert entry["recommended_api"] == "generate-content"
    assert entry["endpoint"] == "https://opencode.ai/inference/go/google/v1beta"
    assert entry["cost_multipliers"][0]["factor"] == 2


def test_entry_tolerates_missing_payload_fields():
    entry = api_model_to_entry({"id": "omen-alpha"})

    assert entry["model_id"] == "omen-alpha"
    assert entry["model_name"] == ""
    assert entry["recommended_api"] == ""
    assert entry["endpoint"] == ""
    assert entry["pricing"] == {
        "input": None,
        "output": None,
        "cached_read": None,
        "cached_write": None,
    }
    assert entry["monthly_usage"] == {}
    assert entry["training"] is None
    assert entry["zdr"] is None


# ─── refresh_snapshot (fetch, shape, persist) ────────────────────────────────


def test_refresh_snapshot_fetches_shapes_and_persists(tmp_path):
    with (
        patch("llm_opencode.llm.user_dir", return_value=tmp_path),
        patch(
            "llm_opencode.get_opencode_models",
            return_value=[
                api_model(),
                anthropic_api_model(),
                free_api_model(),
            ],
        ),
    ):
        snapshot = refresh_snapshot()

    assert set(snapshot) == {"fetched_at", "warnings", "models"}
    assert isinstance(snapshot["fetched_at"], int)
    assert snapshot["warnings"] == []
    assert [m["model_id"] for m in snapshot["models"]] == [
        "glm-5.3",
        "claude-haiku-5-5",
        "beam-free",
    ]
    assert snapshot["models"][0]["pricing"]["input"] == 1.4

    cached = json.loads((tmp_path / "opencode_api_snapshot.json").read_text())
    assert cached == snapshot


def test_refresh_snapshot_warns_for_idless_entries(tmp_path):
    with (
        patch("llm_opencode.llm.user_dir", return_value=tmp_path),
        patch(
            "llm_opencode.get_opencode_models",
            return_value=[api_model(), "not-a-model", {"no": "id"}],
        ),
    ):
        snapshot = refresh_snapshot()

    assert [m["model_id"] for m in snapshot["models"]] == ["glm-5.3"]
    assert snapshot["warnings"] == [
        "models API entry 1: no id — skipped",
        "models API entry 2: no id — skipped",
    ]


# ─── fetch_models_snapshot (snapshot payload) ────────────────────────────────


def test_fetch_models_snapshot_uses_fresh_snapshot_cache(tmp_path):
    payload = {
        "fetched_at": int(time.time()),
        "warnings": [],
        "models": [{"model_id": "glm-5.3"}],
    }
    (tmp_path / "opencode_api_snapshot.json").write_text(json.dumps(payload))

    with (
        patch("llm_opencode.llm.user_dir", return_value=tmp_path),
        patch("llm_opencode.get_opencode_models") as mock_get,
    ):
        result = fetch_models_snapshot()

    assert result == payload
    mock_get.assert_not_called()


def test_fetch_models_snapshot_fetches_and_caches_snapshot(tmp_path):
    with (
        patch("llm_opencode.llm.user_dir", return_value=tmp_path),
        patch(
            "llm_opencode.get_opencode_models",
            return_value=[api_model()],
        ),
    ):
        result = fetch_models_snapshot()

    assert [m["model_id"] for m in result["models"]] == ["glm-5.3"]
    assert json.loads((tmp_path / "opencode_api_snapshot.json").read_text()) == result


def test_fetch_models_snapshot_stale_snapshot_refetches(tmp_path):
    path = tmp_path / "opencode_api_snapshot.json"
    path.write_text(
        json.dumps(
            {"fetched_at": 1, "warnings": [], "models": [{"model_id": "old"}]}
        )
    )

    with (
        patch("llm_opencode.llm.user_dir", return_value=tmp_path),
        patch(
            "llm_opencode.get_opencode_models",
            return_value=[api_model()],
        ),
    ):
        result = fetch_models_snapshot()

    assert result["models"][0]["model_id"] == "glm-5.3"
    assert json.loads(path.read_text()) == result


def test_fetch_models_snapshot_fresh_corrupt_cache_refetches(tmp_path):
    path = tmp_path / "opencode_api_snapshot.json"
    path.write_text("not valid json")

    with (
        patch("llm_opencode.llm.user_dir", return_value=tmp_path),
        patch(
            "llm_opencode.get_opencode_models",
            return_value=[api_model()],
        ),
    ):
        result = fetch_models_snapshot()

    assert result["models"][0]["model_id"] == "glm-5.3"
    assert json.loads(path.read_text()) == result


def test_fetch_models_snapshot_stale_refresh_failure_serves_last_good(tmp_path):
    stale_at = int(time.time()) - 30 * 3600
    payload = {
        "fetched_at": stale_at,
        "warnings": [],
        "models": [{"model_id": "old"}],
    }
    (tmp_path / "opencode_api_snapshot.json").write_text(json.dumps(payload))

    with (
        patch("llm_opencode.llm.user_dir", return_value=tmp_path),
        patch(
            "llm_opencode.get_opencode_models",
            side_effect=DownloadError("no net"),
        ),
    ):
        result = fetch_models_snapshot()

    assert result == payload


def test_fetch_models_snapshot_no_snapshot_fetch_failure_raises(tmp_path):
    with (
        patch("llm_opencode.llm.user_dir", return_value=tmp_path),
        patch(
            "llm_opencode.get_opencode_models",
            side_effect=DownloadError("no net"),
        ),
        pytest.raises(DownloadError),
    ):
        fetch_models_snapshot()


# ─── get_opencode_models (envelope validation) ──────────────────────────────


@patch(
    "llm_opencode.unified.fetch_cached_json",
    return_value={"data": [{"id": "glm-5.3"}]},
)
def test_get_opencode_models_returns_data_list(mock_fetch, tmp_path):
    from llm_opencode import get_opencode_models

    with patch("llm_opencode.llm.user_dir", return_value=tmp_path):
        assert get_opencode_models() == [{"id": "glm-5.3"}]


@patch("llm_opencode.unified.fetch_cached_json", return_value={"unexpected": "shape"})
def test_get_opencode_models_rejects_non_list_envelope(mock_fetch, tmp_path):
    from llm_opencode import get_opencode_models

    with (
        patch("llm_opencode.llm.user_dir", return_value=tmp_path),
        pytest.raises(DownloadError),
    ):
        get_opencode_models()


@patch("llm_opencode.unified.fetch_cached_json", return_value={"data": "not-a-list"})
def test_get_opencode_models_rejects_non_list_data(mock_fetch, tmp_path):
    from llm_opencode import get_opencode_models

    with (
        patch("llm_opencode.llm.user_dir", return_value=tmp_path),
        pytest.raises(DownloadError),
    ):
        get_opencode_models()


# ─── read_snapshot ───────────────────────────────────────────────────────────


def test_snapshot_age_tolerates_missing_or_non_numeric():
    from llm_opencode import snapshot_age

    assert snapshot_age(None) is None
    assert snapshot_age("not-a-number") is None
    assert snapshot_age(0) > 86_400  # epoch age is measured, not fatal


def test_read_snapshot_missing_returns_none(tmp_path):
    with patch("llm_opencode.llm.user_dir", return_value=tmp_path):
        assert read_snapshot() is None


def test_read_snapshot_tolerates_wrong_shape(tmp_path):
    path = tmp_path / "opencode_api_snapshot.json"
    path.write_text(json.dumps({"fetched_at": 1}))

    with patch("llm_opencode.llm.user_dir", return_value=tmp_path):
        assert read_snapshot() is None
