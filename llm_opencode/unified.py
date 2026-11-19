"""The OpenCode Go model catalog, from one JSON source of truth.

OpenCode serves rich per-model metadata from a single endpoint
(:data:`MODELS_URL`): display names, pricing, plan allowances, policy
flags, capabilities, and the per-model API routes. This module fetches
it (with a disk cache), shapes each payload entry into the unified
model entry the rest of the plugin consumes, and snapshots the result.
"""

import json
import time
from pathlib import Path

import llm

MODELS_URL = "https://opencode.ai/inference/go/v1/models"

SNAPSHOT_REFRESH_TIMEOUT = 86400  # 24 hours


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
        import httpx2

        response = httpx2.get(url, follow_redirects=True)
        response.raise_for_status()
        payload = response.json()
    except (httpx2.HTTPError, json.JSONDecodeError):
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

    with open(path, "w") as file:
        json.dump(payload, file)
    return payload


def get_opencode_models():
    payload = fetch_cached_json(
        url=MODELS_URL,
        path=llm.user_dir() / "opencode_models.json",
        cache_timeout=3600,
    )
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise DownloadError(f"Models API payload is not a model list: {MODELS_URL}")
    return data


def api_model_to_entry(model):
    """Shape one models-API payload entry into a unified model entry.

    Pricing is plan-independent in the API: the entry carries one
    ``pricing`` block plus one per-plan ``monthly_usage`` map (plan keys
    ``go`` / ``go-plus`` / ``go-teams``; free models have none).
    ``cached_write`` is kept exactly as served — a scalar price, or a
    TTL-keyed dict of prices.
    """
    recommended = model.get("recommended_api") or ""
    api_block = (model.get("api") or {}).get(recommended) or {}
    cost_text = (model.get("cost") or {}).get("text") or {}
    input_cost = cost_text.get("input") or {}
    output_cost = cost_text.get("output") or {}
    cache_cost = input_cost.get("cache") or {}
    go = model.get("go") or {}
    return {
        "model_id": model.get("id", ""),
        "model_name": model.get("name", ""),
        "canonical_id": model.get("canonical_id", ""),
        "provider": model.get("owned_by", ""),
        "created": model.get("created"),
        "family": model.get("family", ""),
        "description": model.get("description", ""),
        "open_weights": model.get("open_weights"),
        "knowledge": model.get("knowledge", ""),
        "release_date": model.get("release_date", ""),
        "last_updated": model.get("last_updated", ""),
        "recommended_api": recommended,
        "endpoint": api_block.get("base_url", ""),
        "ai_sdk_package": (api_block.get("sdk") or {}).get("aisdk", ""),
        "pricing": {
            "input": input_cost.get("price"),
            "output": output_cost.get("price"),
            "cached_read": cache_cost.get("read"),
            "cached_write": cache_cost.get("write"),
        },
        "cost_overrides": {
            "input": input_cost.get("overrides") or [],
            "output": output_cost.get("overrides") or [],
        },
        "cost_multipliers": cost_text.get("multipliers") or [],
        "monthly_usage": go.get("monthly_usage") or {},
        "training": go.get("training"),
        "zdr": go.get("zdr"),
        "limited_regions": go.get("limited_regions"),
        "featured": go.get("featured"),
        "limits": model.get("limit") or {},
        "capabilities": model.get("capabilities") or {},
    }


def _snapshot(models, warnings):
    return {
        "fetched_at": int(time.time()),
        "warnings": warnings,
        "models": models,
    }


def _snapshot_path():
    return llm.user_dir() / "opencode_api_snapshot.json"


def read_snapshot():
    """Read the last-good snapshot file without touching the network.

    Returns ``None`` when no snapshot has ever been written, or when the file
    is missing, unreadable, or not shaped like a snapshot.
    """
    path = _snapshot_path()
    if path.is_file():
        try:
            with open(path) as file:
                cached = json.load(file)
            if isinstance(cached, dict) and "models" in cached:
                return cached
        except (json.JSONDecodeError, OSError):
            pass
    return None


def snapshot_age(timestamp):
    """Age of a ``fetched_at`` timestamp in seconds, or ``None`` when the
    value is missing or not a number."""
    try:
        return time.time() - float(timestamp)
    except (TypeError, ValueError):
        return None


def _snapshot_is_fresh(snapshot):
    age = snapshot_age(snapshot.get("fetched_at", 0))
    return age is not None and age < SNAPSHOT_REFRESH_TIMEOUT


def refresh_snapshot():
    """Fetch the models API fresh, shape it, persist, and return the
    snapshot. Raises :class:`DownloadError` on fetch failure with no
    usable cache.
    """
    import llm_opencode as _root

    api_models = _root.get_opencode_models()
    models, warnings = [], []
    for index, model in enumerate(api_models):
        if not isinstance(model, dict) or not model.get("id"):
            warnings.append(f"models API entry {index}: no id — skipped")
            continue
        models.append(api_model_to_entry(model))
    snapshot = _snapshot(models, warnings)

    cache_path = _snapshot_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as file:
        json.dump(snapshot, file)

    return snapshot


def fetch_models_snapshot():
    """Serve the model snapshot, refreshing it only when warranted.

    The snapshot payload is ``{"fetched_at", "warnings", "models"}``. The 24
    hour timeout gates refresh attempts only: a fresh snapshot is served as
    is, an expired-but-present snapshot gets one refresh attempt and is
    served with its age when the attempt fails, and a missing snapshot gets
    one full fetch whose failure (:class:`DownloadError`) raises.
    """
    snapshot = read_snapshot()
    if snapshot is None:
        return refresh_snapshot()
    if _snapshot_is_fresh(snapshot):
        return snapshot
    try:
        return refresh_snapshot()
    except Exception:
        return snapshot
