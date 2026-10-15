import json
import time
from pathlib import Path

import llm

from .docs_scraper import (
    DocsScrapeError,
    fetch_docs_page,
    parse_docs_tables,
    parse_html_tables,
)

MODELS_URL = "https://opencode.ai/zen/go/v1/models"

UNIFIED_MODELS_CACHE_TIMEOUT = 86400  # 24 hours


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
        with open(path, "w") as file:
            json.dump(response.json(), file)
        return response.json()
    except httpx2.HTTPError:
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


def get_opencode_models():
    models = fetch_cached_json(
        url=MODELS_URL,
        path=llm.user_dir() / "opencode_models.json",
        cache_timeout=3600,
    )["data"]
    return models


def read_api_models_cache():
    """Read the models-API list cache from disk without refresh or fetch.

    Returns the ``data`` list when the file exists and parses, else ``None``.
    There is intentionally no freshness check and no network fallback: the
    plain listing is offline-only.
    """
    path = llm.user_dir() / "opencode_models.json"
    if path.is_file():
        try:
            with open(path) as file:
                cached = json.load(file)
            data = cached.get("data") if isinstance(cached, dict) else None
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return None


def _unified_entry(model_id, api_model, docs_entry):
    entry = {
        "model_id": model_id,
        "model_name": docs_entry.get("model_name", ""),
        "provider": api_model.get("owned_by", ""),
        "created": api_model.get("created"),
        "matched": docs_entry.get("matched", False),
        "endpoint": docs_entry.get("endpoint", ""),
        "ai_sdk_package": docs_entry.get("ai_sdk_package", ""),
        "model_training": docs_entry.get("model_training", ""),
        "data_retention": docs_entry.get("data_retention", ""),
        "pricing": docs_entry.get("pricing", {}),
        "rate_limits": docs_entry.get("rate_limits", {}),
    }
    if "variant" in docs_entry:
        entry["variant"] = docs_entry["variant"]
    if "promo" in docs_entry:
        entry["promo"] = docs_entry["promo"]
    return entry


def merge_model_data(api_models, docs_data):
    """Merge API model data with docs table data into a unified list.

    Every docs row lands in the output: rows that never joined an API model
    are kept as orphans with ``matched: false`` — the join verdict belongs
    downstream.
    """
    unified = []
    for api_model in api_models:
        model_id = api_model.get("id", "")
        docs_entry = docs_data.get(model_id, {})
        unified.append(_unified_entry(model_id, api_model, docs_entry))

    api_ids = {m.get("id", "") for m in api_models}
    for model_id, docs_entry in docs_data.items():
        if model_id not in api_ids:
            unified.append(_unified_entry(model_id, {}, docs_entry))

    return unified


def _snapshot(models, warnings):
    return {
        "scraped_at": int(time.time()),
        "warnings": warnings,
        "models": models,
    }


def _snapshot_path():
    return llm.user_dir() / "opencode_unified_models.json"


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


def snapshot_age(scraped_at):
    """Age of a ``scraped_at`` timestamp in seconds, or ``None`` when the
    value is missing or not a number."""
    try:
        return time.time() - float(scraped_at)
    except (TypeError, ValueError):
        return None


def _snapshot_is_fresh(snapshot):
    age = snapshot_age(snapshot.get("scraped_at", 0))
    return age is not None and age < UNIFIED_MODELS_CACHE_TIMEOUT


def scrape_snapshot():
    """Scrape the API model list and docs page fresh, persist, and return
    the unified snapshot. Raises DocsScrapeError (docs fetch or no contract
    table matched) or DownloadError (models list fetch) on failure.
    """
    import llm_opencode as _root

    api_models = _root.get_opencode_models()
    tables = parse_html_tables(_root.fetch_docs_page())
    docs_data, warnings = parse_docs_tables(tables)
    unified = merge_model_data(api_models, docs_data)
    snapshot = _snapshot(unified, warnings)

    cache_path = _snapshot_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as file:
        json.dump(snapshot, file)

    return snapshot


def fetch_unified_models():
    """Serve the unified snapshot, refreshing it only when warranted.

    The snapshot payload is ``{"scraped_at", "warnings", "models"}``. The 24
    hour timeout gates refresh attempts only: a fresh snapshot is served as
    is, an expired-but-present snapshot gets one refresh attempt and is
    served with its age when the attempt fails, and a missing snapshot gets
    one full scrape whose failures (DocsScrapeError, DownloadError) raise.
    """
    snapshot = read_snapshot()
    if snapshot is None:
        return scrape_snapshot()
    if _snapshot_is_fresh(snapshot):
        return snapshot
    try:
        return scrape_snapshot()
    except Exception:
        return snapshot
