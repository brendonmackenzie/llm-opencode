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


_DEFAULT_PRICING = {
    "input": None,
    "output": None,
    "cached_read": None,
    "cached_write": None,
    "usage": None,
    "currency": "USD",
}

_DEFAULT_RATE_LIMITS = {"per_5_hours": "", "per_week": "", "per_month": ""}


def _unified_entry(model_id, api_model, docs_entry):
    return {
        "model_id": model_id,
        "model_name": docs_entry.get("model_name", ""),
        "provider": api_model.get("owned_by", ""),
        "created": api_model.get("created"),
        "endpoint": docs_entry.get("endpoint", ""),
        "ai_sdk_package": docs_entry.get("ai_sdk_package", ""),
        "model_training": docs_entry.get("model_training", ""),
        "data_retention": docs_entry.get("data_retention", ""),
        "pricing": docs_entry.get("pricing", dict(_DEFAULT_PRICING)),
        "rate_limits": docs_entry.get("rate_limits", dict(_DEFAULT_RATE_LIMITS)),
    }


def merge_model_data(api_models, docs_data):
    """Merge API model data with docs table data into a unified list."""
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


def fetch_unified_models():
    """Fetch API and docs model data, merged into a unified list.

    Results are cached for 24 hours. Raises DocsScrapeError if the docs
    page cannot be fetched or parsed.
    """
    import llm_opencode as _root

    cache_path = llm.user_dir() / "opencode_unified_models.json"
    if (
        cache_path.is_file()
        and time.time() - cache_path.stat().st_mtime < UNIFIED_MODELS_CACHE_TIMEOUT
    ):
        try:
            with open(cache_path) as file:
                return json.load(file)
        except (json.JSONDecodeError, OSError):
            pass

    api_models = _root.get_opencode_models()
    tables = parse_html_tables(_root.fetch_docs_page())
    if not tables:
        raise DocsScrapeError("No tables found on the OpenCode Go docs page")
    docs_data = parse_docs_tables(tables)
    unified = merge_model_data(api_models, docs_data)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as file:
        json.dump(unified, file)

    return unified
