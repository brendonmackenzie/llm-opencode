import builtins
import json
import os
from unittest.mock import MagicMock, patch

import httpx
import pytest

from llm_opencode import DownloadError, fetch_cached_json


def test_fetch_cached_json_cache_hit(tmp_path):
    cache_file = tmp_path / "cache.json"
    cache_data = {"data": [{"id": "model-1"}]}
    cache_file.write_text(json.dumps(cache_data))

    result = fetch_cached_json("https://example.com/api", cache_file, 3600)
    assert result == cache_data


@patch("httpx.get")
def test_fetch_cached_json_network_fetch(mock_httpx_get, tmp_path):
    mock_response = MagicMock()
    mock_response.json.return_value = {"data": [{"id": "model-1"}]}
    mock_response.raise_for_status.return_value = None
    mock_httpx_get.return_value = mock_response

    cache_file = tmp_path / "cache.json"

    result = fetch_cached_json("https://example.com/api", cache_file, 3600)
    assert result == {"data": [{"id": "model-1"}]}
    assert cache_file.exists()


@patch("httpx.get")
def test_fetch_cached_json_stale_cache_network_success(mock_httpx_get, tmp_path):
    cache_file = tmp_path / "cache.json"
    old_data = {"data": [{"id": "old-model"}]}
    cache_file.write_text(json.dumps(old_data))
    os.utime(cache_file, (0, 0))

    new_data = {"data": [{"id": "new-model"}]}
    mock_response = MagicMock()
    mock_response.json.return_value = new_data
    mock_response.raise_for_status.return_value = None
    mock_httpx_get.return_value = mock_response

    result = fetch_cached_json("https://example.com/api", cache_file, 3600)
    assert result == new_data
    assert json.loads(cache_file.read_text()) == new_data


@patch("httpx.get")
def test_fetch_cached_json_http_error_with_cache(mock_httpx_get, tmp_path):
    cache_file = tmp_path / "cache.json"
    cache_data = {"data": [{"id": "model-1"}]}
    cache_file.write_text(json.dumps(cache_data))
    os.utime(cache_file, (0, 0))

    mock_httpx_get.side_effect = httpx.HTTPError("Connection error")

    result = fetch_cached_json("https://example.com/api", cache_file, 3600)
    assert result == cache_data


@patch("httpx.get")
def test_fetch_cached_json_stale_cache_network_failure(mock_httpx_get, tmp_path):
    cache_file = tmp_path / "cache.json"
    stale_data = {"data": [{"id": "stale-model"}]}
    cache_file.write_text(json.dumps(stale_data))
    os.utime(cache_file, (0, 0))

    mock_httpx_get.side_effect = httpx.HTTPError("Connection error")

    result = fetch_cached_json("https://example.com/api", cache_file, 3600)
    assert result == stale_data


@patch("httpx.get")
def test_fetch_cached_json_http_error_no_cache(mock_httpx_get, tmp_path):
    cache_file = tmp_path / "cache.json"

    mock_httpx_get.side_effect = httpx.HTTPError("Connection error")

    with pytest.raises(DownloadError):
        fetch_cached_json("https://example.com/api", cache_file, 3600)


@patch("httpx.get")
def test_fetch_cached_json_invalid_json_cache(mock_httpx_get, tmp_path):
    cache_file = tmp_path / "cache.json"
    cache_file.write_text("not valid json")

    mock_httpx_get.side_effect = httpx.HTTPError("Connection error")

    with pytest.raises(DownloadError):
        fetch_cached_json("https://example.com/api", cache_file, 3600)


@patch("httpx.get")
def test_fetch_cached_json_fresh_cache_invalid_json_falls_back_to_network(
    mock_httpx_get, tmp_path
):
    cache_file = tmp_path / "cache.json"
    cache_file.write_text("not valid json")

    new_data = {"data": [{"id": "new-model"}]}
    mock_response = MagicMock()
    mock_response.json.return_value = new_data
    mock_response.raise_for_status.return_value = None
    mock_httpx_get.return_value = mock_response

    result = fetch_cached_json("https://example.com/api", cache_file, 3600)
    assert result == new_data
    assert json.loads(cache_file.read_text()) == new_data


@patch("httpx.get")
def test_fetch_cached_json_fresh_cache_oserror_falls_back_to_network(
    mock_httpx_get, tmp_path
):
    cache_file = tmp_path / "cache.json"
    cache_file.write_text(json.dumps({"data": [{"id": "old"}]}))

    new_data = {"data": [{"id": "new"}]}
    mock_response = MagicMock()
    mock_response.json.return_value = new_data
    mock_response.raise_for_status.return_value = None
    mock_httpx_get.return_value = mock_response

    real_open = builtins.open

    def fake_open(file, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if "r" in mode:
            raise PermissionError(f"[Errno 13] Permission denied: {file}")
        return real_open(file, *args, **kwargs)

    with patch("builtins.open", side_effect=fake_open):
        result = fetch_cached_json("https://example.com/api", cache_file, 3600)

    assert result == new_data
    assert json.loads(cache_file.read_text()) == new_data


@patch("httpx.get")
def test_fetch_cached_json_stale_cache_oserror_raises_download_error(
    mock_httpx_get, tmp_path
):
    cache_file = tmp_path / "cache.json"
    cache_file.write_text(json.dumps({"data": [{"id": "stale"}]}))
    os.utime(cache_file, (0, 0))

    mock_httpx_get.side_effect = httpx.HTTPError("Connection error")

    real_open = builtins.open

    def fake_open(file, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if "r" in mode:
            raise PermissionError(f"[Errno 13] Permission denied: {file}")
        return real_open(file, *args, **kwargs)

    with patch("builtins.open", side_effect=fake_open):
        with pytest.raises(DownloadError):
            fetch_cached_json("https://example.com/api", cache_file, 3600)
