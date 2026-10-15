import csv
import io
import json
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from llm.cli import cli

from llm_opencode import DocsScrapeError, sort_models


@patch("llm_opencode.get_opencode_models")
def test_llm_models(mock_get_models, make_opencode_models, user_dir):
    mock_get_models.return_value = make_opencode_models("deepseek-v4-flash", "minimax-m3")
    runner = CliRunner()
    result = runner.invoke(cli, ["models", "list"])
    assert result.exit_code == 0, result.output


@patch("llm_opencode.get_opencode_models")
def test_opencode_models_cli(mock_get_models, user_dir):
    mock_get_models.return_value = [
        {"id": "glm-5"},
        {"id": "minimax-m3"},
    ]

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "models"])
    assert result.exit_code == 0, result.output
    assert "glm-5" in result.output
    assert "minimax-m3" in result.output
    assert "unknown (no scrape data)" in result.output


@patch("llm_opencode.get_opencode_models")
def test_opencode_models_cli_does_not_fabricate_protocols(mock_get_models, user_dir):
    mock_get_models.return_value = [
        {"id": "gpt-6-luna"},
        {"id": "minimax-m3"},
    ]

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "models"])
    assert result.exit_code == 0, result.output
    assert "openai" not in result.output
    assert "anthropic" not in result.output
    assert "/v1/responses" not in result.output


@patch("llm_opencode.get_opencode_models")
def test_opencode_models_cli_json(mock_get_models, user_dir):
    mock_get_models.return_value = [
        {"id": "glm-5"},
        {"id": "minimax-m3"},
    ]

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "models", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data[0]["id"] == "glm-5"


@patch("llm_opencode.get_opencode_models")
def test_opencode_models_cli_empty(mock_get_models, user_dir):
    mock_get_models.return_value = []

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "models"])
    assert result.exit_code == 0, result.output
    assert result.output == ""


@patch("llm_opencode.get_opencode_models")
def test_opencode_models_cli_empty_json(mock_get_models, user_dir):
    mock_get_models.return_value = []

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "models", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == []


@patch("llm_opencode.get_opencode_models")
def test_opencode_default_to_models(mock_get_models, user_dir):
    mock_get_models.return_value = [{"id": "test-model"}]

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode"])
    assert result.exit_code == 0, result.output
    assert "test-model" in result.output


# ─── opencode models detail ──────────────────────────────────────────────────

def _unified_models():
    return [
        {
            "model_id": "grok-4.5",
            "model_name": "Grok 4.5",
            "provider": "opencode",
            "created": 123,
            "matched": True,
            "endpoint": "https://opencode.ai/inference/go/v1/responses",
            "ai_sdk_package": "@ai-sdk/openai",
            "model_training": "Not used",
            "data_retention": "30 days",
            "pricing": {
                "go": {
                    "input": 2.0, "output": 6.0, "cached_read": 0.3,
                    "cached_write": None, "monthly_limit": 15.0,
                    "limit_note": None, "currency": "USD",
                },
                "go_plus": {
                    "input": 2.0, "output": 6.0, "cached_read": 0.3,
                    "cached_write": None, "monthly_limit": 45.0,
                    "limit_note": None, "currency": "USD",
                },
            },
            "rate_limits": {
                "go": {"per_5_hours": "120", "per_week": "300", "per_month": "600"},
                "go_plus": {"per_5_hours": "360", "per_week": "900", "per_month": "1800"},
            },
        },
        {
            "model_id": "minimax-m3",
            "model_name": "MiniMax M3",
            "provider": "opencode",
            "created": 456,
            "matched": True,
            "endpoint": "https://opencode.ai/inference/go/v1/messages",
            "ai_sdk_package": "@ai-sdk/anthropic",
            "model_training": "Not used",
            "data_retention": "60 days",
            "pricing": {
                "go": {
                    "input": 0.3, "output": 1.2, "cached_read": 0.06,
                    "cached_write": None, "monthly_limit": 60.0,
                    "limit_note": "limited time", "currency": "USD",
                },
                "go_plus": {
                    "input": 0.3, "output": 1.2, "cached_read": 0.06,
                    "cached_write": None, "monthly_limit": 180.0,
                    "limit_note": "limited time", "currency": "USD",
                },
            },
            "rate_limits": {
                "go": {"per_5_hours": "3200", "per_week": "8000", "per_month": "16000"},
                "go_plus": {"per_5_hours": "9600", "per_week": "24000", "per_month": "48000"},
            },
        },
    ]


def _snapshot(models):
    return {"scraped_at": 1700000000, "warnings": [], "models": models}


@patch("llm_opencode.fetch_unified_models")
def test_opencode_models_detail_default_table(mock_fetch):
    mock_fetch.return_value = _snapshot(_unified_models())

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "models", "detail"])
    assert result.exit_code == 0, result.output
    assert "Model ID" in result.output
    assert "Protocol" in result.output
    assert "openai-responses" in result.output
    assert "anthropic" in result.output
    assert "grok-4.5" in result.output
    assert "minimax-m3" in result.output
    assert "$2.00" in result.output
    assert "30 days" in result.output
    assert "Monthly Cap" in result.output
    assert "$15.00" in result.output
    assert "$60.00" in result.output
    assert "Req/Month" in result.output
    assert "600" in result.output
    assert "16,000" in result.output
    assert result.output.index("Req/Month") < result.output.index("Input")


@patch("llm_opencode.fetch_unified_models")
def test_opencode_detail_primary_command(mock_fetch):
    mock_fetch.return_value = _snapshot(_unified_models())

    runner = CliRunner()
    primary = runner.invoke(cli, ["opencode", "detail"])
    alias = runner.invoke(cli, ["opencode", "models", "detail"])
    assert primary.exit_code == 0, primary.output
    assert alias.exit_code == 0, alias.output
    assert primary.output == alias.output


@patch("llm_opencode.fetch_unified_models")
def test_opencode_detail_plan_default_shows_go(mock_fetch):
    mock_fetch.return_value = _snapshot(_unified_models())

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "detail"])
    assert result.exit_code == 0, result.output
    assert "$15.00" in result.output
    assert "$60.00" in result.output
    assert "$45.00" not in result.output
    assert "$180.00" not in result.output


@patch("llm_opencode.fetch_unified_models")
def test_opencode_detail_plan_go_plus(mock_fetch):
    mock_fetch.return_value = _snapshot(_unified_models())

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "detail", "--plan", "go_plus"])
    assert result.exit_code == 0, result.output
    assert "$45.00" in result.output
    assert "$180.00" in result.output
    assert "1,800" in result.output
    assert "48,000" in result.output
    assert "$15.00" not in result.output
    assert "$60.00" not in result.output
    # No re-scrape: the plan filter reads the plan-keyed data already in the entry.
    assert mock_fetch.call_count == 1


@patch("llm_opencode.fetch_unified_models")
def test_opencode_detail_plan_all_renders_plan_labelled_rows(mock_fetch):
    mock_fetch.return_value = _snapshot(_unified_models())

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "detail", "--plan", "all"])
    assert result.exit_code == 0, result.output
    assert "Plan" in result.output
    # Two plan-labelled rows per model.
    assert result.output.count("minimax-m3") == 2
    assert result.output.count("grok-4.5") == 2
    assert "$15.00" in result.output
    assert "$45.00" in result.output


@patch("llm_opencode.fetch_unified_models")
def test_opencode_detail_plan_all_csv_plan_labelled_rows(mock_fetch):
    mock_fetch.return_value = _snapshot(_unified_models())

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "detail", "--plan", "all", "--format", "csv"])
    assert result.exit_code == 0, result.output

    rows = [row for row in csv.reader(io.StringIO(result.output)) if row]
    assert len(rows) == 5  # header + two rows per model
    per_model = {}
    for row in rows[1:]:
        per_model.setdefault(row[0], set()).add(row[2])
    assert per_model == {"grok-4.5": {"go", "go_plus"}, "minimax-m3": {"go", "go_plus"}}


@patch("llm_opencode.fetch_unified_models")
def test_opencode_plan_all_csv_rows_carry_their_own_plan_data(mock_fetch):
    mock_fetch.return_value = _snapshot(_unified_models())

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "detail", "--plan", "all", "--format", "csv"])
    assert result.exit_code == 0, result.output

    rows = [row for row in csv.reader(io.StringIO(result.output)) if row]
    by_key = {(row[0], row[2]): row for row in rows[1:]}
    grok_go = by_key[("grok-4.5", "go")]
    grok_plus = by_key[("grok-4.5", "go_plus")]
    assert grok_go[9] == "15.0"
    assert grok_plus[9] == "45.0"
    assert grok_go[17] == "600"
    assert grok_plus[17] == "1800"


@patch("llm_opencode.fetch_unified_models")
def test_opencode_detail_json_unaffected_by_plan_filter(mock_fetch):
    mock_fetch.return_value = _snapshot(_unified_models())

    runner = CliRunner()
    result = runner.invoke(
        cli, ["opencode", "detail", "--plan", "go_plus", "--format", "json"]
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["count"] == 2
    for model in data["models"]:
        assert set(model["pricing"]) == {"go", "go_plus"}
        assert set(model["rate_limits"]) == {"go", "go_plus"}


@patch("llm_opencode.fetch_unified_models")
def test_opencode_models_detail_default_sort_requests(mock_fetch):
    mock_fetch.return_value = _snapshot(_unified_models())

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "models", "detail"])
    assert result.exit_code == 0, result.output
    assert result.output.index("minimax-m3") < result.output.index("grok-4.5")


@patch("llm_opencode.fetch_unified_models")
def test_opencode_models_detail_sort_by_monthly_limit(mock_fetch):
    mock_fetch.return_value = _snapshot(_unified_models())

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "detail", "--sort-by", "monthly-limit"])
    assert result.exit_code == 0, result.output
    assert result.output.index("grok-4.5") < result.output.index("minimax-m3")


@patch("llm_opencode.fetch_unified_models")
def test_opencode_models_detail_sort_by_usage_removed(mock_fetch):
    mock_fetch.return_value = _snapshot(_unified_models())

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "detail", "--sort-by", "usage"])
    assert result.exit_code == 2
    assert "is not one of" in result.output


@patch("llm_opencode.fetch_unified_models")
def test_opencode_models_detail_sort_by_name(mock_fetch):
    mock_fetch.return_value = _snapshot(_unified_models())

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "models", "detail", "--sort-by", "name"])
    assert result.exit_code == 0, result.output
    assert result.output.index("grok-4.5") < result.output.index("minimax-m3")


@patch("llm_opencode.fetch_unified_models")
def test_opencode_models_detail_requests_missing_pinned_last(mock_fetch):
    models = _unified_models()
    models.append(
        {
            "model_id": "hy3-preview",
            "model_name": "Hy3 Preview",
            "provider": "opencode",
            "created": 789,
            "matched": True,
            "endpoint": "",
            "ai_sdk_package": "",
            "model_training": "",
            "data_retention": "",
            "pricing": {
                "go": {
                    "input": 0.1, "output": 0.2, "cached_read": None,
                    "cached_write": None, "monthly_limit": None,
                    "limit_note": None, "currency": "USD",
                }
            },
            "rate_limits": {"go": {"per_5_hours": "", "per_week": "", "per_month": ""}},
        }
    )
    mock_fetch.return_value = _snapshot(models)

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "models", "detail"])
    assert result.exit_code == 0, result.output
    assert result.output.index("hy3-preview") > result.output.index("minimax-m3")


@patch("llm_opencode.fetch_unified_models")
def test_opencode_models_detail_requests_sort_order(mock_fetch):
    models = [
        {
            "model_id": "model-a",
            "model_name": "Model A",
            "pricing": {"go": {"input": 1.0, "monthly_limit": 15.0}},
            "rate_limits": {"go": {"per_month": "3000"}},
        },
        {
            "model_id": "model-b",
            "model_name": "Model B",
            "pricing": {"go": {"input": 0.5, "monthly_limit": 20.0}},
            "rate_limits": {"go": {"per_month": "4000"}},
        },
    ]
    mock_fetch.return_value = _snapshot(models)

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "models", "detail"])
    assert result.exit_code == 0, result.output
    assert result.output.index("model-b") < result.output.index("model-a")


@patch("llm_opencode.fetch_unified_models")
def test_opencode_models_detail_json(mock_fetch):
    mock_fetch.return_value = _snapshot(_unified_models())

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "models", "detail", "--format", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["count"] == 2
    assert [m["model_id"] for m in data["models"]] == ["minimax-m3", "grok-4.5"]
    assert data["models"][0]["pricing"]["go"]["input"] == 0.3
    assert data["models"][0]["pricing"]["go_plus"]["monthly_limit"] == 180.0
    assert data["models"][0]["matched"] is True
    assert "sources" in data["meta"]
    assert data["meta"]["scraped_at"] == 1700000000
    assert data["meta"]["warnings"] == []


@patch("llm_opencode.fetch_unified_models")
def test_opencode_models_detail_json_surfaces_scrape_warnings(mock_fetch):
    mock_fetch.return_value = _snapshot(
        [{"model_id": "grok-4.5", "pricing": {"go": {}}}]
    )
    mock_fetch.return_value["warnings"] = [
        "table 3: unrecognized table shape: ['model', 'fallbacks']"
    ]

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "models", "detail", "--format", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["meta"]["warnings"] == [
        "table 3: unrecognized table shape: ['model', 'fallbacks']"
    ]


@patch("llm_opencode.fetch_unified_models")
def test_opencode_models_detail_json_unregistered_section(mock_fetch):
    models = _unified_models() + [
        {
            "model_id": "omen-alpha",
            "model_name": "Omen Alpha",
            "provider": "opencode",
            "created": 999,
            "matched": False,
            "endpoint": "",
            "ai_sdk_package": "",
            "model_training": "",
            "data_retention": "",
            "pricing": {},
            "rate_limits": {},
        },
        {
            "model_id": "gemini-4-argon-contributor",
            "model_name": "Gemini 4 Argon Contributor",
            "provider": "opencode",
            "created": 1000,
            "matched": True,
            "endpoint": "https://opencode.ai/inference/go/v1/models/gemini-4-argon-contributor",
            "ai_sdk_package": "@ai-sdk/google",
            "model_training": "",
            "data_retention": "",
            "pricing": {},
            "rate_limits": {},
        },
    ]
    mock_fetch.return_value = _snapshot(models)

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "models", "detail", "--format", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["count"] == 4
    assert [
        (u["model_id"], u["skip_reason"], u["protocol"])
        for u in data["unregistered"]
    ] == [
        ("omen-alpha", "protocol unknown: no docs row", None),
        (
            "gemini-4-argon-contributor",
            "unsupported protocol: google-style",
            None,
        ),
    ]
    by_id = {m["model_id"]: m for m in data["models"]}
    assert by_id["grok-4.5"]["protocol"] == "openai-responses"
    assert by_id["grok-4.5"]["registered"] is True
    assert by_id["omen-alpha"]["registered"] is False
    assert by_id["omen-alpha"]["skip_reason"] == "protocol unknown: no docs row"


@patch("llm_opencode.fetch_unified_models")
def test_opencode_models_detail_json_package_mismatch_warns_not_vetoes(mock_fetch):
    models = _unified_models()
    models[0]["ai_sdk_package"] = "@ai-sdk/google"
    mock_fetch.return_value = _snapshot(models)

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "models", "detail", "--format", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["meta"]["warnings"] == [
        'protocol corroboration: grok-4.5 endpoint path resolves openai-responses '
        'but ai_sdk_package is "@ai-sdk/google"'
    ]
    assert data["unregistered"] == []
    by_id = {m["model_id"]: m for m in data["models"]}
    assert by_id["grok-4.5"]["registered"] is True
    assert by_id["grok-4.5"]["protocol"] == "openai-responses"


@patch("llm_opencode.fetch_unified_models")
def test_opencode_models_detail_csv(mock_fetch):
    mock_fetch.return_value = _snapshot(_unified_models())

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "detail", "--format", "csv"])
    assert result.exit_code == 0, result.output

    rows = list(csv.reader(io.StringIO(result.output)))
    assert rows[0] == [
        "model_id", "model_name", "plan", "protocol", "matched",
        "input_price", "output_price", "cached_read", "cached_write",
        "monthly_limit", "limit_note",
        "endpoint", "ai_sdk_package", "data_retention", "model_training",
        "rate_per_5h", "rate_per_week", "rate_per_month",
    ]
    assert "provider" not in rows[0]
    assert rows[1][0] == "minimax-m3"
    assert rows[1][2] == "go"
    assert rows[1][3] == "anthropic"
    assert rows[1][5] == "0.3"
    assert rows[1][9] == "60.0"
    assert rows[1][10] == "limited time"


@patch("llm_opencode.fetch_unified_models")
def test_opencode_models_detail_csv_protocol_for_unknown_model(mock_fetch):
    models = _unified_models() + [
        {
            "model_id": "omen-alpha",
            "model_name": "Omen Alpha",
            "provider": "opencode",
            "created": 999,
            "matched": False,
            "endpoint": "",
            "ai_sdk_package": "",
            "model_training": "",
            "data_retention": "",
            "pricing": {},
            "rate_limits": {},
        },
    ]
    mock_fetch.return_value = _snapshot(models)

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "detail", "--format", "csv"])
    assert result.exit_code == 0, result.output

    rows = list(csv.reader(io.StringIO(result.output)))
    omen = next(row for row in rows[1:] if row[0] == "omen-alpha")
    assert omen[3] == ""  # protocol unknown, rendered without a crash


@patch("llm_opencode.fetch_unified_models")
def test_opencode_models_detail_table_unregistered_section(mock_fetch):
    models = _unified_models() + [
        {
            "model_id": "omen-alpha",
            "model_name": "Omen Alpha",
            "provider": "opencode",
            "created": 999,
            "matched": False,
            "endpoint": "",
            "ai_sdk_package": "",
            "model_training": "",
            "data_retention": "",
            "pricing": {},
            "rate_limits": {},
        },
        {
            "model_id": "gemini-4-argon-contributor",
            "model_name": "Gemini 4 Argon Contributor",
            "provider": "opencode",
            "created": 1000,
            "matched": True,
            "endpoint": "https://opencode.ai/inference/go/v1/models/gemini-4-argon-contributor",
            "ai_sdk_package": "@ai-sdk/google",
            "model_training": "",
            "data_retention": "",
            "pricing": {},
            "rate_limits": {},
        },
    ]
    mock_fetch.return_value = _snapshot(models)

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "detail"])
    assert result.exit_code == 0, result.output
    assert "Unregistered:" in result.output
    assert "omen-alpha — protocol unknown: no docs row" in result.output
    assert (
        "gemini-4-argon-contributor — unsupported protocol: google-style"
        in result.output
    )
    # Graceful protocol rendering for unregistered models.
    assert result.output.count("unknown") >= 2


@patch("llm_opencode.fetch_unified_models")
def test_opencode_models_detail_pricing_plan_go_plus(mock_fetch):
    mock_fetch.return_value = _snapshot(_unified_models())

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "detail", "--plan", "go_plus", "--format", "pricing"])
    assert result.exit_code == 0, result.output
    assert "$45.00" in result.output
    assert "$180.00" in result.output
    assert "$15.00" not in result.output
    assert "$60.00" not in result.output


@patch("llm_opencode.fetch_unified_models")
def test_opencode_models_detail_pricing_plan_all_labels_plans(mock_fetch):
    mock_fetch.return_value = _snapshot(_unified_models())

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "detail", "--plan", "all", "--format", "pricing"])
    assert result.exit_code == 0, result.output
    assert "grok-4.5 (go)" in result.output
    assert "grok-4.5 (go_plus)" in result.output
    assert "minimax-m3 (go)" in result.output
    assert "minimax-m3 (go_plus)" in result.output


@patch("llm_opencode.fetch_unified_models")
def test_opencode_models_detail_pricing_sorted_by_input(mock_fetch):
    mock_fetch.return_value = _snapshot(_unified_models())

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "models", "detail", "--format", "pricing"])
    assert result.exit_code == 0, result.output
    assert result.output.index("minimax-m3") < result.output.index("grok-4.5")


@patch("llm_opencode.fetch_unified_models")
def test_opencode_models_detail_search_filter(mock_fetch):
    mock_fetch.return_value = _snapshot(_unified_models())

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "models", "detail", "--search", "grok"])
    assert result.exit_code == 0, result.output
    assert "grok-4.5" in result.output
    assert "minimax-m3" not in result.output


@patch("llm_opencode.fetch_unified_models")
def test_opencode_models_detail_search_matches_name(mock_fetch):
    mock_fetch.return_value = _snapshot(_unified_models())

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "models", "detail", "--search", "minimax"])
    assert result.exit_code == 0, result.output
    assert "minimax-m3" in result.output
    assert "grok-4.5" not in result.output


@patch("llm_opencode.fetch_unified_models")
def test_opencode_models_detail_provider_filter_removed(mock_fetch):
    mock_fetch.return_value = _snapshot(_unified_models())

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "detail", "--provider", "opencode"])
    assert result.exit_code == 2
    assert "No such option" in result.output


@patch("llm_opencode.fetch_unified_models")
def test_opencode_models_detail_model_exact_filter(mock_fetch):
    mock_fetch.return_value = _snapshot(_unified_models())

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "models", "detail", "--model", "grok-4.5"])
    assert result.exit_code == 0, result.output
    assert "grok-4.5" in result.output
    assert "minimax-m3" not in result.output


@patch("llm_opencode.fetch_unified_models")
def test_opencode_models_detail_empty(mock_fetch):
    mock_fetch.return_value = _snapshot([])

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "models", "detail"])
    assert result.exit_code == 0, result.output
    assert "No models found." in result.output


@patch("llm_opencode.fetch_unified_models")
def test_opencode_models_detail_pricing_no_priced_models(mock_fetch):
    mock_fetch.return_value = _snapshot(
        [{"model_id": "grok-4.5", "pricing": {"go": {"input": None}}}]
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "models", "detail", "--format", "pricing"])
    assert result.exit_code == 0, result.output
    assert "No pricing data found." in result.output


@patch("llm_opencode.fetch_unified_models")
def test_opencode_models_detail_pricing_missing_values(mock_fetch):
    mock_fetch.return_value = _snapshot(
        [
            {
                "model_name": "Mystery Model",
                "pricing": {
                    "go": {
                        "input": 1.0, "output": 2.0, "cached_read": None,
                        "cached_write": None, "monthly_limit": None,
                        "limit_note": None,
                    },
                },
            }
        ]
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "models", "detail", "--format", "pricing"])
    assert result.exit_code == 0, result.output
    assert "Mystery Model" in result.output
    assert "Cached Read:  —" in result.output
    assert "Cached Write: —" in result.output


@patch("llm_opencode.fetch_unified_models")
def test_opencode_models_detail_pricing_cached_write(mock_fetch):
    mock_fetch.return_value = _snapshot(
        [
            {
                "model_id": "grok-4.5",
                "pricing": {
                    "go": {
                        "input": 2.0, "output": 6.0, "cached_read": 0.3,
                        "cached_write": 1.5, "monthly_limit": 15.0,
                        "limit_note": None,
                    },
                },
            }
        ]
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "models", "detail", "--format", "pricing"])
    assert result.exit_code == 0, result.output
    assert "Cached Write: $1.50" in result.output
    assert "Monthly cap:" in result.output


@patch("llm_opencode.fetch_unified_models")
def test_opencode_models_detail_pricing_free_note(mock_fetch):
    mock_fetch.return_value = _snapshot(
        [
            {
                "model_id": "longcat-2.5-preview-free",
                "pricing": {
                    "go": {
                        "input": 0.0, "output": 0.0, "cached_read": 0.0,
                        "cached_write": None, "monthly_limit": None,
                        "limit_note": None, "pricing_note": "free",
                    },
                },
            }
        ]
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "models", "detail", "--format", "pricing"])
    assert result.exit_code == 0, result.output
    assert "$0.00 (free)" in result.output


@patch("llm_opencode.fetch_unified_models")
def test_opencode_models_detail_output_file(mock_fetch, tmp_path):
    mock_fetch.return_value = _snapshot(_unified_models())
    output_file = tmp_path / "models.csv"

    runner = CliRunner()
    result = runner.invoke(
        cli, ["opencode", "models", "detail", "--format", "csv", "-o", str(output_file)]
    )
    assert result.exit_code == 0, result.output
    assert output_file.exists()
    rows = list(csv.reader(io.StringIO(output_file.read_text())))
    assert rows[1][0] == "minimax-m3"


@patch("llm_opencode.fetch_unified_models", side_effect=DocsScrapeError("boom"))
def test_opencode_models_detail_scrape_failure_shows_docs_link(mock_fetch):
    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "models", "detail"])
    assert result.exit_code == 1
    assert "scraping failed" in result.output
    assert "https://opencode.ai/docs/go/" in result.output


# ─── sort_models ─────────────────────────────────────────────────────────────

def _model(model_id, name, input_price, monthly_limit, per_month):
    return {
        "model_id": model_id,
        "model_name": name,
        "pricing": {
            "go": {
                "input": input_price, "output": 1.0, "cached_read": None,
                "cached_write": None, "monthly_limit": monthly_limit,
            },
        },
        "rate_limits": {"go": {"per_month": str(per_month)}},
    }


def test_sort_models_requests_default():
    models = [
        _model("a", "A", 1.0, 15.0, 300),
        _model("b", "B", 2.0, 60.0, 16000),
        _model("c", "C", 0.5, None, 0),
    ]
    result = sort_models(models, sort_by="requests")
    assert [m["model_id"] for m in result] == ["b", "a", "c"]


def test_sort_models_requests_tiebreak_input():
    models = [
        _model("a", "A", 1.0, 15.0, 4000),
        _model("b", "B", 0.5, 20.0, 4000),
    ]
    result = sort_models(models, sort_by="requests")
    assert [m["model_id"] for m in result] == ["b", "a"]


def test_sort_models_requests_invalid_rate_limit_pinned_last():
    models = [
        _model("a", "A", 1.0, 15.0, 300),
        {
            "model_id": "b",
            "model_name": "B",
            "pricing": {"go": {"input": 0.5, "monthly_limit": 15.0}},
            "rate_limits": {"go": {"per_month": "not-a-number"}},
        },
    ]
    result = sort_models(models, sort_by="requests")
    assert [m["model_id"] for m in result] == ["a", "b"]


def test_sort_models_by_input():
    models = [
        _model("a", "A", 2.0, 15.0, 300),
        _model("b", "B", 0.5, 15.0, 300),
    ]
    result = sort_models(models, sort_by="input")
    assert [m["model_id"] for m in result] == ["b", "a"]


def test_sort_models_by_input_missing_last():
    models = [
        _model("a", "A", None, 15.0, 300),
        _model("b", "B", 2.0, 15.0, 300),
    ]
    result = sort_models(models, sort_by="input")
    assert [m["model_id"] for m in result] == ["b", "a"]


def test_sort_models_by_output():
    models = [
        _model("a", "A", 1.0, 15.0, 300),
        _model("b", "B", 1.0, 15.0, 300),
    ]
    models[0]["pricing"]["go"]["output"] = 5.0
    models[1]["pricing"]["go"]["output"] = 1.0
    result = sort_models(models, sort_by="output")
    assert [m["model_id"] for m in result] == ["b", "a"]


def test_sort_models_by_monthly_limit():
    models = [
        _model("a", "A", 1.0, 60.0, 300),
        _model("b", "B", 1.0, 15.0, 300),
    ]
    result = sort_models(models, sort_by="monthly-limit")
    assert [m["model_id"] for m in result] == ["b", "a"]


def test_sort_models_monthly_limit_missing_pinned_last():
    models = [
        _model("a", "A", 1.0, None, 300),
        _model("b", "B", 1.0, 15.0, 300),
    ]
    result = sort_models(models, sort_by="monthly-limit")
    assert [m["model_id"] for m in result] == ["b", "a"]


def test_sort_models_usage_removed():
    models = [_model("a", "A", 1.0, 15.0, 300)]
    with pytest.raises(KeyError):
        sort_models(models, sort_by="usage")


def test_sort_models_respects_plan_filter():
    models = [
        _model("a", "A", 1.0, 15.0, 300),
        _model("b", "B", 1.0, 60.0, 300),
    ]
    for model in models:
        model["pricing"]["go_plus"] = {
            "input": 1.0, "output": 1.0, "cached_read": None,
            "cached_write": None, "monthly_limit": 200.0,
        }
    models[0]["pricing"]["go_plus"]["monthly_limit"] = 5.0
    result = sort_models(models, sort_by="monthly-limit", plan="go_plus")
    assert [m["model_id"] for m in result] == ["a", "b"]


def test_sort_models_by_name():
    models = [
        _model("b", "Beta", 1.0, 15.0, 300),
        _model("a", "Alpha", 1.0, 15.0, 300),
    ]
    result = sort_models(models, sort_by="name")
    assert [m["model_id"] for m in result] == ["a", "b"]


def test_sort_models_by_model_id():
    models = [
        _model("b", "B", 1.0, 15.0, 300),
        _model("a", "A", 1.0, 15.0, 300),
    ]
    result = sort_models(models, sort_by="model_id")
    assert [m["model_id"] for m in result] == ["a", "b"]



def test_sort_models_empty():
    assert sort_models([]) == []


def test_format_unified_table_requests_missing_shows_dash():
    from llm_opencode import format_unified_table

    model = _model("a", "A", 0.5, 15.0, 0)
    model["pricing"]["go"]["cached_read"] = 0.1
    model["pricing"]["go"]["cached_write"] = 0.2
    model["data_retention"] = "30 days"

    result = format_unified_table([model])

    assert result.count("—") == 1


def test_plan_helpers_tolerate_missing_or_non_dict_blocks():
    from llm_opencode.cli import _plan_pricing, _plan_rate_limits

    assert _plan_pricing({"pricing": "flat"}) == {}
    assert _plan_rate_limits({}) == {}
    block = {"input": 1.0, "monthly_limit": 10.0}
    assert _plan_pricing({"pricing": None}) == {}
    assert _plan_pricing({"pricing": {"go_plus": block}}) == block
    assert _plan_rate_limits({"rate_limits": {"go_plus": {"per_month": "9"}}}) == {
        "per_month": "9"
    }


def test_format_unified_table_tolerates_non_dict_pricing():
    from llm_opencode import format_unified_table

    result = format_unified_table([{"model_id": "x", "pricing": None}])
    assert "x" in result
