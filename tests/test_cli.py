import csv
import io
import json
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from llm.cli import cli

from llm_opencode import MODELS_URL, DownloadError, sort_models
from tests._payload import (
    anthropic_api_model,
    api_model,
    entry,
    free_api_model,
    google_api_model,
    responses_api_model,
    snapshot,
)


@patch("llm_opencode.get_opencode_models")
def test_llm_models(mock_get_models, make_opencode_models, user_dir):
    mock_get_models.return_value = make_opencode_models("deepseek-v4-flash", "minimax-m3")
    runner = CliRunner()
    result = runner.invoke(cli, ["models", "list"])
    assert result.exit_code == 0, result.output


def test_opencode_models_listing_removed(user_dir):
    """The offline listing and its alias are gone; detail is the sole command."""
    runner = CliRunner()
    for args in (["opencode", "models"], ["opencode", "models", "detail"]):
        result = runner.invoke(cli, args)
        assert result.exit_code == 2, result.output
        assert "No such command" in result.output


def test_opencode_bare_shows_help(user_dir):
    runner = CliRunner()
    result = runner.invoke(cli, ["opencode"])

    assert result.exit_code == 0, result.output
    assert "detail" in result.output  # the one command it carries


# ─── opencode detail ─────────────────────────────────────────────────────────


def _unified_models():
    return [
        entry(api_model()),
        entry(anthropic_api_model()),
        entry(responses_api_model()),
    ]


def _payload(models, fetched_at=1700000000):
    return snapshot(*models, fetched_at=fetched_at)


@patch("llm_opencode.fetch_models_snapshot")
def test_opencode_models_detail_default_table(mock_fetch):
    mock_fetch.return_value = _payload(_unified_models())

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "detail"])
    assert result.exit_code == 0, result.output
    assert "Model ID" in result.output
    assert "Protocol" in result.output
    assert "Usage/Mo" in result.output
    assert "openai" in result.output
    assert "anthropic" in result.output
    assert "openai-responses" in result.output
    assert "glm-5.3" in result.output
    assert "claude-haiku-5-5" in result.output
    assert "gpt-6-luna" in result.output
    assert "$1.40" in result.output
    assert "$500" in result.output
    assert "1M" in result.output
    assert "Unregistered:" not in result.output


@patch("llm_opencode.fetch_models_snapshot")
def test_opencode_detail_primary_command(mock_fetch):
    mock_fetch.return_value = _payload(_unified_models())

    runner = CliRunner()
    primary = runner.invoke(cli, ["opencode", "detail"])
    assert primary.exit_code == 0, primary.output


@patch("llm_opencode.fetch_models_snapshot")
def test_opencode_detail_plan_default_shows_go(mock_fetch):
    mock_fetch.return_value = _payload(_unified_models())

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "detail"])
    assert result.exit_code == 0, result.output
    assert "$500" in result.output  # glm-5.3 go
    assert "$1000" not in result.output  # go-plus
    assert "$505" not in result.output  # go-teams


@patch("llm_opencode.fetch_models_snapshot")
def test_opencode_detail_plan_go_plus(mock_fetch):
    mock_fetch.return_value = _payload(_unified_models())

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "detail", "--plan", "go-plus"])
    assert result.exit_code == 0, result.output
    assert "$1000" in result.output
    assert "$500" not in result.output
    # No refetch: the plan filter reads the plan-keyed data already in the entry.
    assert mock_fetch.call_count == 1


@patch("llm_opencode.fetch_models_snapshot")
def test_opencode_detail_plan_go_teams(mock_fetch):
    mock_fetch.return_value = _payload(_unified_models())

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "detail", "--plan", "go-teams"])
    assert result.exit_code == 0, result.output
    assert "$505" in result.output
    assert "$500" not in result.output


@patch("llm_opencode.fetch_models_snapshot")
def test_opencode_detail_plan_all_renders_plan_labelled_rows(mock_fetch):
    mock_fetch.return_value = _payload(_unified_models())

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "detail", "--plan", "all"])
    assert result.exit_code == 0, result.output
    assert "Plan" in result.output
    # Three plan-labelled rows per model.
    assert result.output.count("glm-5.3") == 3
    assert result.output.count("claude-haiku-5-5") == 3
    assert "$500" in result.output
    assert "$1000" in result.output


@patch("llm_opencode.fetch_models_snapshot")
def test_opencode_detail_plan_all_csv_plan_labelled_rows(mock_fetch):
    mock_fetch.return_value = _payload(_unified_models())

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "detail", "--plan", "all", "--format", "csv"])
    assert result.exit_code == 0, result.output

    rows = [row for row in csv.reader(io.StringIO(result.output)) if row]
    assert len(rows) == 10  # header + three rows per model
    per_model = {}
    for row in rows[1:]:
        per_model.setdefault(row[0], set()).add(row[5])
    assert per_model == {
        "glm-5.3": {"go", "go-plus", "go-teams"},
        "claude-haiku-5-5": {"go", "go-plus", "go-teams"},
        "gpt-6-luna": {"go", "go-plus", "go-teams"},
    }


@patch("llm_opencode.fetch_models_snapshot")
def test_opencode_plan_all_csv_rows_carry_their_own_plan_data(mock_fetch):
    mock_fetch.return_value = _payload(_unified_models())

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "detail", "--plan", "all", "--format", "csv"])
    assert result.exit_code == 0, result.output

    rows = [row for row in csv.reader(io.StringIO(result.output)) if row]
    by_key = {(row[0], row[5]): row for row in rows[1:]}
    glm_go = by_key[("glm-5.3", "go")]
    glm_plus = by_key[("glm-5.3", "go-plus")]
    assert glm_go[13] == "500"
    assert glm_plus[13] == "1000"


@patch("llm_opencode.fetch_models_snapshot")
def test_opencode_detail_json_unaffected_by_plan_filter(mock_fetch):
    mock_fetch.return_value = _payload(_unified_models())

    runner = CliRunner()
    result = runner.invoke(
        cli, ["opencode", "detail", "--plan", "go-plus", "--format", "json"]
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["count"] == 3
    for model in data["models"]:
        assert set(model["monthly_usage"]) == {"go", "go-plus", "go-teams"}


@patch("llm_opencode.fetch_models_snapshot")
def test_opencode_detail_json_order_ignores_plan_filter(mock_fetch):
    # Asymmetric allowances: sorting by go puts glm-5.3 first, sorting by
    # go-plus would put claude-haiku-5-5 first — so the assertion bites.
    models = _unified_models()
    models[0]["monthly_usage"] = {"go": 500, "go-plus": 30, "go-teams": 505}
    models[1]["monthly_usage"] = {"go": 15, "go-plus": 1000, "go-teams": 20}
    mock_fetch.return_value = _payload(models)

    runner = CliRunner()
    default = runner.invoke(cli, ["opencode", "detail", "--format", "json"])
    plus = runner.invoke(
        cli, ["opencode", "detail", "--plan", "go-plus", "--format", "json"]
    )
    everything = runner.invoke(
        cli, ["opencode", "detail", "--plan", "all", "--format", "json"]
    )
    assert default.exit_code == plus.exit_code == everything.exit_code == 0
    models_default = json.loads(default.output)["models"]
    assert [m["model_id"] for m in models_default] == [
        "glm-5.3",
        "claude-haiku-5-5",
        "gpt-6-luna",
    ]
    assert json.loads(plus.output)["models"] == models_default
    assert json.loads(everything.output)["models"] == models_default


@patch("llm_opencode.fetch_models_snapshot")
def test_opencode_detail_table_surfaces_snapshot_warnings(mock_fetch):
    payload = _payload(_unified_models())
    payload["warnings"] = ["models API entry 7: no id — skipped"]
    mock_fetch.return_value = payload

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "detail"])
    assert result.exit_code == 0, result.output
    assert "Warnings:" in result.output
    assert "  models API entry 7: no id — skipped" in result.output
    assert "Model ID" in result.output


@patch("llm_opencode.fetch_models_snapshot")
def test_opencode_detail_csv_carries_no_warnings_block(mock_fetch):
    payload = _payload(_unified_models())
    payload["warnings"] = ["models API entry 7: no id — skipped"]
    mock_fetch.return_value = payload

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "detail", "--format", "csv"])
    assert result.exit_code == 0, result.output
    assert "Warnings:" not in result.output


@patch("llm_opencode.fetch_models_snapshot")
def test_opencode_models_detail_default_sort_usage(mock_fetch):
    mock_fetch.return_value = _payload(_unified_models())

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "detail"])
    assert result.exit_code == 0, result.output
    # glm-5.3 carries the largest go-plan usage ($500) so it sorts first.
    assert result.output.index("glm-5.3") < result.output.index("claude-haiku-5-5")


@patch("llm_opencode.fetch_models_snapshot")
def test_opencode_models_detail_sort_by_input(mock_fetch):
    mock_fetch.return_value = _payload(_unified_models())

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "detail", "--sort-by", "input"])
    assert result.exit_code == 0, result.output
    assert result.output.index("claude-haiku-5-5") < result.output.index("glm-5.3")


@patch("llm_opencode.fetch_models_snapshot")
def test_opencode_models_detail_sort_requests_removed(mock_fetch):
    mock_fetch.return_value = _payload(_unified_models())

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "detail", "--sort-by", "requests"])
    assert result.exit_code == 2
    assert "is not one of" in result.output


@patch("llm_opencode.fetch_models_snapshot")
def test_opencode_models_detail_sort_by_name(mock_fetch):
    mock_fetch.return_value = _payload(_unified_models())

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "detail", "--sort-by", "name"])
    assert result.exit_code == 0, result.output
    assert result.output.index("Claude Haiku 5.5") < result.output.index("GLM-5.3")


@patch("llm_opencode.fetch_models_snapshot")
def test_opencode_models_detail_usage_missing_pinned_last(mock_fetch):
    models = _unified_models() + [entry(free_api_model())]
    mock_fetch.return_value = _payload(models)

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "detail"])
    assert result.exit_code == 0, result.output
    assert result.output.index("beam-free") > result.output.index("claude-haiku-5-5")


@patch("llm_opencode.fetch_models_snapshot")
def test_opencode_models_detail_json(mock_fetch):
    mock_fetch.return_value = _payload(_unified_models())

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "detail", "--format", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["count"] == 3
    assert [m["model_id"] for m in data["models"]] == [
        "glm-5.3",
        "claude-haiku-5-5",
        "gpt-6-luna",
    ]
    assert data["models"][0]["pricing"]["input"] == 1.4
    assert data["models"][0]["monthly_usage"]["go-plus"] == 1000
    assert data["models"][0]["registered"] is True
    assert data["meta"]["sources"] == [MODELS_URL]
    assert data["meta"]["fetched_at"] == 1700000000
    assert data["meta"]["warnings"] == []


@patch("llm_opencode.fetch_models_snapshot")
def test_opencode_models_detail_json_unregistered_section(mock_fetch):
    models = _unified_models() + [
        entry(api_model(model_id="omen-alpha", api={}, recommended_api="")),
        entry(google_api_model()),
    ]
    mock_fetch.return_value = _payload(models)

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "detail", "--format", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["count"] == 5
    assert [
        (u["model_id"], u["skip_reason"])
        for u in data["unregistered"]
    ] == [
        (
            "omen-alpha",
            "unsupported protocol: unrecognized recommended_api (missing)",
        ),
        ("gemini-4-argon", "unsupported protocol: google-style"),
    ]
    by_id = {m["model_id"]: m for m in data["models"]}
    assert by_id["glm-5.3"]["protocol"] == "openai"
    assert by_id["glm-5.3"]["registered"] is True
    assert by_id["omen-alpha"]["registered"] is False
    assert by_id["omen-alpha"]["skip_reason"] == (
        "unsupported protocol: unrecognized recommended_api (missing)"
    )


@patch("llm_opencode.fetch_models_snapshot")
def test_opencode_models_detail_csv(mock_fetch):
    mock_fetch.return_value = _payload(_unified_models())

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "detail", "--format", "csv"])
    assert result.exit_code == 0, result.output

    rows = list(csv.reader(io.StringIO(result.output)))
    assert rows[0] == [
        "model_id", "model_name", "canonical_id", "provider", "family",
        "plan", "protocol", "recommended_api", "supports_schema",
        "input_price", "output_price", "cached_read", "cached_write", "monthly_usage",
        "training", "zdr", "limited_regions", "featured", "open_weights",
        "knowledge", "release_date", "last_updated",
        "context_limit", "output_limit", "endpoint", "ai_sdk_package",
        "description",
    ]
    glm = rows[1]
    assert glm[0] == "glm-5.3"
    assert glm[5] == "go"
    assert glm[6] == "openai"
    assert glm[9] == "1.4"
    assert glm[13] == "500"


@patch("llm_opencode.fetch_models_snapshot")
def test_opencode_models_detail_csv_ttl_cache_write_is_json(mock_fetch):
    mock_fetch.return_value = _payload([entry(responses_api_model())])

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "detail", "--format", "csv"])
    assert result.exit_code == 0, result.output

    rows = [row for row in csv.reader(io.StringIO(result.output)) if row]
    assert rows[1][12] == '{"30m": 0.125, "12h": 0.25}'


@patch("llm_opencode.fetch_models_snapshot")
def test_opencode_models_detail_csv_protocol_for_unknown_model(mock_fetch):
    models = _unified_models() + [
        entry(api_model(model_id="omen-alpha", api={}, recommended_api="")),
    ]
    mock_fetch.return_value = _payload(models)

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "detail", "--format", "csv"])
    assert result.exit_code == 0, result.output

    rows = list(csv.reader(io.StringIO(result.output)))
    omen = next(row for row in rows[1:] if row[0] == "omen-alpha")
    assert omen[6] == ""  # protocol unknown, rendered without a crash


@patch("llm_opencode.fetch_models_snapshot")
def test_opencode_models_detail_table_unregistered_section(mock_fetch):
    models = _unified_models() + [
        entry(api_model(model_id="omen-alpha", api={}, recommended_api="")),
        entry(google_api_model()),
    ]
    mock_fetch.return_value = _payload(models)

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "detail"])
    assert result.exit_code == 0, result.output
    assert "Unregistered:" in result.output
    assert (
        "omen-alpha — unsupported protocol: unrecognized recommended_api (missing)"
        in result.output
    )
    assert (
        "gemini-4-argon — unsupported protocol: google-style"
        in result.output
    )
    # Graceful protocol rendering for unregistered models.
    assert result.output.count("unknown") >= 2


@patch("llm_opencode.fetch_models_snapshot")
def test_opencode_models_detail_pricing_plan_go_plus(mock_fetch):
    mock_fetch.return_value = _payload(_unified_models())

    runner = CliRunner()
    result = runner.invoke(
        cli, ["opencode", "detail", "--plan", "go-plus", "--format", "pricing"]
    )
    assert result.exit_code == 0, result.output
    assert "Included use: $1000/mo" in result.output
    assert "Included use: $500/mo" not in result.output


@patch("llm_opencode.fetch_models_snapshot")
def test_opencode_models_detail_pricing_plan_all_labels_plans(mock_fetch):
    mock_fetch.return_value = _payload(_unified_models())

    runner = CliRunner()
    result = runner.invoke(
        cli, ["opencode", "detail", "--plan", "all", "--format", "pricing"]
    )
    assert result.exit_code == 0, result.output
    assert "glm-5.3 (go)" in result.output
    assert "glm-5.3 (go-plus)" in result.output
    assert "glm-5.3 (go-teams)" in result.output


@patch("llm_opencode.fetch_models_snapshot")
def test_opencode_models_detail_pricing_sorted_by_input(mock_fetch):
    mock_fetch.return_value = _payload(_unified_models())

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "detail", "--format", "pricing"])
    assert result.exit_code == 0, result.output
    # claude-haiku and gpt-6-luna share the $0.10 input price; glm-5.3 is dearer.
    assert result.output.index("glm-5.3") > result.output.index("claude-haiku-5-5")


@patch("llm_opencode.fetch_models_snapshot")
def test_opencode_models_detail_pricing_ttl_cache_write_renders_cheapest_tier(mock_fetch):
    mock_fetch.return_value = _payload([entry(responses_api_model())])

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "detail", "--format", "pricing"])
    assert result.exit_code == 0, result.output
    assert "Cached Write: $0.125 (30m)" in result.output


@patch("llm_opencode.fetch_models_snapshot")
def test_opencode_models_detail_pricing_cost_notes(mock_fetch):
    mock_fetch.return_value = _payload(_unified_models())

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "detail", "--format", "pricing"])
    assert result.exit_code == 0, result.output
    # claude-haiku carries context-tiered overrides; glm-5.3 does not.
    claude_block = result.output[result.output.index("claude-haiku-5-5"):]
    assert "context-tiered prices or time-window multipliers" in claude_block
    glm_block = result.output[result.output.index("glm-5.3"):]
    assert "context-tiered prices or time-window multipliers" not in glm_block


@patch("llm_opencode.fetch_models_snapshot")
def test_opencode_models_detail_search_filter(mock_fetch):
    mock_fetch.return_value = _payload(_unified_models())

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "detail", "--search", "glm"])
    assert result.exit_code == 0, result.output
    assert "glm-5.3" in result.output
    assert "claude-haiku-5-5" not in result.output


@patch("llm_opencode.fetch_models_snapshot")
def test_opencode_models_detail_search_matches_name(mock_fetch):
    mock_fetch.return_value = _payload(_unified_models())

    runner = CliRunner()
    result = runner.invoke(
        cli, ["opencode", "detail", "--search", "haiku"]
    )
    assert result.exit_code == 0, result.output
    assert "claude-haiku-5-5" in result.output
    assert "glm-5.3" not in result.output


@patch("llm_opencode.fetch_models_snapshot")
def test_opencode_models_detail_provider_filter_removed(mock_fetch):
    mock_fetch.return_value = _payload(_unified_models())

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "detail", "--provider", "opencode"])
    assert result.exit_code == 2
    assert "No such option" in result.output


@patch("llm_opencode.fetch_models_snapshot")
def test_opencode_models_detail_model_exact_filter(mock_fetch):
    mock_fetch.return_value = _payload(_unified_models())

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "detail", "--model", "glm-5.3"])
    assert result.exit_code == 0, result.output
    assert "glm-5.3" in result.output
    assert "claude-haiku-5-5" not in result.output


@patch("llm_opencode.fetch_models_snapshot")
def test_opencode_models_detail_empty(mock_fetch):
    mock_fetch.return_value = _payload([])

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "detail"])
    assert result.exit_code == 0, result.output
    assert "No models found." in result.output


@patch("llm_opencode.fetch_models_snapshot")
def test_opencode_detail_schema_column(mock_fetch):
    mock_fetch.return_value = _payload(
        [
            entry(api_model()),  # openai, structured_output: true
            entry(anthropic_api_model()),  # advertises true, wire family says no
            entry(api_model(model_id="omen-alpha"), capabilities={"structured_output": False}),
        ]
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "detail"])
    assert result.exit_code == 0, result.output
    assert "Schema" in result.output

    rows = {
        line.split()[0]: line for line in result.output.splitlines()
        if line.startswith(("glm-5.3", "claude-haiku-5-5", "omen-alpha"))
    }
    assert " yes" in rows["glm-5.3"]
    assert " no" in rows["claude-haiku-5-5"]
    assert " no" in rows["omen-alpha"]


@patch("llm_opencode.fetch_models_snapshot")
def test_opencode_detail_csv_carries_supports_schema(mock_fetch):
    mock_fetch.return_value = _payload(
        [
            entry(api_model()),
            entry(anthropic_api_model()),
        ]
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "detail", "--format", "csv"])
    assert result.exit_code == 0, result.output

    rows = [row for row in csv.reader(io.StringIO(result.output)) if row]
    assert "supports_schema" in rows[0]
    index = rows[0].index("supports_schema")
    by_id = {row[0]: row for row in rows[1:]}
    assert by_id["glm-5.3"][index] == "True"
    assert by_id["claude-haiku-5-5"][index] == "False"


@patch("llm_opencode.fetch_models_snapshot")
def test_opencode_detail_schema_capable_filter(mock_fetch):
    mock_fetch.return_value = _payload(
        [
            entry(api_model()),
            entry(responses_api_model()),
            entry(anthropic_api_model()),
            entry(api_model(model_id="omen-alpha"), capabilities={"structured_output": False}),
        ]
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "detail", "--schema-capable"])
    assert result.exit_code == 0, result.output
    assert "glm-5.3" in result.output
    assert "gpt-6-luna" in result.output
    assert "claude-haiku-5-5" not in result.output
    assert "omen-alpha" not in result.output

    # A selection filter like --search: applies to JSON too.
    json_result = runner.invoke(
        cli, ["opencode", "detail", "--schema-capable", "--format", "json"]
    )
    assert json_result.exit_code == 0, json_result.output
    data = json.loads(json_result.output)
    assert [m["model_id"] for m in data["models"]] == ["glm-5.3", "gpt-6-luna"]
    assert all(m["supports_schema"] is True for m in data["models"])


@patch("llm_opencode.fetch_models_snapshot")
def test_opencode_models_detail_pricing_no_priced_models(mock_fetch):
    mock_fetch.return_value = _payload(
        [entry(api_model(), pricing={"input": None})]
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "detail", "--format", "pricing"])
    assert result.exit_code == 0, result.output
    assert "No pricing data found." in result.output


@patch("llm_opencode.fetch_models_snapshot")
def test_opencode_models_detail_pricing_missing_values(mock_fetch):
    mock_fetch.return_value = _payload(
        [
            entry(
                api_model(model_id="mystery-model", name="Mystery Model"),
                pricing={
                    "input": 1.0, "output": 2.0, "cached_read": None,
                    "cached_write": None,
                },
            )
        ]
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "detail", "--format", "pricing"])
    assert result.exit_code == 0, result.output
    assert "mystery-model" in result.output
    assert "Cached Read:  —" in result.output
    assert "Cached Write: —" in result.output


@patch("llm_opencode.fetch_models_snapshot")
def test_opencode_models_detail_pricing_free_model(mock_fetch):
    mock_fetch.return_value = _payload([entry(free_api_model())])

    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "detail", "--format", "pricing"])
    assert result.exit_code == 0, result.output
    assert "Input:        $0.00" in result.output
    assert "Included use:" not in result.output  # free models carry no allowance


@patch("llm_opencode.fetch_models_snapshot")
def test_opencode_models_detail_output_file(mock_fetch, tmp_path):
    mock_fetch.return_value = _payload(_unified_models())
    output_file = tmp_path / "models.csv"

    runner = CliRunner()
    result = runner.invoke(
        cli, ["opencode", "detail", "--format", "csv", "-o", str(output_file)]
    )
    assert result.exit_code == 0, result.output
    assert output_file.exists()
    rows = list(csv.reader(io.StringIO(output_file.read_text())))
    assert rows[1][0] == "glm-5.3"


@patch("llm_opencode.fetch_models_snapshot", side_effect=DownloadError("no net"))
def test_opencode_models_detail_fetch_failure_shows_api_url(mock_fetch):
    runner = CliRunner()
    result = runner.invoke(cli, ["opencode", "detail"])
    assert result.exit_code == 1
    assert "models API unavailable" in result.output
    assert MODELS_URL in result.output


# ─── sort_models ─────────────────────────────────────────────────────────────


def _usage_model(model_id, usage, input_price):
    return entry(
        api_model(model_id=model_id),
        pricing={"input": input_price, "output": 1.0, "cached_read": None, "cached_write": None},
        monthly_usage={"go": usage} if usage is not None else {},
    )


def test_sort_models_usage_default():
    models = [
        _usage_model("a", 15, 1.0),
        _usage_model("b", 60, 2.0),
        _usage_model("c", None, 0.5),
    ]
    result = sort_models(models, sort_by="usage")
    assert [m["model_id"] for m in result] == ["b", "a", "c"]


def test_sort_models_usage_tiebreak_input():
    models = [
        _usage_model("a", 40, 1.0),
        _usage_model("b", 40, 0.5),
    ]
    result = sort_models(models, sort_by="usage")
    assert [m["model_id"] for m in result] == ["b", "a"]


def test_sort_models_usage_respects_plan_filter():
    models = [
        _usage_model("a", 15, 1.0),
        _usage_model("b", 60, 1.0),
    ]
    models[0]["monthly_usage"]["go-plus"] = 50
    models[1]["monthly_usage"]["go-plus"] = 5
    result = sort_models(models, sort_by="usage", plan="go-plus")
    assert [m["model_id"] for m in result] == ["a", "b"]


def test_sort_models_usage_all_mode_uses_most_generous_plan():
    models = [
        _usage_model("a", 15, 1.0),
        _usage_model("b", 60, 1.0),
    ]
    models[0]["monthly_usage"]["go-teams"] = 2000
    result = sort_models(models, sort_by="usage", plan="all")
    assert [m["model_id"] for m in result] == ["a", "b"]


def test_sort_models_by_input():
    models = [
        _usage_model("a", 15, 2.0),
        _usage_model("b", 15, 0.5),
    ]
    result = sort_models(models, sort_by="input")
    assert [m["model_id"] for m in result] == ["b", "a"]


def test_sort_models_by_input_missing_last():
    models = [
        _usage_model("a", 15, None),
        _usage_model("b", 15, 2.0),
    ]
    result = sort_models(models, sort_by="input")
    assert [m["model_id"] for m in result] == ["b", "a"]


def test_sort_models_by_output():
    models = [
        _usage_model("a", 15, 1.0),
        _usage_model("b", 15, 1.0),
    ]
    models[0]["pricing"]["output"] = 5.0
    models[1]["pricing"]["output"] = 1.0
    result = sort_models(models, sort_by="output")
    assert [m["model_id"] for m in result] == ["b", "a"]


def test_sort_models_requests_key_removed():
    models = [_usage_model("a", 15, 1.0)]
    with pytest.raises(KeyError):
        sort_models(models, sort_by="requests")


def test_sort_models_by_name():
    models = [
        entry(api_model(model_id="b", name="Beta")),
        entry(api_model(model_id="a", name="Alpha")),
    ]
    result = sort_models(models, sort_by="name")
    assert [m["model_id"] for m in result] == ["a", "b"]


def test_sort_models_by_model_id():
    models = [
        entry(api_model(model_id="b")),
        entry(api_model(model_id="a")),
    ]
    result = sort_models(models, sort_by="model_id")
    assert [m["model_id"] for m in result] == ["a", "b"]


def test_sort_models_empty():
    assert sort_models([]) == []


# ─── rendering tolerance ─────────────────────────────────────────────────────


def test_format_unified_table_tolerates_missing_blocks():
    from llm_opencode import format_unified_table

    model = entry(
        api_model(),
        pricing={"input": 0.5, "output": 1.0, "cached_read": 0.1, "cached_write": 0.2},
        monthly_usage={},
    )

    result = format_unified_table([model])

    assert result.count("—") == 2  # usage + schema dashes; policies render yes/no
    assert "glm-5.3" in result


def test_format_unified_table_tolerates_non_dict_pricing():
    from llm_opencode import format_unified_table

    result = format_unified_table([{"model_id": "x", "pricing": None}])
    assert "x" in result


def test_row_usage_strict_per_plan():
    from llm_opencode.cli import _row_usage

    model = {"monthly_usage": {"go": 5, "go-plus": 200, "go-teams": 105}}
    assert _row_usage(model, "go") == 5
    assert _row_usage(model, "go-plus") == 200
    assert _row_usage(model, "go-teams") == 105
    # Strict: a plan the API didn't price renders nothing, never borrowed.
    assert _row_usage({"monthly_usage": {"go": 5}}, "go-plus") is None
    assert _row_usage({"monthly_usage": None}, "go") is None
    assert _row_usage({"monthly_usage": "flat"}, "go") is None


def test_fmt_price_handles_sub_cent_values():
    from llm_opencode.cli import _fmt_price

    assert _fmt_price(None) == "—"
    assert _fmt_price(0) == "$0.00"
    assert _fmt_price(2) == "$2.00"
    assert _fmt_price(0.003) == "$0.003"
    assert _fmt_price(0.0036) == "$0.0036"


def test_fmt_cached_write_scalar_and_ttl():
    from llm_opencode.cli import _fmt_cached_write

    assert _fmt_cached_write(None) == "—"
    assert _fmt_cached_write(0.125) == "$0.12"  # .2f banker's rounding
    assert _fmt_cached_write({"30m": 0.25, "12h": 0.5}) == "$0.25 (30m)"
    assert _fmt_cached_write({}) == "—"


def test_fmt_tokens_compact():
    from llm_opencode.cli import _fmt_tokens

    assert _fmt_tokens(None) == "—"
    assert _fmt_tokens(1000000) == "1M"
    assert _fmt_tokens(1048576) == "1M"
    assert _fmt_tokens(500000) == "500K"
    assert _fmt_tokens(262144) == "262K"
    assert _fmt_tokens(512) == "512"
