from unittest.mock import MagicMock, patch

import llm_opencode
from llm_opencode import (
    SKIP_GOOGLE_STYLE,
    SKIP_NO_DOCS_ROW,
    SKIP_UNRECOGNIZED_PATH,
    analyze_routes,
    corroboration_warnings,
    derive_route,
    unregistered_routes,
)

CHAT_ENDPOINT = "https://opencode.ai/inference/go/v1/chat/completions"
RESPONSES_ENDPOINT = "https://opencode.ai/inference/go/v1/responses"
ANTHROPIC_ENDPOINT = "https://opencode.ai/inference/go/v1/messages"
GOOGLE_ENDPOINT = "https://opencode.ai/inference/go/v1/models/gemini-4-argon-contributor"


def _entry(model_id, endpoint="", package="", **extras):
    entry = {
        "model_id": model_id,
        "model_name": "",
        "endpoint": endpoint,
        "ai_sdk_package": package,
    }
    entry.update(extras)
    return entry


def _snapshot(*entries):
    return {"scraped_at": 1, "warnings": [], "models": list(entries)}


# ─── the three routes (known-protocol / no-row / google-style) ───────────────

def test_derive_route_chat_completions():
    route = derive_route(_entry("glm-5", CHAT_ENDPOINT, "@ai-sdk/openai-compatible"))

    assert route["registered"] is True
    assert route["protocol"] == "openai"
    assert route["api_base"] == "https://opencode.ai/inference/go/v1"
    assert route["skip_reason"] is None
    assert route["corroboration_warning"] is None


def test_derive_route_responses():
    route = derive_route(_entry("gpt-6-luna", RESPONSES_ENDPOINT, "@ai-sdk/openai"))

    assert route["registered"] is True
    assert route["protocol"] == "openai-responses"
    assert route["api_base"] == "https://opencode.ai/inference/go/v1"


def test_derive_route_anthropic_messages():
    route = derive_route(_entry("minimax-m3", ANTHROPIC_ENDPOINT, "@ai-sdk/anthropic"))

    assert route["registered"] is True
    assert route["protocol"] == "anthropic"
    assert route["api_base"] == "https://opencode.ai/inference/go"


def test_derive_route_no_docs_row_skips():
    route = derive_route(_entry("omen-alpha"))

    assert route["registered"] is False
    assert route["protocol"] is None
    assert route["skip_reason"] == SKIP_NO_DOCS_ROW


def test_derive_route_google_style_skips():
    route = derive_route(_entry("gemini-4-argon-contributor", GOOGLE_ENDPOINT, "@ai-sdk/google"))

    assert route["registered"] is False
    assert route["protocol"] is None
    assert route["skip_reason"] == SKIP_GOOGLE_STYLE


def test_derive_route_unrecognized_path_skips():
    route = derive_route(_entry("whisper-huge", "https://x.test/v1/audio/transcriptions"))

    assert route["registered"] is False
    assert route["skip_reason"] == f"{SKIP_UNRECOGNIZED_PATH} (/v1/audio/transcriptions)"


def test_derive_route_keeps_entry_fields():
    route = derive_route(
        _entry("glm-5", CHAT_ENDPOINT, "", matched=True, data_retention="30 days")
    )

    assert route["matched"] is True
    assert route["data_retention"] == "30 days"


def test_derive_route_variant_suffix_routes_from_inherited_endpoint():
    route = derive_route(_entry("grok-4.7#le-200k-tokens", RESPONSES_ENDPOINT))

    assert route["registered"] is True
    assert route["protocol"] == "openai-responses"


def test_derive_route_google_style_variant_interacts_with_suffixing():
    route = derive_route(
        _entry("gemini-4-argon-contributor#le-1m-tokens", GOOGLE_ENDPOINT)
    )

    assert route["registered"] is False
    assert route["skip_reason"] == SKIP_GOOGLE_STYLE


# ─── per-model API base over the hardcoded host ──────────────────────────────

def test_derive_route_per_model_host_honored_over_hardcoded():
    route = derive_route(
        _entry("hy4-preview", "https://example-alt.test/inference/go/v1/chat/completions")
    )

    assert route["api_base"] == "https://example-alt.test/inference/go/v1"


def test_derive_route_per_model_anthropic_base_drops_v1():
    route = derive_route(_entry("minimax-m3", ANTHROPIC_ENDPOINT))
    assert route["api_base"] == "https://opencode.ai/inference/go"


def test_derive_route_fallback_base_when_host_absent():
    route = derive_route(_entry("glm-5", "/v1/chat/completions"))

    assert route["registered"] is True
    assert route["protocol"] == "openai"
    assert route["api_base"] == "https://opencode.ai/zen/go/v1"


def test_derive_route_fallback_anthropic_base_when_host_absent():
    route = derive_route(_entry("minimax-m3", "/v1/messages"))

    assert route["registered"] is True
    assert route["protocol"] == "anthropic"
    assert route["api_base"] == "https://opencode.ai/zen/go"


def test_derive_route_default_anthropic_base_without_arg():
    model = llm_opencode.OpenCodeGoAnthropicChat(model_id="opencode-go/minimax-m3")
    assert model.api_base == "https://opencode.ai/zen/go"


# ─── corroboration: disagreement warns, never vetoes ─────────────────────────

def test_corroboration_mismatch_warns_but_still_registers():
    route = derive_route(_entry("glm-5", CHAT_ENDPOINT, "@ai-sdk/google"))

    assert route["registered"] is True
    assert route["protocol"] == "openai"
    assert route["corroboration_warning"] == (
        'protocol corroboration: glm-5 endpoint path resolves openai '
        'but ai_sdk_package is "@ai-sdk/google"'
    )


def test_corroboration_agreement_warns_nothing():
    assert (
        derive_route(_entry("glm-5", CHAT_ENDPOINT, "@ai-sdk/openai-compatible"))[
            "corroboration_warning"
        ]
        is None
    )
    assert (
        derive_route(_entry("gpt-6-luna", RESPONSES_ENDPOINT, "@ai-sdk/openai"))[
            "corroboration_warning"
        ]
        is None
    )
    assert (
        derive_route(_entry("minimax-m3", ANTHROPIC_ENDPOINT, "@ai-sdk/anthropic"))[
            "corroboration_warning"
        ]
        is None
    )


def test_corroboration_missing_package_warns_nothing():
    route = derive_route(_entry("glm-5", CHAT_ENDPOINT, ""))

    assert route["corroboration_warning"] is None
    assert route["registered"] is True


# ─── routing plumbing ────────────────────────────────────────────────────────

def _routing_fixture():
    return [
        _entry("glm-5", CHAT_ENDPOINT, "@ai-sdk/openai-compatible"),
        _entry("gpt-6-luna", RESPONSES_ENDPOINT, "@ai-sdk/openai"),
        _entry("minimax-m3", ANTHROPIC_ENDPOINT, "@ai-sdk/anthropic"),
        _entry("omen-alpha"),
        _entry(
            "gemini-4-argon-contributor",
            GOOGLE_ENDPOINT,
            "@ai-sdk/google",
        ),
        _entry("corroborate-me", CHAT_ENDPOINT, "@ai-sdk/google"),
    ]


def test_analyze_routes_preserves_order():
    routes = analyze_routes(_routing_fixture())

    assert [route["model_id"] for route in routes] == [
        "glm-5",
        "gpt-6-luna",
        "minimax-m3",
        "omen-alpha",
        "gemini-4-argon-contributor",
        "corroborate-me",
    ]
    assert [route["registered"] for route in routes] == [True, True, True, False, False, True]


def test_corroboration_warnings_collects_mismatches_only():
    warnings = corroboration_warnings(analyze_routes(_routing_fixture()))

    assert warnings == [
        'protocol corroboration: corroborate-me endpoint path resolves openai '
        'but ai_sdk_package is "@ai-sdk/google"'
    ]


def test_unregistered_routes_shapes_for_detail_renderer():
    section = unregistered_routes(analyze_routes(_routing_fixture()))

    assert section == [
        {
            "model_id": "omen-alpha",
            "endpoint": "",
            "protocol": None,
            "skip_reason": "protocol unknown: no docs row",
        },
        {
            "model_id": "gemini-4-argon-contributor",
            "endpoint": GOOGLE_ENDPOINT,
            "protocol": None,
            "skip_reason": "unsupported protocol: google-style",
        },
    ]


def test_registration_module_has_no_model_id_lists():
    import llm_opencode.routing as routing

    for name in (
        "OPENAI_PROTOCOL_MODELS",
        "OPENAI_RESPONSES_PROTOCOL_MODELS",
        "ANTHROPIC_PROTOCOL_MODELS",
        "UNKNOWN_MODELS_DEFAULT",
    ):
        assert not hasattr(routing, name), name


def test_register_models_routes_from_snapshot_entries(user_dir):
    with (
        patch("llm.get_key", return_value="sk-test"),
        patch("llm_opencode.fetch_unified_models") as mock_fetch,
    ):
        mock_fetch.return_value = _snapshot(*_routing_fixture())
        register = MagicMock()
        llm_opencode.register_models(register)

    registered = [call[0][0].model_id for call in register.call_args_list]
    assert registered == [
        "opencode-go/glm-5",
        "opencode-go/gpt-6-luna",
        "opencode-go/minimax-m3",
        "opencode-go/corroborate-me",
    ]
