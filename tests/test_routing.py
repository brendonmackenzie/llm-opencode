from unittest.mock import MagicMock, patch

import llm_opencode
from llm_opencode import (
    SKIP_GOOGLE_STYLE,
    SKIP_NO_API_BASE,
    SKIP_UNRECOGNIZED_API,
    analyze_routes,
    derive_route,
    unregistered_routes,
)
from tests._payload import (
    CHAT_BASE_URL,
    anthropic_api_model,
    api_model,
    entry,
    google_api_model,
    snapshot,
)

ANTHROPIC_BASE_URL = "https://opencode.ai/inference/go/anthropic/v1"


# ─── the three routes (known-protocol / unknown / google-style) ────────────


def test_derive_route_chat_completions():
    route = derive_route(entry(api_model()))

    assert route["registered"] is True
    assert route["protocol"] == "openai"
    assert route["api_base"] == CHAT_BASE_URL
    assert route["skip_reason"] is None


def test_derive_route_responses():
    from tests._payload import responses_api_model

    route = derive_route(entry(responses_api_model()))

    assert route["registered"] is True
    assert route["protocol"] == "openai-responses"
    assert route["api_base"] == CHAT_BASE_URL


def test_derive_route_anthropic_messages():
    route = derive_route(entry(anthropic_api_model()))

    assert route["registered"] is True
    assert route["protocol"] == "anthropic"
    # The Anthropic SDK appends /v1/messages: the base drops the mount.
    assert route["api_base"] == "https://opencode.ai/inference/go/anthropic"


def test_derive_route_google_style_skips():
    route = derive_route(entry(google_api_model()))

    assert route["registered"] is False
    assert route["protocol"] is None
    assert route["skip_reason"] == SKIP_GOOGLE_STYLE


def test_derive_route_unrecognized_recommended_api_skips():
    route = derive_route(entry(api_model(), recommended_api="embeddings"))

    assert route["registered"] is False
    assert route["skip_reason"] == (
        f"{SKIP_UNRECOGNIZED_API} (embeddings)"
    )


def test_derive_route_missing_recommended_api_skips():
    route = derive_route(entry(api_model(), recommended_api=""))

    assert route["registered"] is False
    assert route["skip_reason"] == f"{SKIP_UNRECOGNIZED_API} (missing)"


def test_derive_route_missing_base_url_skips():
    route = derive_route(
        entry(
            api_model(
                api={"chat-completions": {"base_url": "", "sdk": {"aisdk": "x"}}}
            )
        )
    )

    assert route["registered"] is False
    assert route["skip_reason"] == SKIP_NO_API_BASE


def test_derive_route_keeps_entry_fields():
    route = derive_route(entry(api_model(), zdr=False))

    assert route["zdr"] is False
    assert route["model_name"] == "GLM-5.3"


# ─── api_base derivation ─────────────────────────────────────────────────────


def test_derive_route_per_model_host_honored_over_default():
    route = derive_route(
        entry(
            api_model(
                api={
                    "chat-completions": {
                        "base_url": "https://example-alt.test/openai/v1",
                        "sdk": {"aisdk": "@ai-sdk/openai-compatible"},
                    }
                }
            )
        )
    )

    assert route["api_base"] == "https://example-alt.test/openai/v1"


def test_derive_route_anthropic_base_without_v1_mount_stands_as_is():
    route = derive_route(
        entry(
            anthropic_api_model(
                api={
                    "messages": {
                        "base_url": "https://example-alt.test/anthropic",
                        "sdk": {"aisdk": "@ai-sdk/anthropic"},
                    }
                }
            )
        )
    )

    assert route["api_base"] == "https://example-alt.test/anthropic"


def test_derive_route_default_anthropic_base_without_arg():
    model = llm_opencode.OpenCodeGoAnthropicChat(model_id="opencode-go/claude-haiku-5-5")
    assert model.api_base == llm_opencode.BASE_URL_ANTHROPIC


# ─── routing plumbing ───────────────────────────────────────────────────────


def _routing_fixture():
    from tests._payload import responses_api_model

    return [
        entry(api_model()),
        entry(responses_api_model()),
        entry(anthropic_api_model()),
        entry(api_model(model_id="omen-alpha", api={}, recommended_api="")),
        entry(google_api_model()),
        entry(api_model(model_id="embed-me", recommended_api="embeddings")),
    ]


def test_analyze_routes_preserves_order():
    routes = analyze_routes(_routing_fixture())

    assert [route["model_id"] for route in routes] == [
        "glm-5.3",
        "gpt-6-luna",
        "claude-haiku-5-5",
        "omen-alpha",
        "gemini-4-argon",
        "embed-me",
    ]
    assert [route["registered"] for route in routes] == [
        True,
        True,
        True,
        False,
        False,
        False,
    ]


def test_unregistered_routes_shapes_for_detail_renderer():
    section = unregistered_routes(analyze_routes(_routing_fixture()))

    assert section == [
        {
            "model_id": "omen-alpha",
            "endpoint": "",
            "protocol": None,
            "skip_reason": f"{SKIP_UNRECOGNIZED_API} (missing)",
        },
        {
            "model_id": "gemini-4-argon",
            "endpoint": "https://opencode.ai/inference/go/google/v1beta",
            "protocol": None,
            "skip_reason": "unsupported protocol: google-style",
        },
        {
            "model_id": "embed-me",
            "endpoint": "",
            "protocol": None,
            "skip_reason": f"{SKIP_UNRECOGNIZED_API} (embeddings)",
        },
    ]


def test_routing_module_has_no_model_id_lists():
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
        patch("llm_opencode.fetch_models_snapshot") as mock_fetch,
    ):
        mock_fetch.return_value = snapshot(*_routing_fixture())
        register = MagicMock()
        llm_opencode.register_models(register)

    registered = [call[0][0].model_id for call in register.call_args_list]
    assert registered == [
        "opencode-go/glm-5.3",
        "opencode-go/gpt-6-luna",
        "opencode-go/claude-haiku-5-5",
    ]
