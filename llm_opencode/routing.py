import llm

# Fallback API bases, used only when a model class is instantiated
# directly (outside snapshot registration). The API's per-model
# ``base_url`` is the source of truth for registered models.
BASE_URL_OPENAI = "https://opencode.ai/inference/go/openai/v1"
BASE_URL_ANTHROPIC = "https://opencode.ai/inference/go/anthropic"

# recommended_api → wire protocol the plugin speaks. The API names the
# route family per model; any other value is either google-style (known,
# unsupported) or unrecognized.
_RECOMMENDED_API_PROTOCOLS = {
    "chat-completions": "openai",
    "responses": "openai-responses",
    "messages": "anthropic",
}

# The recommended_api of Google-style models: a route family the plugin
# knows by name but speaks no client for.
_GOOGLE_RECOMMENDED_API = "generate-content"

# Skip reasons surfaced in the detail JSON's unregistered section.
SKIP_NO_API_BASE = "no api base_url"
SKIP_GOOGLE_STYLE = "unsupported protocol: google-style"
SKIP_UNRECOGNIZED_API = "unsupported protocol: unrecognized recommended_api"


def _api_base(endpoint, protocol):
    """SDK-ready API base from the model's advertised ``base_url``.

    OpenAI SDKs append the wire path after the ``/v1`` mount, so the
    advertised base stands as-is; the Anthropic SDK appends
    ``/v1/messages``, so its base drops the trailing ``/v1``.
    """
    if protocol == "anthropic" and endpoint.rstrip("/").endswith("/v1"):
        return endpoint.rstrip("/")[: -len("/v1")]
    return endpoint


def derive_route(entry):
    """Derive the registration route for one unified snapshot entry.

    Returns a copy of the entry with the derived fields the detail renderer
    consumes: ``registered``, ``protocol``, ``api_base`` and
    ``skip_reason``. A model whose ``recommended_api`` the plugin cannot
    speak is not registered, never guessed: google-style →
    :data:`SKIP_GOOGLE_STYLE`, an unknown value →
    :data:`SKIP_UNRECOGNIZED_API`, a missing ``base_url`` →
    :data:`SKIP_NO_API_BASE`.
    """
    route = dict(entry)
    route["registered"] = False
    route["protocol"] = None
    route["api_base"] = None
    route["skip_reason"] = None

    recommended = entry.get("recommended_api") or ""
    if recommended == _GOOGLE_RECOMMENDED_API:
        route["skip_reason"] = SKIP_GOOGLE_STYLE
        return route
    protocol = _RECOMMENDED_API_PROTOCOLS.get(recommended)
    if protocol is None:
        route["skip_reason"] = f"{SKIP_UNRECOGNIZED_API} ({recommended or 'missing'})"
        return route
    endpoint = entry.get("endpoint") or ""
    if not endpoint:
        route["skip_reason"] = SKIP_NO_API_BASE
        return route

    route["protocol"] = protocol
    route["api_base"] = _api_base(endpoint, protocol)
    route["registered"] = True
    return route


def analyze_routes(entries):
    """Derive registration routes for unified snapshot entries, in order."""
    return [derive_route(entry) for entry in entries]


def unregistered_routes(routes):
    """The detail JSON's unregistered section: skipped models with reasons."""
    return [
        {
            "model_id": route.get("model_id", ""),
            "endpoint": route.get("endpoint", ""),
            "protocol": route["protocol"],
            "skip_reason": route["skip_reason"],
        }
        for route in routes
        if not route["registered"]
    ]


@llm.hookimpl
def register_models(register):
    """Register models from the snapshot, routed by recommended_api.

    Registration serves the snapshot file whatever its age and never fetches
    for it; only when no snapshot has ever been written does it run a single
    blocking bootstrap fetch (failure to bootstrap registers nothing).
    Models whose route the plugin cannot speak are skipped, never guessed:
    google-style → ``unsupported protocol: google-style``; an unknown
    recommended_api → ``unsupported protocol: unrecognized recommended_api``;
    a missing base_url → ``no api base_url``.
    Registration prints nothing.
    """
    from llm_opencode import (
        OpenCodeGoAnthropicAsyncChat,
        OpenCodeGoAnthropicChat,
        OpenCodeGoAsyncChat,
        OpenCodeGoChat,
        OpenCodeGoResponsesAsyncChat,
        OpenCodeGoResponsesChat,
        fetch_models_snapshot,
        read_snapshot,
    )

    key = llm.get_key("", "opencode", "OPENCODE_KEY")
    if not key:
        return
    snapshot = read_snapshot()
    if snapshot is None:
        try:
            snapshot = fetch_models_snapshot()
        except Exception:
            return
    entries = snapshot.get("models", []) if isinstance(snapshot, dict) else snapshot
    for route in analyze_routes(entries):
        if not route["registered"]:
            continue
        llm_model_id = f"opencode-go/{route['model_id']}"
        # The OpenAI-family classes send model_name on the wire, so it must
        # be the raw API model id — display names live in `llm opencode detail`.
        wire_name = route["model_id"]
        # Schema support follows the API's per-model claim. Only the OpenAI
        # wire families have llm-side schema plumbing; the Anthropic class
        # has no schema translation, so it stays honestly False.
        supports_schema = bool(
            (route.get("capabilities") or {}).get("structured_output")
        )
        if route["protocol"] == "openai":
            register(
                OpenCodeGoChat(
                    model_id=llm_model_id,
                    model_name=wire_name,
                    api_base=route["api_base"],
                    supports_schema=supports_schema,
                ),
                OpenCodeGoAsyncChat(
                    model_id=llm_model_id,
                    model_name=wire_name,
                    api_base=route["api_base"],
                    supports_schema=supports_schema,
                ),
            )
        elif route["protocol"] == "openai-responses":
            register(
                OpenCodeGoResponsesChat(
                    model_id=llm_model_id,
                    model_name=wire_name,
                    api_base=route["api_base"],
                    supports_schema=supports_schema,
                ),
                OpenCodeGoResponsesAsyncChat(
                    model_id=llm_model_id,
                    model_name=wire_name,
                    api_base=route["api_base"],
                    supports_schema=supports_schema,
                ),
            )
        elif route["protocol"] == "anthropic":
            register(
                OpenCodeGoAnthropicChat(
                    model_id=llm_model_id, api_base=route["api_base"]
                ),
                OpenCodeGoAnthropicAsyncChat(
                    model_id=llm_model_id, api_base=route["api_base"]
                ),
            )
