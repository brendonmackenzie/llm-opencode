import re

import llm

# Fallback API bases, used only when a model's Endpoint cell carries a path
# without a host (or when the docs can vouch for the protocol but not the
# host). The Endpoint cell is the per-model source of truth for everything else.
BASE_URL_OPENAI = "https://opencode.ai/zen/go/v1"
BASE_URL_ANTHROPIC = "https://opencode.ai/zen/go"

# Path→protocol mapping is authoritative: the Endpoint cell's path decides what
# wire protocol a model speaks. The AI SDK Package column corroborates only.
_PROTOCOL_ROUTES = (
    ("openai", "/v1/chat/completions"),
    ("openai-responses", "/v1/responses"),
    ("anthropic", "/v1/messages"),
)

# Google-style per-model resource path: no OpenAI/Anthropic wire family the
# plugin speaks serves this shape.
_GOOGLE_PATH_RE = re.compile(r"/v1(?:beta)?/models/[^/]+/?$")

# Skip reasons surfaced in the detail JSON's unregistered section.
SKIP_NO_DOCS_ROW = "protocol unknown: no docs row"
SKIP_GOOGLE_STYLE = "unsupported protocol: google-style"
SKIP_UNRECOGNIZED_PATH = "unsupported protocol: unrecognized endpoint path"

# Corroboration: the package column must agree with the path's protocol;
# disagreement warns (visible in detail) but never vetoes registration.
_EXPECTED_PACKAGES = {
    "openai": {"@ai-sdk/openai", "@ai-sdk/openai-compatible"},
    "openai-responses": {"@ai-sdk/openai"},
    "anthropic": {"@ai-sdk/anthropic"},
}


def _split_host_and_path(endpoint):
    """Split an endpoint into ``(scheme://host, path)``; bare paths have no host."""
    match = re.match(r"^(https?://[^/]+)(/.*)$", endpoint)
    if match:
        return match.group(1), match.group(2)
    return None, endpoint if endpoint.startswith("/") else "/" + endpoint


def _match_path(path):
    """Match an endpoint path against the authoritative protocol routes.

    Returns ``(protocol, suffix)`` where suffix is the path tail that named
    the protocol; ``(None, None)`` when the path speaks no known protocol.
    """
    if _GOOGLE_PATH_RE.search(path):
        return "google-style", None
    for protocol, suffix in _PROTOCOL_ROUTES:
        if path.endswith(suffix):
            return protocol, suffix
    return None, None


def _api_base(host, path, protocol, suffix):
    """Per-model API base derived from the endpoint URL.

    The OpenAI SDKs append the wire path to ``base_url`` (so the base keeps
    its ``/v1`` mount); the Anthropic SDK appends ``/v1/messages`` (so the
    base drops it). Hostless endpoint cells fall back to the zen constants.
    """
    if host is None:
        return BASE_URL_ANTHROPIC if protocol == "anthropic" else BASE_URL_OPENAI
    if protocol == "anthropic":
        return host + path[: -len(suffix)]
    return host + path[: -len(suffix)] + "/v1"


def derive_route(entry):
    """Derive the registration route for one unified snapshot entry.

    Returns a copy of the entry with the derived fields the detail renderer
    consumes: ``registered``, ``protocol``, ``api_base``, ``skip_reason`` and
    ``corroboration_warning``. A model the docs cannot vouch for (no Endpoint
    cell) or one speaking an unsupported protocol is not registered; the AI
    SDK Package column can disagree with the path but only warns.
    """
    route = dict(entry)
    route["registered"] = False
    route["protocol"] = None
    route["api_base"] = None
    route["skip_reason"] = None
    route["corroboration_warning"] = None

    endpoint = entry.get("endpoint") or ""
    if not endpoint:
        route["skip_reason"] = SKIP_NO_DOCS_ROW
        return route

    host, path = _split_host_and_path(endpoint)
    protocol, suffix = _match_path(path)
    if protocol == "google-style":
        route["skip_reason"] = SKIP_GOOGLE_STYLE
        return route
    if protocol is None:
        route["skip_reason"] = f"{SKIP_UNRECOGNIZED_PATH} ({path})"
        return route

    route["protocol"] = protocol
    route["api_base"] = _api_base(host, path, protocol, suffix)
    package = entry.get("ai_sdk_package") or ""
    if package and package not in _EXPECTED_PACKAGES[protocol]:
        route["corroboration_warning"] = (
            f'protocol corroboration: {route.get("model_id") or ""} endpoint path '
            f'resolves {protocol} but ai_sdk_package is "{package}"'
        )
    route["registered"] = True
    return route


def analyze_routes(entries):
    """Derive registration routes for unified snapshot entries, in order."""
    return [derive_route(entry) for entry in entries]


def corroboration_warnings(routes):
    """Scrape warnings for endpoint-path vs AI SDK Package disagreements."""
    return [route["corroboration_warning"] for route in routes if route["corroboration_warning"]]


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
    """Register models from the unified snapshot, routed by endpoint path.

    Registration serves the snapshot file whatever its age and never scrapes
    for it; only when no snapshot has ever been written does it run a single
    blocking bootstrap scrape (failure to bootstrap registers nothing).
    Models the snapshot cannot vouch for are skipped, never guessed: no docs
    row → ``protocol unknown: no docs row``; a Google-style resource path →
    ``unsupported protocol: google-style``. Corroboration disagreement warns
    (it surfaces in ``detail``) but never vetoes — registration prints nothing.
    """
    from llm_opencode import (
        OpenCodeGoAnthropicAsyncChat,
        OpenCodeGoAnthropicChat,
        OpenCodeGoAsyncChat,
        OpenCodeGoChat,
        OpenCodeGoResponsesAsyncChat,
        OpenCodeGoResponsesChat,
        fetch_unified_models,
        read_snapshot,
    )

    key = llm.get_key("", "opencode", "OPENCODE_KEY")
    if not key:
        return
    snapshot = read_snapshot()
    if snapshot is None:
        try:
            snapshot = fetch_unified_models()
        except Exception:
            return
    entries = snapshot.get("models", []) if isinstance(snapshot, dict) else snapshot
    for route in analyze_routes(entries):
        if not route["registered"]:
            continue
        llm_model_id = f"opencode-go/{route['model_id']}"
        if route["protocol"] == "openai":
            register(
                OpenCodeGoChat(
                    model_id=llm_model_id,
                    model_name=route["model_id"],
                    api_base=route["api_base"],
                ),
                OpenCodeGoAsyncChat(
                    model_id=llm_model_id,
                    model_name=route["model_id"],
                    api_base=route["api_base"],
                ),
            )
        elif route["protocol"] == "openai-responses":
            register(
                OpenCodeGoResponsesChat(
                    model_id=llm_model_id,
                    model_name=route["model_id"],
                    api_base=route["api_base"],
                ),
                OpenCodeGoResponsesAsyncChat(
                    model_id=llm_model_id,
                    model_name=route["model_id"],
                    api_base=route["api_base"],
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
