import llm

OPENAI_PROTOCOL_MODELS = {
    "glm-5",
    "glm-5.1",
    "glm-5.2",
    "glm-5.3",
    "glm-5.3-flash",
    "kimi-k2.5",
    "kimi-k2.6",
    "kimi-k2.7-code",
    "kimi-k3",
    "mimo-v2",
    "mimo-v2-pro",
    "mimo-v2.5",
    "mimo-v2.5-pro",
    "mimo-v2.6-flash",
    "mimo-v2.6-pro",
    "mimo-v2-omni",
    "deepseek-v4-pro",
    "deepseek-v4-flash",
    "deepseek-v4-flash-vision-exp",
    "deepseek-v4.1-flash",
    "deepseek-flash",
    "hy3",
    "hy4-preview",
    "longcat-2.0",
    "longcat-2.5-preview-free",
    "space-bunny",
}

OPENAI_RESPONSES_PROTOCOL_MODELS = {
    "gpt-5.6-luna",
    "gpt-6-luna",
    "grok-4.6",
    "grok-4.7",
    "muse-spark-1.2-contributor",
    "muse-spark-1.3-contributor",
}

ANTHROPIC_PROTOCOL_MODELS = {
    "minimax-m3",
    "minimax-m2.7",
    "minimax-m2.5",
    "qwen3.5-plus",
    "qwen3.6-plus",
    "qwen3.7-plus",
    "qwen3.7-max",
    "qwen3.8-max",
    "qwen3.8-flash",
    "claude-haiku-5-5",
}

UNKNOWN_MODELS_DEFAULT = "openai"

BASE_URL_OPENAI = "https://opencode.ai/zen/go/v1"
BASE_URL_ANTHROPIC = "https://opencode.ai/zen/go"


def _get_protocol(model_id):
    if model_id in ANTHROPIC_PROTOCOL_MODELS:
        return "anthropic"
    if model_id in OPENAI_RESPONSES_PROTOCOL_MODELS:
        return "openai-responses"
    return "openai"


@llm.hookimpl
def register_models(register):
    from llm_opencode import (
        OpenCodeGoAnthropicAsyncChat,
        OpenCodeGoAnthropicChat,
        OpenCodeGoAsyncChat,
        OpenCodeGoChat,
        OpenCodeGoResponsesAsyncChat,
        OpenCodeGoResponsesChat,
        get_opencode_models,
    )

    key = llm.get_key("", "opencode", "OPENCODE_KEY")
    if not key:
        return
    for model_definition in get_opencode_models():
        model_id = model_definition["id"]
        llm_model_id = f"opencode-go/{model_id}"
        protocol = _get_protocol(model_id)

        if protocol == "openai":
            register(
                OpenCodeGoChat(
                    model_id=llm_model_id,
                    model_name=model_id,
                    api_base=BASE_URL_OPENAI,
                ),
                OpenCodeGoAsyncChat(
                    model_id=llm_model_id,
                    model_name=model_id,
                    api_base=BASE_URL_OPENAI,
                ),
            )
        elif protocol == "openai-responses":
            register(
                OpenCodeGoResponsesChat(
                    model_id=llm_model_id,
                    model_name=model_id,
                    api_base=BASE_URL_OPENAI,
                ),
                OpenCodeGoResponsesAsyncChat(
                    model_id=llm_model_id,
                    model_name=model_id,
                    api_base=BASE_URL_OPENAI,
                ),
            )
        elif protocol == "anthropic":
            register(
                OpenCodeGoAnthropicChat(model_id=llm_model_id),
                OpenCodeGoAnthropicAsyncChat(model_id=llm_model_id),
            )
