# llm-opencode

An [LLM](https://llm.datasette.io/) plugin for [OpenCode Go](https://opencode.ai/docs/go/) subscription models.

## Installation

Install this plugin in the same environment as LLM:

```bash
llm install llm-opencode
```

Or with uv:

```bash
uv tool install llm --with llm-opencode
```

Or install from a local checkout:

```bash
uv tool install llm --with /path/to/llm-opencode --reinstall
```

## Configuration

Set your OpenCode Go API key:

```bash
llm keys set opencode
```

Paste your key when prompted. You can get an API key from the [OpenCode console](https://opencode.ai/auth).

Alternatively, set the `OPENCODE_KEY` environment variable.

## Usage

List available models:

```bash
llm opencode models
```

List all models registered with LLM:

```bash
llm models | grep opencode-go
```

Run a prompt with an OpenAI-protocol model:

```bash
llm -m opencode-go/deepseek-v4-flash "Explain quantum computing in one paragraph"
```

Run a prompt with an Anthropic-protocol model:

```bash
llm -m opencode-go/minimax-m3 "Write a haiku about programming"
```

Start an interactive chat:

```bash
llm chat -m opencode-go/glm-5.1
```

## Available Models

The model list is fetched dynamically from the OpenCode Go API. Run `llm opencode models` for the current list.

Available models broadly fall into two protocol groups:

| Protocol | Models |
|----------|--------|
| OpenAI | DeepSeek, GLM, Kimi K2.5/2.6, MiMo V2/V2.5, MiMo Omni |
| Anthropic | MiniMax M2.5/M2.7/M3, Qwen3.5/3.6/3.7 Plus/Max |

## Options

### Anthropic-protocol models

Anthropic-protocol models (MiniMax, Qwen) support the following options:

- `-o max_tokens N` — Maximum number of tokens to generate (default: 4096)
- `-o temperature F` — Temperature for sampling (0.0–1.0)

Example:

```bash
llm -m opencode-go/minimax-m3 -o max_tokens 100 -o temperature 0.7 "Hello"
```

## Development

Set up a local development environment with uv:

```bash
cd llm-opencode
uv venv
uv pip install -e '.[test]'
```

Run tests with coverage:

```bash
uv run pytest --cov=.
```

Reinstall the plugin into the llm tool after changes:

```bash
uv tool install llm --with /path/to/llm-opencode --reinstall
```
