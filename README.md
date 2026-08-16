# llm-opencode

An [LLM](https://llm.datasette.io/) plugin for [OpenCode Go](https://opencode.ai/docs/go/) subscription models.

## Installation

Since the plugin is not yet on PyPI, install it directly from GitHub using `uv tool install`:

```bash
uv tool install llm --with "llm-opencode @ git+https://github.com/brendonmackenzie/llm-opencode.git"
```

If you already have `llm` installed with other plugins, add it to the existing install. For example:

```bash
uv tool install llm \
  --with "llm-opencode @ git+https://github.com/brendonmackenzie/llm-opencode.git" \
  --with llm-git --with llm-openrouter \ 
  --with llm-plugin-generator --with llm-uv-tool \
  --with toml --reinstall
```

To install from a local checkout during development:

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

## Model details

Show rich model metadata (pricing, rate limits, endpoints, and data retention
policies) merged from the OpenCode Go API and docs:

```bash
llm opencode models detail                # pricing table (default)
llm opencode models detail --format json  # full JSON
llm opencode models detail --format csv   # CSV export
llm opencode models detail --format pricing  # sorted by input cost
```

Filter the details:

```bash
llm opencode models detail --search qwen
llm opencode models detail --provider anthropic
llm opencode models detail --model minimax-m3
llm opencode models detail --format csv -o models.csv
```

By default, models are sorted by **plan value** (requests per month per dollar —
best deal first), tie-broken by input price. The `detail` table also shows this
metric as its **Plan Value** column (e.g. `2,636 req/$`). Change the sort order
with `--sort-by`:

```bash
llm opencode models detail                          # best plan value first (default)
llm opencode models detail --sort-by input          # cheapest per-token first
llm opencode models detail --sort-by usage          # cheapest plan first
llm opencode models detail --sort-by name           # alphabetical by display name
llm opencode models detail --sort-by model_id       # by model ID
```

The `pricing` format always sorts by input price ascending.

The unified data is cached locally for 24 hours. If the OpenCode docs page
cannot be fetched, the command prints a message with a link to the docs page
instead.

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
