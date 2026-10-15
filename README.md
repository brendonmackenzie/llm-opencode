# llm-opencode

An [LLM](https://llm.datasette.io/) plugin for [OpenCode Go](https://opencode.ai/docs/go/) subscription models.

> [!WARNING]
> This plugin is **alpha software**. The API and CLI surface may change without notice, and some features may be removed. Expect breaking changes between releases. Pin a commit in the install command if you need stability.

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

List available models (fully offline — reads the local snapshot, or the cached
API model list when no snapshot has been scraped yet):

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
llm opencode detail                       # pricing table (default)
llm opencode detail --format json         # full JSON
llm opencode detail --format csv          # CSV export
llm opencode detail --format pricing      # sorted by input cost
llm opencode models detail                # alias for `llm opencode detail`
```

Filter the details:

```bash
llm opencode detail --search qwen
llm opencode detail --plan go_plus
llm opencode detail --model minimax-m3
llm opencode detail --format csv -o models.csv
```

By default, models are sorted by **plan value** (requests per month per dollar —
best deal first), tie-broken by input price. The `detail` table also shows this
metric as its **Plan Value** column (e.g. `2,636 req/$`). Change the sort order
with `--sort-by`:

```bash
llm opencode detail                           # best plan value first (default)
llm opencode detail --sort-by input           # cheapest per-token first
llm opencode detail --sort-by monthly-limit   # lowest monthly cap first
llm opencode detail --sort-by name            # alphabetical by display name
llm opencode detail --sort-by model_id        # by model ID
```

The `pricing` format always sorts by input price ascending.

### Snapshot resilience

Model metadata is served from a single snapshot file, `opencode_unified_models.json`
in the llm user dir, whose payload carries `scraped_at`, scrape `warnings`, and the
merged `models` list:

- `llm opencode detail` is the sole refresh point. Its 24-hour timeout decides
  only whether a fresh docs scrape is *attempted*: a fresh snapshot is served
  as-is, and an expired-but-present snapshot gets one refresh attempt — when
  the attempt fails (for example with networking disabled), the last-good
  snapshot is served with a staleness banner. Only when no snapshot file has
  ever been written does `llm opencode detail` fail with a link to the docs
  page.
- Model registration never scrapes (the snapshot is served whatever its age,
  and prints nothing), except for a one-time blocking bootstrap scrape the
  very first time no snapshot file exists; if the bootstrap fails, no models
  are registered. Delete the snapshot file (or install on a fresh machine)
  and running `llm opencode detail` performs the scrape that repopulates it
  and re-registration picks models back up.

- Plain listing (`llm opencode models`, or bare `llm opencode`) never touches
  the network: it reads the snapshot (IDs + protocols), or the cached API model
  list when no snapshot has ever been written (IDs with
  `unknown (no scrape data)` protocol markers). A fresh install — neither file
  present — prints a hint to run `llm opencode detail` instead.

## Available Models

The model list is served from the local snapshot or cache; run `llm opencode detail` to populate it (the only command that fetches).

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
