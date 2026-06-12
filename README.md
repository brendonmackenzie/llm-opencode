# llm-opencode

An [LLM](https://llm.datasette.io/) plugin for [OpenCode Go](https://opencode.ai/docs/go/) subscription models.

## Installation

Install this plugin in the same environment as LLM:

```bash
llm install llm-opencode
```

Or install from the project directory:

```bash
llm install -e /path/to/llm-opencode
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

| Model ID | Protocol | Description |
|----------|----------|-------------|
| `glm-5` | OpenAI | GLM-5 |
| `glm-5.1` | OpenAI | GLM-5.1 |
| `kimi-k2.5` | OpenAI | Kimi K2.5 |
| `kimi-k2.6` | OpenAI | Kimi K2.6 |
| `mimo-v2` | OpenAI | MiMo V2 |
| `mimo-v2-pro` | OpenAI | MiMo V2 Pro |
| `mimo-v2.5` | OpenAI | MiMo V2.5 |
| `mimo-v2.5-pro` | OpenAI | MiMo V2.5 Pro |
| `mimo-v2-omni` | OpenAI | MiMo V2 Omni |
| `deepseek-v4-pro` | OpenAI | DeepSeek V4 Pro |
| `deepseek-v4-flash` | OpenAI | DeepSeek V4 Flash |
| `minimax-m2.5` | Anthropic | MiniMax M2.5 |
| `minimax-m2.7` | Anthropic | MiniMax M2.7 |
| `minimax-m3` | Anthropic | MiniMax M3 |
| `qwen3.5-plus` | Anthropic | Qwen3.5 Plus |
| `qwen3.6-plus` | Anthropic | Qwen3.6 Plus |
| `qwen3.7-plus` | Anthropic | Qwen3.7 Plus |
| `qwen3.7-max` | Anthropic | Qwen3.7 Max |

The model list is fetched dynamically from the OpenCode Go API and may change as new models are added.

## Options

### Anthropic-protocol models

Anthropic-protocol models (MiniMax, Qwen) support the following options:

- `-o max_tokens N` - Maximum number of tokens to generate (default: 4096)
- `-o temperature F` - Temperature for sampling (0.0-1.0)

Example:

```bash
llm -m opencode-go/minimax-m3 -o max_tokens 100 -o temperature 0.7 "Hello"
```

## Development

To set up for development:

```bash
cd llm-opencode
pip install -e '.[test]'
```

Run tests:

```bash
pytest tests/
```