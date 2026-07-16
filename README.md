# 🤖 `llm` — Starlark module for AI and LLM services

[![godoc](https://pkg.go.dev/badge/github.com/starpkg/llm.svg)](https://pkg.go.dev/github.com/starpkg/llm)
[![license](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/starpkg/llm/graph/badge.svg)](https://codecov.io/gh/starpkg/llm)
![binary footprint](https://img.shields.io/badge/binary_footprint-%2B0.6_MB-blue)

A Starlark module for interacting with OpenAI and OpenAI-compatible API
services. Generate text with chat completions and create images with DALL-E /
GPT Image 1, directly from your Starlark scripts.

## Overview

Within the Star\* ecosystem — *"starpkg = support for necessary LOCAL
operations + simple abstractions over common ONLINE services, for ease of
use"* — `llm` is an **online-service abstraction**. It wraps a remote SaaS API
behind three small script-facing builtins, so a script reaches a chat or image
model without juggling HTTP, auth headers, retries, or response shapes.

- **Chat completions** — `chat` for OpenAI GPT and reasoning models, with
  multi-turn conversations and multimodal inputs (text + images).
- **Image generation** — `draw` for DALL-E 2/3 and GPT Image 1.
- **Message builder** — `message` assembles a conversation message (text and/or
  image) for `chat`.
- **Many providers** — the OpenAI API, Azure OpenAI Service, the Anthropic
  Claude API, and any OpenAI-compatible endpoint, selected by `openai_provider`.
- **Streaming** — real-time responses with an optional per-chunk callback.
- **Robust calls** — customizable retry behavior, graceful error handling
  (`retry`, `allow_error`), and custom/provider-specific parameters via
  `kwargs`.

It is an L4 domain module: it depends downward on `starpkg/base` (the
module/config system), `1set/starlet` (the Machine + `dataconv`), and
transitively `1set/starlight` + `go.starlark.net`. Nothing in the ecosystem
depends on it.

For the complete per-builtin reference — signatures, parameters, returns,
errors, examples — and the configuration accessors, see
**[docs/API.md](docs/API.md)**.

## Installation

```bash
go get github.com/starpkg/llm
```

## Quick Start

Wire the module into a Starlet interpreter, then `load("llm", …)` from a script.
The Go layer provides two constructors: `NewModule()` (empty config) and
`NewModuleWithConfig(serviceProvider, endpointURL, apiKey, gptModel, dalleModel, apiVersion)`
(preset config).

```go
package main

import (
    "fmt"
    "os"

    "github.com/1set/starlet"
    "github.com/starpkg/llm"
)

func main() {
    apiKey := os.Getenv("OPENAI_API_KEY")
    mod := llm.NewModuleWithConfig("openai", "", apiKey, "gpt-4o", "dall-e-3", "")

    interpreter := starlet.New(
        starlet.WithModuleLoader("llm", mod.LoadModule()),
    )

    script := `
load("llm", "chat", "draw")

# Generate text using GPT
response = chat(text="Explain quantum computing in simple terms.", max_tokens=100)
print("GPT response:", response)

# Generate an image using DALL-E
image_url = draw(prompt="A cute robot explaining quantum computing", size="1024x1024")
print("Image URL:", image_url)
`

    if err := interpreter.ExecScript("example.star", script); err != nil {
        fmt.Println("Error:", err)
    }
}
```

Hold a multi-turn conversation, building messages with `message`:

```python
load("llm", "message", "chat")

messages = [
    message(role="user", text="Hello, who are you?"),
    message(role="assistant", text="I'm an AI assistant. How can I help you today?"),
    message(role="user", text="Can you explain what an LLM is?"),
]
print(chat(messages=messages, max_tokens=200))
```

## Starlark API at a glance

Top-level builtins (`load("llm", …)`):

- `message(role?, text?, image?, image_file?, image_url?)` — build a
  conversation message object for `chat`.
- `chat(text?, image?, image_file?, image_url?, messages?, model?, n?, max_tokens?, max_completion_tokens?, temperature?, top_p?, frequency_penalty?, presence_penalty?, stop?, response_format?, reasoning_effort?, retry?, full_response?, allow_error?, stream?, stream_callback?, kwargs?)`
  — send a chat completion request (blocking or streaming).
- `draw(prompt, model?, n?, quality?, size?, style?, response_format?, background?, moderation?, output_format?, output_compression?, retry?, full_response?, allow_error?)`
  — generate an image with DALL-E or GPT Image 1.

See **[docs/API.md](docs/API.md)** for the full signatures, return values,
errors, and examples of every builtin above.

## Configuration

The module's options (`openai_provider`, `openai_endpoint_url`,
`openai_api_key`, `openai_gpt_model`, `openai_dalle_model`, `api_version`,
`legacy_mode`, `request_timeout`) are configured via environment variables
(`LLM_*`) or per-option `get_<key>` / `set_<key>` accessor builtins, and serve as
defaults for `chat` / `draw`. `openai_api_key` is secret — it exposes only
`set_openai_api_key`, never a getter. `request_timeout` (default `120`s) bounds
each request (a total deadline for blocking `chat`/`draw`; a connect +
first-response bound for streaming, so long streams aren't truncated). See the
[Configuration section of docs/API.md](docs/API.md#configuration) for the full
option table, defaults, accessors, and the `legacy_mode` behavior.

**Trust model** — a script can point the client at its own provider/endpoint/key,
so a *host-injected* API key can be sent to a script-chosen endpoint, and
`image_file` reads arbitrary host files (bounded to 64 MiB). Only inject a host
key, and only enable host file access, for scripts you trust — see
[Safety / trust model](docs/API.md#safety--trust-model).

## License

This package is licensed under the MIT License — see the LICENSE file for
details.
