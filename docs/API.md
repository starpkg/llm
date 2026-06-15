# `llm` — Starlark API Reference

The complete reference for every script-facing builtin and configuration
accessor exposed by the `llm` module. For an overview, installation, and a
quickstart, see the [README](../README.md).

The module exposes three top-level builtins via `load("llm", …)` — `message`,
`chat`, and `draw` — plus a set of configuration accessors (`get_<key>` /
`set_<key>`) generated from the module's options. Module options serve as
defaults: they are used when the corresponding argument is not passed to `chat`
/ `draw`.

## Contents

- [Functions](#functions)
  - [`message`](#messagerole-text-image-image_file-image_url)
  - [`chat`](#chattext-image-image_file-image_url-messages-model-n-max_tokens-max_completion_tokens-temperature-top_p-frequency_penalty-presence_penalty-stop-response_format-reasoning_effort-retry-full_response-allow_error-stream-stream_callback-kwargs)
  - [`draw`](#drawprompt-model-n-quality-size-style-response_format-background-moderation-output_format-output_compression-retry-full_response-allow_error)
- [Examples](#examples)
- [Configuration](#configuration)

## Functions

### `message(role?, text?, image?, image_file?, image_url?)`

Creates a message object for the `chat` function. It is data only — no API call
is made. `chat`'s `messages=[...]` consumes exactly these objects.

**Parameters:**

- `role` (string): The role of the message (default: `"user"`)
- `text` (string): The text content of the message
- `image` (string/bytes): Raw image data to include in the message
- `image_file` (string): Path to an image file to include in the message
- `image_url` (string): URL of an image to include in the message

**Returns:** A dictionary representing the message, containing only the
non-empty subset of `role` / `text` / `image` / `image_file` / `image_url`.

**Example:**

```python
load("llm", "message")

msg = message(role="user", text="Hello, who are you?")
```

### `chat(text?, image?, image_file?, image_url?, messages?, model?, n?, max_tokens?, max_completion_tokens?, temperature?, top_p?, frequency_penalty?, presence_penalty?, stop?, response_format?, reasoning_effort?, retry?, full_response?, allow_error?, stream?, stream_callback?, kwargs?)`

Sends a chat completion request to the OpenAI (or OpenAI-compatible) API. The
top-level `text` / `image` / `image_file` / `image_url` arguments are assembled
into an implicit `user` message that is prepended to any `messages` list.

**Parameters:**

- `text` (string): Text content for a user message
- `image` (string/bytes): Raw image data to include in the user message
- `image_file` (string): Path to an image file to include in the user message
- `image_url` (string): URL of an image to include in the user message
- `messages` (list): List of message dictionaries (from the `message` function)
- `model` (string): Model to use (defaults to the `openai_gpt_model` config option)
- `n` (int): Number of completions to generate (default: `1`)
- `max_tokens` (int): Maximum number of tokens to generate; `0` (the default)
  leaves it unset so the API applies its own default. Deprecated for o1-series
  models — use `max_completion_tokens` instead
- `max_completion_tokens` (int): Upper bound for generated completion tokens
  (for o1/o3/o4-series models); `0` (the default) leaves it unset
- `temperature` (float): Sampling temperature (default: `1.0`)
- `top_p` (float): Nucleus sampling parameter (default: `1.0`)
- `frequency_penalty` (float): Frequency penalty (default: `0.0`)
- `presence_penalty` (float): Presence penalty (default: `0.0`)
- `stop` (string/list): One or more stop sequences
- `response_format` (string): Format of the response, `"text"` or `"json"`
  (default: `"text"`)
- `reasoning_effort` (string): Controls reasoning effort for reasoning-capable
  models — `"low"`, `"medium"`, or `"high"`
- `retry` (int): Number of attempts (default: `1`). Retries stop immediately on
  an HTTP 400 (bad request), which can never succeed on retry
- `full_response` (bool): Return the full converted API response object instead
  of the content (default: `False`)
- `allow_error` (bool): Return `None` instead of raising on request failure
  (default: `False`)
- `stream` (bool): Enable streaming mode (default: `False`)
- `stream_callback` (callable): Function invoked with each chunk (converted to a
  Starlark value) as it arrives in streaming mode
- `kwargs` (dict): Additional custom or provider-specific parameters; passed
  through to the API request's `ChatTemplateKwargs`

**Returns:** The generated content string when `n == 1`, or a list of content
strings when `n > 1`. With `full_response=True`, the converted full API response
object (its shape depends on `legacy_mode` — see
[Configuration](#configuration)). In streaming mode the content is aggregated
from all chunks, and the full response (when requested) includes token usage
accumulated across chunks. Returns `None` if the response contains no choices.

**Errors:** Returns an error (or `None` under `allow_error=True`) when no model
is set, on an unsupported `response_format`, on invalid `kwargs`, or on any API
/ network failure after the configured retries.

### `draw(prompt, model?, n?, quality?, size?, style?, response_format?, background?, moderation?, output_format?, output_compression?, retry?, full_response?, allow_error?)`

Generates an image using DALL-E 2, DALL-E 3, or GPT Image 1. The model is
selected by `model` (or the `openai_dalle_model` config option), and the valid
parameter set / defaults depend on which model is chosen.

**Parameters:**

- `prompt` (string, required): Text prompt for the image generation
  - Max length: 32000 characters for `gpt-image-1`, 4000 for `dall-e-3`, 1000
    for `dall-e-2`
- `model` (string): Model to use (defaults to the `openai_dalle_model` config
  option). Supported: `"dall-e-2"`, `"dall-e-3"`, `"gpt-image-1"`
- `n` (int): Number of images to generate (default: `1`)
  - `dall-e-2` / `gpt-image-1`: 1-10 images; `dall-e-3`: only `1` is supported
- `quality` (string): Image quality (default: `"auto"` for `gpt-image-1`,
  `"standard"` for DALL-E)
  - GPT Image 1: `"auto"`, `"high"`, `"medium"`, `"low"`
  - DALL-E 3: `"standard"`, `"hd"`
  - DALL-E 2: `"standard"` only
- `size` (string): Image size (default: `"auto"` for `gpt-image-1`,
  `"1024x1024"` for DALL-E)
  - GPT Image 1: `"auto"`, `"1024x1024"`, `"1536x1024"` (landscape),
    `"1024x1536"` (portrait)
  - DALL-E 3: `"1024x1024"`, `"1792x1024"`, `"1024x1792"`
  - DALL-E 2: `"256x256"`, `"512x512"`, `"1024x1024"`
- `style` (string): Image style — DALL-E 3 only (default: `"vivid"`)
  - DALL-E 3: `"vivid"`, `"natural"`
- `response_format` (string): Response format — DALL-E only (default: `"url"`)
  - DALL-E 2/3: `"url"`, `"b64_json"`
  - GPT Image 1: ignored; the API returns base64, which the module decodes and
    returns as raw image bytes
- `background` (string): Background type — GPT Image 1 only (default: `"auto"`)
  - GPT Image 1: `"auto"`, `"transparent"`, `"opaque"`
- `moderation` (string): Content moderation level — GPT Image 1 only (default:
  `"auto"`)
  - GPT Image 1: `"auto"`, `"low"`
- `output_format` (string): Output image format — GPT Image 1 only (default:
  `"png"`)
  - GPT Image 1: `"png"`, `"jpeg"`, `"webp"`
- `output_compression` (int): Compression level 0-100 — GPT Image 1 only
  (default: `100`)
  - GPT Image 1: only supported with `"jpeg"` or `"webp"` output formats
- `retry` (int): Number of attempts (default: `1`). Retries stop immediately on
  an HTTP 400 (bad request)
- `full_response` (bool): Return the full converted API response object instead
  of the content (default: `False`)
- `allow_error` (bool): Return `None` instead of raising on request failure
  (default: `False`)

**Returns:** A URL string (DALL-E with `response_format="url"`, the default) or
raw image bytes (DALL-E with `response_format="b64_json"`, and always for GPT
Image 1 — the base64 returned by the API is decoded before it reaches the
script). When `n > 1`, returns a list of those values. With
`full_response=True`, returns the converted full API response instead.

**Errors:** Returns an error (or `None` under `allow_error=True`) when `prompt`
is empty, no model is set, a model-specific parameter is out of range (e.g.
`dall-e-3` with `n > 1`, an invalid `background` / `moderation` / `output_format`
/ `quality` / `size`, `output_compression` outside 0-100), the response carries
no image data, or on any API / network failure after the configured retries.

## Examples

### Chat completion

```python
load("llm", "chat")

# Simple text generation
response = chat(
    text="What are the three laws of robotics?",
    max_tokens=200,
)
print(response)

# Using JSON mode
json_resp = chat(
    text="Generate a JSON object with the three laws of robotics. Include each law as a separate field.",
    response_format="json",
    max_tokens=200,
)
print(json_resp)
```

### Image generation

```python
load("llm", "draw")

# Generate with DALL-E 3
dalle_image = draw(
    prompt="A futuristic city with flying cars and tall skyscrapers",
    model="dall-e-3",
    quality="hd",
    size="1024x1024",
    style="vivid",
)
print("DALL-E 3 image URL:", dalle_image)

# Generate with GPT Image 1 (returns raw image bytes)
gpt_image = draw(
    prompt="A realistic portrait of a robot scientist in a laboratory",
    model="gpt-image-1",
    quality="high",
    size="1024x1536",  # Portrait orientation
    background="transparent",
    output_format="png",
)
print("GPT Image 1 image bytes length:", len(gpt_image))

# Content moderation with GPT Image 1
moderated_image = draw(
    prompt="A family-friendly cartoon character playing in a park",
    model="gpt-image-1",
    quality="medium",
    moderation="low",
    output_format="webp",
    output_compression=85,
)
print("Moderated image bytes length:", len(moderated_image))

# Generate multiple images with GPT Image 1
multiple_images = draw(
    prompt="Abstract geometric patterns in bright colors",
    model="gpt-image-1",
    n=3,
    quality="medium",
    size="1024x1024",
    output_format="jpeg",
    output_compression=90,
)
print("Generated {} images".format(len(multiple_images)))

# Get the full response with token usage information
full_resp = draw(
    prompt="An artistic illustration of AI concepts and neural networks",
    model="gpt-image-1",
    quality="high",
    full_response=True,
)
print("Image data:", full_resp.data[0].b64_json[:100] + "...")
if hasattr(full_resp, "usage"):
    usage = full_resp.usage
    print("Total tokens: {}".format(usage.total_tokens))
    print("Input tokens: {}".format(usage.input_tokens))
    print("Output tokens: {}".format(usage.output_tokens))
```

### Streaming mode

```python
load("llm", "chat")

# Handle each chunk of the response
def handle_chunk(chunk):
    # Access the delta content from the first choice
    if len(chunk.choices) > 0:
        delta = chunk.choices[0].delta
        if delta.content:
            # Print each chunk as it arrives (Starlark's print always adds a newline)
            print(delta.content)

# Stream the response with a callback
full_response = chat(
    text="Write a short poem about coding in Python, one line at a time.",
    max_tokens=200,
    stream=True,
    stream_callback=handle_chunk,
)

# The full_response contains the complete aggregated text
print("\n\nFull response:", full_response)
```

### Streaming with progress tracking

```python
load("llm", "chat")

# A simple progress tracker
tracker = {
    "tokens": 0,
    "started": False,
    "done": False,
}

def process_chunk(chunk):
    if not tracker["started"]:
        print("Generating response...")
        tracker["started"] = True

    # Count chunks received
    if len(chunk.choices) > 0:
        delta = chunk.choices[0].delta
        if delta.content:
            tracker["tokens"] += 1

            # Show progress (Starlark's print always adds a newline)
            if tracker["tokens"] % 10 == 0:
                print(".")

# Generate a longer response with progress tracking
response = chat(
    text="Explain how transformers work in machine learning, with detailed technical information.",
    max_tokens=500,
    stream=True,
    stream_callback=process_chunk,
)

print("\nDone! Received", tracker["tokens"], "chunks.")
print("\nFinal response:\n", response)
```

### Accessing token usage information

```python
load("llm", "chat")

# Get the full response with token usage information
full_resp = chat(
    text="Explain the concept of transfer learning in AI.",
    max_tokens=300,
    stream=True,
    full_response=True,
)

# Access token usage information
if hasattr(full_resp, "usage"):
    usage = full_resp.usage
    print("Prompt tokens: {}".format(usage.prompt_tokens))
    print("Completion tokens: {}".format(usage.completion_tokens))
    print("Total tokens: {}".format(usage.total_tokens))

    # Calculate an approximate cost (example rates for gpt-4)
    # Note: token counts are accumulated from all stream responses for accuracy
    prompt_cost = usage.prompt_tokens * 0.00003  # $0.03 per 1000 tokens
    completion_cost = usage.completion_tokens * 0.00006  # $0.06 per 1000 tokens
    total_cost = prompt_cost + completion_cost

    print("Approximate cost: ${}".format(total_cost))
```

### Multimodal interaction

```python
load("llm", "chat")

# Ask about an image
response = chat(
    text="What's in this image?",
    image_url="https://example.com/image.jpg",
    max_tokens=150,
)
print(response)
```

### Advanced chat with message history

```python
load("llm", "message", "chat")

# Build a conversation history
messages = [
    message(role="user", text="Hello, who are you?"),
    message(role="assistant", text="I'm an AI assistant. How can I help you today?"),
    message(role="user", text="Can you explain what an LLM is?"),
]

# Continue the conversation
response = chat(
    messages=messages,
    max_tokens=200,
)
print(response)
```

### Using reasoning models

```python
load("llm", "chat", "set_openai_endpoint_url", "set_openai_api_key")

# Set the endpoint and API key for a reasoning-capable model provider
set_openai_endpoint_url("https://reasoning-model-api-endpoint.com")
set_openai_api_key("your-api-key-here")

# Call a reasoning-capable model with a specific reasoning effort
response = chat(
    text="Solve this step by step: If 3x + 7 = 22, what is the value of x?",
    model="reasoning-model-name",
    reasoning_effort="high",  # Can be "low", "medium", or "high"
    max_tokens=300,
    full_response=True,
)

# If the model provides reasoning content, it is available on the message
if hasattr(response.choices[0].message, "reasoning_content"):
    print("Reasoning:")
    print(response.choices[0].message.reasoning_content)

print("\nFinal answer:")
print(response.choices[0].message.content)
```

### Using custom parameters with kwargs

```python
load("llm", "chat")

# Use kwargs for custom, experimental, or provider-specific parameters
response = chat(
    text="Generate a creative story about space exploration.",
    max_tokens=200,
    kwargs={
        "custom_parameter": "value",
        "experimental_feature": True,
        "custom_config": {
            "setting_a": "option1",
            "setting_b": 42,
        },
    },
)
print(response)

# Combine standard parameters with custom kwargs
response = chat(
    text="Explain machine learning in simple terms.",
    model="gpt-4",
    temperature=0.7,
    max_tokens=200,
    kwargs={
        "safety_level": "strict",
        "response_style": "educational",
        "custom_instruction": "Use analogies when possible",
    },
    full_response=True,
)

# Access both the standard response and any custom fields
print("Content:", response.choices[0].message.content)
if hasattr(response, "custom_fields"):
    print("Custom response fields:", response.custom_fields)
```

## Configuration

The module is built on `starpkg/base`'s configurable-module system. Each option
is exposed to scripts as a pair of generated accessor builtins (loaded from the
`llm` module alongside the functions above):

- **`get_<key>()`** — returns the current value of the option.
- **`set_<key>(value)`** — sets the option (takes a single value).

An option's value resolves in priority order: an explicit `set_<key>` value, the
environment variable, then the default. These options serve as defaults used by
`chat` / `draw` when the corresponding argument is not provided.

`openai_api_key` is a **secret** option (`SetSecret(true)` in the code), so it
exposes **only** `set_openai_api_key` — there is **no** `get_openai_api_key`
builtin. The key can be set from a script but never read back.

| Option | Getter | Setter | Type | Env var | Default | Description |
|--------|--------|--------|------|---------|---------|-------------|
| `openai_provider` | `get_openai_provider` | `set_openai_provider` | string | `LLM_OPENAI_PROVIDER` | `openai` | Provider type: `openai`, `azure`, or `anthropic` |
| `openai_endpoint_url` | `get_openai_endpoint_url` | `set_openai_endpoint_url` | string | `LLM_OPENAI_ENDPOINT_URL` | `""` | API endpoint base URL (required for Azure; overrides the default for the others) |
| `openai_api_key` | _(secret — none)_ | `set_openai_api_key` | string | `LLM_OPENAI_API_KEY` | `""` | API key (required). Secret: no getter is exposed to scripts |
| `openai_gpt_model` | `get_openai_gpt_model` | `set_openai_gpt_model` | string | `LLM_OPENAI_GPT_MODEL` | `""` | Default model for `chat` |
| `openai_dalle_model` | `get_openai_dalle_model` | `set_openai_dalle_model` | string | `LLM_OPENAI_DALLE_MODEL` | `""` | Default model for `draw` |
| `api_version` | `get_api_version` | `set_api_version` | string | `LLM_API_VERSION` | `2024-02-01` | API version (used by Azure and Anthropic) |
| `legacy_mode` | `get_legacy_mode` | `set_legacy_mode` | bool | `LLM_LEGACY_MODE` | `true` | Conversion mode for `full_response` objects (see below) |

**Example:**

```python
load("llm", "set_openai_endpoint_url", "set_openai_api_key", "set_openai_gpt_model", "chat")

set_openai_endpoint_url("https://api.openai.com/v1")
set_openai_api_key("sk-...")          # secret: no get_openai_api_key
set_openai_gpt_model("gpt-4o")

print(chat(text="Hello!"))
```

These accessors are also reachable as module attributes — e.g.
`llm.set_openai_api_key("sk-...")` — when the module is loaded under its `llm`
name rather than via `load(...)` of individual symbols.

### `legacy_mode` and `full_response`

`legacy_mode` controls how `full_response=True` results are turned into Starlark
values. When `true` (the default) the response is exposed via direct struct
access (field names like `choices[0].message.content`). When `false`, the
response is converted through JSON. The default preserves the historical
behavior; flip it with `set_legacy_mode(False)` only if you need JSON-shaped
access.
