# 🤖 `llm` - Starlark Module for AI and LLM Services

A powerful Starlark module for interacting with OpenAI and OpenAI-compatible API services. Easily generate text using chat completions and create images with DALL-E from your Starlark scripts.

## Features

- Chat completions using OpenAI GPT models
- Image generation using DALL-E models
- Support for OpenAI API
- Support for Azure OpenAI services
- Support for multimodal inputs (text and images)
- **Streaming mode** for real-time responses
- Customizable retry behavior and error handling

## Installation

```bash
go get github.com/starpkg/llm
```

## Usage in Go

```go
package main

import (
	"fmt"
	"os"

	"github.com/1set/starlet"
	"github.com/starpkg/llm"
)

func main() {
	// Create a new LLM module with API key
	apiKey := os.Getenv("OPENAI_API_KEY")
	mod := llm.NewModuleWithConfig("openai", "", apiKey, "gpt-4o", "dall-e-3")

	// Create a Starlet interpreter with the module
	interpreter := starlet.New(
		starlet.WithModuleLoader("llm", mod.LoadModule()),
	)

	// Run a Starlark script
	script := `
load("llm", "chat", "draw")

# Generate text using GPT
response = chat(
    text="Explain quantum computing in simple terms.",
    model="gpt-4o",
    max_tokens=100,
)
print("GPT response:", response)

# Generate an image using DALL-E
image_url = draw(
    prompt="A cute robot explaining quantum computing to children",
    model="dall-e-3",
    size="1024x1024",
)
print("Image URL:", image_url)
`

	// Execute the script
	if err := interpreter.ExecScript("example.star", script); err != nil {
		fmt.Println("Error:", err)
	}
}
```

## Starlark API

### Configuration

The module has the following configuration options:

- `openai_provider`: The provider type ("openai", "azure", or "anthropic")
- `openai_endpoint_url`: The API endpoint URL (required for Azure, optional for OpenAI)
- `openai_api_key`: The API key (required)
- `openai_gpt_model`: The default GPT model to use
- `openai_dalle_model`: The default DALL-E model to use

### Functions

#### `message(role?, text?, image?, image_file?, image_url?)`

Creates a message object for the chat function. Parameters:

- `role`: The role of the message (default: "user")
- `text`: The text content of the message
- `image`: Raw image data to include in the message
- `image_file`: Path to an image file to include in the message
- `image_url`: URL of an image to include in the message

Returns a dictionary representing the message.

#### `chat(text?, image?, image_file?, image_url?, messages?, model?, n?, max_tokens?, max_completion_tokens?, temperature?, top_p?, frequency_penalty?, presence_penalty?, stop?, response_format?, reasoning_effort?, retry?, full_response?, allow_error?, stream?, stream_callback?, kwargs?)`

Sends a chat completion request to the OpenAI API. Parameters:

- `text`: Text content for a user message
- `image`: Raw image data to include in the user message
- `image_file`: Path to an image file to include in the user message
- `image_url`: URL of an image to include in the user message
- `messages`: List of message dictionaries (from `message()` function)
- `model`: Model to use (defaults to `openai_gpt_model` config)
- `n`: Number of completions to generate (default: 1)
- `max_tokens`: Maximum number of tokens to generate (deprecated for o1 series models)
- `max_completion_tokens`: Upper bound for generated completion tokens (for o1 series models)
- `temperature`: Sampling temperature (default: 1.0)
- `top_p`: Nucleus sampling parameter (default: 1.0)
- `frequency_penalty`: Frequency penalty (default: 0.0)
- `presence_penalty`: Presence penalty (default: 0.0)
- `stop`: List of stop sequences
- `response_format`: Format of the response ("text" or "json") (default: "text")
- `reasoning_effort`: Controls reasoning effort for reasoning-capable models ("low", "medium", or "high")
- `kwargs`: Dictionary of additional parameters to pass to the API (for custom or non-standard parameters)
- `retry`: Number of retry attempts (default: 1)
- `full_response`: Return the full API response (default: false)
- `allow_error`: Return None instead of an error (default: false)
- `stream`: Enable streaming mode (default: false)
- `stream_callback`: Function to call for each chunk in streaming mode

Returns the generated text or a list of generated texts if `n > 1`. In streaming mode, the return value is constructed by combining all chunks. When `full_response=True` in streaming mode, the response includes token usage information accumulated from all stream chunks.

#### `draw(prompt, model?, n?, quality?, size?, style?, response_format?, background?, moderation?, output_format?, output_compression?, retry?, full_response?, allow_error?)`

Generates an image using DALL-E or GPT Image 1. Parameters:

- `prompt`: Text prompt for the image generation (required)
  - Max length: 32000 characters for gpt-image-1, 4000 for dall-e-3, 1000 for dall-e-2
- `model`: Model to use (defaults to `openai_dalle_model` config)
  - Supported models: "dall-e-2", "dall-e-3", "gpt-image-1"
- `n`: Number of images to generate (default: 1)
  - dall-e-2/gpt-image-1: 1-10 images, dall-e-3: only 1 image supported
- `quality`: Image quality (default: "auto" for gpt-image-1, "standard" for DALL-E)
  - GPT Image 1: "auto", "high", "medium", "low"
  - DALL-E 3: "standard", "hd"
  - DALL-E 2: "standard" only
- `size`: Image size (default: "auto" for gpt-image-1, "1024x1024" for DALL-E)
  - GPT Image 1: "auto", "1024x1024", "1536x1024" (landscape), "1024x1536" (portrait)
  - DALL-E 3: "1024x1024", "1792x1024", "1024x1792"
  - DALL-E 2: "256x256", "512x512", "1024x1024"
- `style`: Image style (DALL-E 3 only, default: "vivid")
  - DALL-E 3: "vivid", "natural"
- `response_format`: Response format (DALL-E only, default: "url")
  - DALL-E 2/3: "url", "b64_json"
  - GPT Image 1: Always returns base64-encoded images (parameter ignored)
- `background`: Background type (GPT Image 1 only, default: "auto")
  - GPT Image 1: "auto", "transparent", "opaque"
- `moderation`: Content moderation level (GPT Image 1 only, default: "auto")
  - GPT Image 1: "auto", "low"
- `output_format`: Output image format (GPT Image 1 only, default: "png")
  - GPT Image 1: "png", "jpeg", "webp"
- `output_compression`: Compression level 0-100 (GPT Image 1 only, default: 100)
  - GPT Image 1: Only supported with "jpeg" or "webp" output formats
- `retry`: Number of retry attempts (default: 1)
- `full_response`: Return the full API response (default: false)
- `allow_error`: Return None instead of an error (default: false)

Returns the image URL (DALL-E) or base64-encoded image data (GPT Image 1), or a list if `n > 1`.

## Examples

### Chat Completion

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

### Image Generation

```python
load("llm", "draw")

# Generate with DALL-E 3 (existing functionality)
dalle_image = draw(
    prompt="A futuristic city with flying cars and tall skyscrapers",
    model="dall-e-3",
    quality="hd",
    size="1024x1024",
    style="vivid"
)
print("DALL-E 3 image URL:", dalle_image)

# Generate with GPT Image 1 (new functionality)
gpt_image = draw(
    prompt="A realistic portrait of a robot scientist in a laboratory",
    model="gpt-image-1",
    quality="high",
    size="1024x1536",  # Portrait orientation
    background="transparent",
    output_format="png"
)
print("GPT Image 1 base64 data length:", len(gpt_image))

# Using content moderation with GPT Image 1
moderated_image = draw(
    prompt="A family-friendly cartoon character playing in a park",
    model="gpt-image-1",
    quality="medium",
    moderation="low",
    output_format="webp",
    output_compression=85
)
print("Moderated image data:", moderated_image[:100] + "...")

# Generate multiple images with GPT Image 1
multiple_images = draw(
    prompt="Abstract geometric patterns in bright colors",
    model="gpt-image-1",
    n=3,
    quality="medium",
    size="1024x1024",
    output_format="jpeg",
    output_compression=90
)
print(f"Generated {len(multiple_images)} images")

# Get full response with token usage information
full_resp = draw(
    prompt="An artistic illustration of AI concepts and neural networks",
    model="gpt-image-1",
    quality="high",
    full_response=True
)
print("Image data:", full_resp.data[0].b64_json[:100] + "...")
if hasattr(full_resp, "usage"):
    usage = full_resp.usage
    print(f"Total tokens: {usage.total_tokens}")
    print(f"Input tokens: {usage.input_tokens}")
    print(f"Output tokens: {usage.output_tokens}")
```

### Streaming Mode

```python
load("llm", "chat")

# Function to handle each chunk of the response
def handle_chunk(chunk):
    # Access the delta content from the first choice
    if len(chunk.choices) > 0:
        delta = chunk.choices[0].delta
        if delta.content:
            # Print each chunk as it arrives
            print(delta.content, end="", flush=True)

# Stream response with a callback
full_response = chat(
    text="Write a short poem about coding in Python, one line at a time.",
    max_tokens=200,
    stream=True,
    stream_callback=handle_chunk,
)

# The full_response contains the complete aggregated text
print("\n\nFull response:", full_response)
```

### Streaming with Progress Tracking

```python
load("llm", "chat")

# Create a simple progress tracker
tracker = {
    "tokens": 0,
    "started": False,
    "done": False
}

def process_chunk(chunk):
    if not tracker["started"]:
        print("Generating response...")
        tracker["started"] = True
    
    # Count tokens received
    if len(chunk.choices) > 0:
        delta = chunk.choices[0].delta
        if delta.content:
            tracker["tokens"] += 1
            
            # Show progress
            if tracker["tokens"] % 10 == 0:
                print(".", end="", flush=True)

# Generate a longer response with progress tracking
response = chat(
    text="Explain how transformers work in machine learning, with detailed technical information.",
    max_tokens=500,
    stream=True,
    stream_callback=process_chunk
)

print("\nDone! Received", tracker["tokens"], "chunks.")
print("\nFinal response:\n", response)
```

### Accessing Token Usage Information

```python
load("llm", "chat")

# Get the full response with token usage information
full_resp = chat(
    text="Explain the concept of transfer learning in AI.",
    max_tokens=300,
    stream=True,
    full_response=True
)

# Access token usage information
if hasattr(full_resp, "usage"):
    usage = full_resp.usage
    print(f"Prompt tokens: {usage.prompt_tokens}")
    print(f"Completion tokens: {usage.completion_tokens}")
    print(f"Total tokens: {usage.total_tokens}")
    
    # Calculate approximate cost (example rates for gpt-4)
    # Note: Token counts are accumulated from all stream responses for better accuracy
    prompt_cost = usage.prompt_tokens * 0.00003  # $0.03 per 1000 tokens
    completion_cost = usage.completion_tokens * 0.00006  # $0.06 per 1000 tokens
    total_cost = prompt_cost + completion_cost
    
    print(f"Approximate cost: ${total_cost:.6f}")
```

### Multimodal Interaction

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

### Advanced Chat with Message History

```python
load("llm", "message", "chat")

# Create a conversation history
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

### Using Reasoning Models

```python
load("llm", "chat")

# Set endpoint and API key for a reasoning-capable model provider
llm.set_openai_endpoint_url("https://reasoning-model-api-endpoint.com")
llm.set_openai_api_key("your-api-key-here")

# Call a reasoning-capable model with specific reasoning effort
response = chat(
    text="Solve this step by step: If 3x + 7 = 22, what is the value of x?",
    model="reasoning-model-name",
    reasoning_effort="high",  # Can be "low", "medium", or "high"
    max_tokens=300,
    full_response=True,
)

# If the model provides reasoning content, it will be available
if hasattr(response.choices[0].message, "reasoning_content"):
    print("Reasoning:")
    print(response.choices[0].message.reasoning_content)
    
print("\nFinal answer:")
print(response.choices[0].message.content)
```

### Using Custom Parameters with kwargs

```python
load("llm", "chat")

# Example 1: Using kwargs for custom or experimental parameters
# Some API providers or custom deployments may support additional parameters
response = chat(
    text="Generate a creative story about space exploration.",
    max_tokens=200,
    kwargs={
        "custom_parameter": "value",
        "experimental_feature": True,
        "custom_config": {
            "setting_a": "option1",
            "setting_b": 42
        }
    }
)
print(response)

# Example 2: Using kwargs for provider-specific parameters
# Different OpenAI-compatible providers may have unique parameters
response = chat(
    text="What are the benefits of renewable energy?",
    max_tokens=150,
    kwargs={
        "provider_specific_param": "custom_value",
        "optimization_level": "high",
        "cache_enabled": True
    }
)
print(response)

# Example 3: Combining standard parameters with custom kwargs
response = chat(
    text="Explain machine learning in simple terms.",
    model="gpt-4",
    temperature=0.7,
    max_tokens=200,
    kwargs={
        "safety_level": "strict",
        "response_style": "educational",
        "custom_instruction": "Use analogies when possible"
    },
    full_response=True
)

# Access both standard response and any custom fields
print("Content:", response.choices[0].message.content)
if hasattr(response, 'custom_fields'):
    print("Custom response fields:", response.custom_fields)
```

## License

This package is licensed under the MIT License - see the LICENSE file for details.