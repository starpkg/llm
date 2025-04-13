# 🤖 `llm` - Starlark Module for AI and LLM Services

A powerful Starlark module for interacting with OpenAI and OpenAI-compatible API services. Easily generate text using chat completions and create images with DALL-E from your Starlark scripts.

## Features

- Chat completions using OpenAI GPT models
- Image generation using DALL-E models
- Support for OpenAI API
- Support for Azure OpenAI services
- Support for multimodal inputs (text and images)
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

- `openai_provider`: The provider type ("openai" or "azure")
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

#### `chat(text?, image?, image_file?, image_url?, messages?, model?, n?, max_tokens?, temperature?, top_p?, frequency_penalty?, presence_penalty?, stop?, response_format?, retry?, full_response?, allow_error?)`

Sends a chat completion request to the OpenAI API. Parameters:

- `text`: Text content for a user message
- `image`: Raw image data to include in the user message
- `image_file`: Path to an image file to include in the user message
- `image_url`: URL of an image to include in the user message
- `messages`: List of message dictionaries (from `message()` function)
- `model`: Model to use (defaults to `openai_gpt_model` config)
- `n`: Number of completions to generate (default: 1)
- `max_tokens`: Maximum number of tokens to generate (default: 64)
- `temperature`: Sampling temperature (default: 1.0)
- `top_p`: Nucleus sampling parameter (default: 1.0)
- `frequency_penalty`: Frequency penalty (default: 0.0)
- `presence_penalty`: Presence penalty (default: 0.0)
- `stop`: List of stop sequences
- `response_format`: Format of the response ("text" or "json") (default: "text")
- `retry`: Number of retry attempts (default: 1)
- `full_response`: Return the full API response (default: false)
- `allow_error`: Return None instead of an error (default: false)

Returns the generated text or a list of generated texts if `n > 1`.

#### `draw(prompt, model?, n?, quality?, size?, style?, response_format?, retry?, full_response?, allow_error?)`

Generates an image using DALL-E. Parameters:

- `prompt`: Text prompt for the image generation (required)
- `model`: Model to use (defaults to `openai_dalle_model` config)
- `n`: Number of images to generate (default: 1)
- `quality`: Image quality ("standard", "hd") (default: "standard")
- `size`: Image size ("256x256", "512x512", "1024x1024", "1792x1024", "1024x1792") (default: "1024x1024")
- `style`: Image style ("vivid", "natural") (default: "vivid")
- `response_format`: Format of the response ("url" or "b64_json") (default: "url")
- `retry`: Number of retry attempts (default: 1)
- `full_response`: Return the full API response (default: false)
- `allow_error`: Return None instead of an error (default: false)

Returns the image URL or a list of image URLs if `n > 1`.

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

# Generate an image
image_url = draw(
    prompt="A futuristic city with flying cars and tall skyscrapers",
    quality="hd",
    size="1024x1024",
)
print(image_url)
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

## License

This package is licensed under the MIT License - see the LICENSE file for details.