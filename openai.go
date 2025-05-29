// Package llm provides a Starlark module that calls OpenAI models.
//
// Configuration options:
//   - openai_provider: Provider type (openai, azure, anthropic)
//   - openai_endpoint_url: API endpoint URL
//   - openai_api_key: API key for authentication
//   - openai_gpt_model: Default GPT model name
//   - openai_dalle_model: Default DALL-E model name
//   - api_version: API version (for Azure)
//   - legacy_mode: Use legacy mode for data conversion (default: true)
//
// The chat function supports both blocking and streaming modes:
//   - In blocking mode (default), the function waits for the complete response
//   - In streaming mode (stream=True), responses are received incrementally and can be processed via a callback
//   - Streaming mode can improve responsiveness for large responses or when displaying partial results is desired
//   - To use streaming mode, set stream=True and optionally provide stream_callback=function
//   - The stream_callback receives each chunk of the response as it arrives
//   - In both streaming and blocking modes, the function returns the same format: either the complete content or full response
//   - For streaming, the content is automatically aggregated from all chunks
//
// Token parameters for different models:
//   - max_tokens: Maximum number of tokens to generate (default: 64) - works with most models
//   - max_completion_tokens: Upper bound for generated completion tokens - for o1 series models
//   - For o1, o3, o4 series models, use max_completion_tokens instead of max_tokens
//
// When legacy_mode is true (default), response objects are converted using direct struct
// access (ConvertJSONStruct). When false, JSON conversion is used (GoToStarlarkViaJSON).
package llm

import (
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"image/png"
	"io"
	"mime"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/1set/starlet"
	"github.com/1set/starlet/dataconv"
	"github.com/1set/starlet/dataconv/types"
	oai "github.com/sashabaranov/go-openai"
	"github.com/starpkg/base"
	"go.starlark.net/starlark"
)

// ModuleName defines the expected name for this module when used in Starlark's load() function, e.g., load('llm', 'chat')
const ModuleName = "llm"

// Configuration key constants
const (
	configKeyProvider    = "openai_provider"
	configKeyEndpointURL = "openai_endpoint_url"
	configKeyAPIKey      = "openai_api_key"
	configKeyGPTModel    = "openai_gpt_model"
	configKeyDALLEModel  = "openai_dalle_model"
	configKeyAPIVersion  = "api_version"
	configKeyLegacyMode  = "legacy_mode"
)

// Provider type constants
const (
	// ProviderOpenAI represents the default OpenAI API provider
	ProviderOpenAI = "openai"
	// ProviderAzure represents the Azure OpenAI Service provider
	ProviderAzure = "azure"
	// ProviderAnthropic represents the Anthropic Claude API provider
	ProviderAnthropic = "anthropic"
)

// Default values for API versions
const (
	defaultAPIVersion = "2024-02-01" // Azure's default API version
)

// Module wraps the ConfigurableModule with specific functionality for calling OpenAI models.
type Module struct {
	cfgMod *base.ConfigurableModule
	ext    *base.ConfigurableModuleExt
	cli    *oai.Client
}

// chatParams contains all the parameters required for a chat completion request
type chatParams struct {
	// Message params
	msgText       *types.NullableStringOrBytes
	msgImageBytes *types.NullableStringOrBytes
	msgImageFile  *types.NullableStringOrBytes
	msgImageURL   *types.NullableStringOrBytes
	messages      *types.OneOrMany[*starlark.Dict]

	// Model request params
	userModel           *types.NullableStringOrBytes
	numOfChoices        int
	maxTokens           int
	maxCompletionTokens int
	temperature         types.FloatOrInt
	topP                types.FloatOrInt
	frequencyPenalty    types.FloatOrInt
	presencePenalty     types.FloatOrInt
	stopSequences       *types.OneOrMany[starlark.String]
	responseFormat      *types.NullableStringOrBytes
	reasoningEffort     *types.NullableStringOrBytes

	// Call params
	retryTimes   int
	fullResponse bool
	allowError   bool

	// Stream params
	stream         bool
	streamCallback starlark.Callable
}

// chatResult represents the result of a chat completion request
type chatResult struct {
	id      string
	model   string
	choices []chatChoice
}

// chatChoice represents a single choice in a chat completion response
type chatChoice struct {
	index        int
	content      string
	role         string
	finishReason string
}

// NewModule creates a new instance of Module with default empty configurations.
func NewModule() *Module {
	return newModuleWithOptions(
		genConfigOption(configKeyProvider, "OpenAI provider type", ProviderOpenAI),
		genConfigOption(configKeyEndpointURL, "OpenAI API endpoint URL", empty),
		genConfigOption(configKeyAPIKey, "OpenAI API key", empty).SetSecret(true),
		genConfigOption(configKeyGPTModel, "GPT model name", empty),
		genConfigOption(configKeyDALLEModel, "DALL-E model name", empty),
		genConfigOption(configKeyAPIVersion, "API version", defaultAPIVersion),
		genConfigOption(configKeyLegacyMode, "Use legacy mode for data conversion", true),
	)
}

// NewModuleWithConfig creates a new instance of Module with the given configuration values.
func NewModuleWithConfig(serviceProvider, endpointURL, apiKey, gptModel, dalleModel, apiVersion string) *Module {
	// If apiVersion is empty, use the default
	if apiVersion == "" {
		apiVersion = defaultAPIVersion
	}

	return newModuleWithOptions(
		genConfigOption(configKeyProvider, "OpenAI provider with preset value", serviceProvider),
		genConfigOption(configKeyEndpointURL, "OpenAI API endpoint URL with preset value", endpointURL),
		genConfigOption(configKeyAPIKey, "OpenAI API key with preset value", apiKey).SetSecret(true),
		genConfigOption(configKeyGPTModel, "GPT model name with preset value", gptModel),
		genConfigOption(configKeyDALLEModel, "DALL-E model name with preset value", dalleModel),
		genConfigOption(configKeyAPIVersion, "API version with preset value", apiVersion),
		genConfigOption(configKeyLegacyMode, "Use legacy mode for data conversion", true),
	)
}

// genConfigOption creates a configuration option with common settings.
// It sets up the name, description, default value, and environment variable.
func genConfigOption[T any](name, description string, defaultValue T) *base.ConfigOption[T] {
	return base.NewConfigOption(defaultValue).
		WithName(name).
		WithDescription(description).
		WithEnvVar(strings.ToUpper(ModuleName + "_" + name))
}

// newModuleWithOptions creates a Module with the given configuration options.
func newModuleWithOptions(providerOpt, endpointOpt, apiKeyOpt, gptModelOpt, dalleModelOpt, apiVersionOpt *base.ConfigOption[string], legacyModeOpt *base.ConfigOption[bool]) *Module {
	cm, _ := base.NewConfigurableModuleWithConfigOptions(
		providerOpt,
		endpointOpt,
		apiKeyOpt,
		gptModelOpt,
		dalleModelOpt,
		apiVersionOpt,
		legacyModeOpt,
	)
	return &Module{
		cfgMod: cm,
		ext:    cm.Extend(),
	}
}

// LoadModule returns the Starlark module loader with the email-specific functions.
func (m *Module) LoadModule() starlet.ModuleLoader {
	additionalFuncs := starlark.StringDict{
		"message": starlark.NewBuiltin("message", newMessageStruct),
		"chat":    m.genChatFunc(),
		"draw":    m.genDrawFunc(),
	}
	return m.cfgMod.LoadModule(ModuleName, additionalFuncs)
}

var (
	none  = starlark.None // none is a convenience variable for starlark.None
	empty string          // empty is a convenience variable for an empty string
)

func newMessageStruct(thread *starlark.Thread, b *starlark.Builtin, args starlark.Tuple, kwargs []starlark.Tuple) (starlark.Value, error) {
	// Parse arguments
	var (
		role          = types.NewNullableStringOrBytes(oai.ChatMessageRoleUser)
		msgText       = types.NewNullableStringOrBytesNoDefault()
		msgImageBytes = types.NewNullableStringOrBytesNoDefault()
		msgImageFile  = types.NewNullableStringOrBytesNoDefault()
		msgImageURL   = types.NewNullableStringOrBytesNoDefault()
	)
	if err := starlark.UnpackArgs(b.Name(), args, kwargs, "role?", role,
		"text?", msgText, "image?", msgImageBytes, "image_file?", msgImageFile, "image_url?", msgImageURL,
	); err != nil {
		return none, err
	}

	// Create a new message
	md := starlark.NewDict(2)

	// Add key values
	prepared := map[string]*types.NullableStringOrBytes{
		"role":       role,
		"text":       msgText,
		"image":      msgImageBytes,
		"image_file": msgImageFile,
		"image_url":  msgImageURL,
	}
	for key, val := range prepared {
		if !val.IsNullOrEmpty() {
			md.SetKey(starlark.String(key), val.StarlarkString())
		}
	}

	return md, nil
}

func (m *Module) genDrawFunc() starlark.Callable {
	return starlark.NewBuiltin(ModuleName+".draw", func(thread *starlark.Thread, b *starlark.Builtin, args starlark.Tuple, kwargs []starlark.Tuple) (starlark.Value, error) {
		var (
			prompt = types.NewNullableStringOrBytesNoDefault()
			// model request
			userModel      = types.NewNullableStringOrBytesNoDefault()
			numOfChoices   = 1
			quality        = types.NewNullableStringOrBytesNoDefault()
			size           = types.NewNullableStringOrBytesNoDefault()
			style          = types.NewNullableStringOrBytes("vivid")
			responseFormat = types.NewNullableStringOrBytes("url")
			// GPT Image 1 specific parameters
			background        = types.NewNullableStringOrBytesNoDefault()
			moderation        = types.NewNullableStringOrBytesNoDefault()
			outputFormat      = types.NewNullableStringOrBytesNoDefault()
			outputCompression = 0
			// call
			retryTimes   = 1
			fullResponse = false
			allowError   = false
		)
		if err := starlark.UnpackArgs(b.Name(), args, kwargs,
			"prompt", prompt, "model?", userModel, "n?", &numOfChoices, "quality?", quality, "size?", size, "style?", style, "response_format?", responseFormat,
			"background?", background, "moderation?", moderation, "output_format?", outputFormat, "output_compression?", &outputCompression,
			"retry?", &retryTimes, "full_response?", &fullResponse, "allow_error?", &allowError,
		); err != nil {
			return none, err
		}

		// get prompt
		if prompt.IsNullOrEmpty() {
			return none, errors.New("prompt is required")
		}

		// get model
		model := m.getModel(configKeyDALLEModel, userModel.GoString())
		if model == "" {
			return none, errors.New("dalle model is not set")
		}

		// validate model-specific parameters and set defaults
		isGPTImage1 := strings.ToLower(model) == "gpt-image-1"
		isDallE3 := strings.ToLower(model) == "dall-e-3"
		isDallE2 := strings.ToLower(model) == "dall-e-2"

		// Set default values based on model
		if isGPTImage1 {
			// GPT Image 1 defaults
			if quality.IsNullOrEmpty() {
				quality = types.NewNullableStringOrBytes("auto")
			}
			if size.IsNullOrEmpty() {
				size = types.NewNullableStringOrBytes("auto")
			}
			if background.IsNullOrEmpty() {
				background = types.NewNullableStringOrBytes("auto")
			}
			if moderation.IsNullOrEmpty() {
				moderation = types.NewNullableStringOrBytes("auto")
			}
			if outputFormat.IsNullOrEmpty() {
				outputFormat = types.NewNullableStringOrBytes("png")
			}
			if outputCompression == 0 {
				outputCompression = 100
			}
		} else {
			// DALL-E defaults
			if quality.IsNullOrEmpty() {
				quality = types.NewNullableStringOrBytes("standard")
			}
			if size.IsNullOrEmpty() {
				size = types.NewNullableStringOrBytes("1024x1024")
			}
		}

		// Validate DALL-E 3 specific constraints
		if isDallE3 && numOfChoices > 1 {
			return none, errors.New("dall-e-3 only supports n=1")
		}

		// Validate GPT Image 1 specific parameters
		if isGPTImage1 {
			// Validate background
			if !background.IsNullOrEmpty() {
				bg := strings.ToLower(background.GoString())
				if bg != "auto" && bg != "transparent" && bg != "opaque" {
					return none, errors.New("background must be 'auto', 'transparent', or 'opaque' for gpt-image-1")
				}
			}

			// Validate moderation
			if !moderation.IsNullOrEmpty() {
				mod := strings.ToLower(moderation.GoString())
				if mod != "auto" && mod != "low" {
					return none, errors.New("moderation must be 'auto' or 'low' for gpt-image-1")
				}
			}

			// Validate output format
			if !outputFormat.IsNullOrEmpty() {
				of := strings.ToLower(outputFormat.GoString())
				if of != "png" && of != "jpeg" && of != "webp" {
					return none, errors.New("output_format must be 'png', 'jpeg', or 'webp' for gpt-image-1")
				}
			}

			// Validate output compression
			if outputCompression < 0 || outputCompression > 100 {
				return none, errors.New("output_compression must be between 0 and 100 for gpt-image-1")
			}

			// Validate quality options for GPT Image 1
			if !quality.IsNullOrEmpty() {
				qual := strings.ToLower(quality.GoString())
				if qual != "auto" && qual != "high" && qual != "medium" && qual != "low" {
					return none, errors.New("quality must be 'auto', 'high', 'medium', or 'low' for gpt-image-1")
				}
			}

			// Validate size options for GPT Image 1
			if !size.IsNullOrEmpty() {
				sz := strings.ToLower(size.GoString())
				if sz != "auto" && sz != "1024x1024" && sz != "1536x1024" && sz != "1024x1536" {
					return none, errors.New("size must be 'auto', '1024x1024', '1536x1024', or '1024x1536' for gpt-image-1")
				}
			}
		} else {
			// Warn about GPT Image 1 specific parameters being used with DALL-E
			if !background.IsNullOrEmpty() || !moderation.IsNullOrEmpty() || !outputFormat.IsNullOrEmpty() || outputCompression != 100 {
				// For DALL-E models, ignore these parameters silently to maintain compatibility
			}

			// Validate DALL-E quality options
			if !quality.IsNullOrEmpty() {
				qual := strings.ToLower(quality.GoString())
				if isDallE3 && qual != "standard" && qual != "hd" {
					return none, errors.New("quality must be 'standard' or 'hd' for dall-e-3")
				}
				if isDallE2 && qual != "standard" {
					return none, errors.New("quality must be 'standard' for dall-e-2")
				}
			}
		}

		// build request
		req := oai.ImageRequest{
			Prompt:  prompt.GoString(),
			Model:   model,
			N:       numOfChoices,
			Quality: quality.GoString(),
			Size:    size.GoString(),
		}

		// Add DALL-E 3 specific parameters
		if isDallE3 {
			req.Style = style.GoString()
			req.ResponseFormat = responseFormat.GoString()
		}

		// Add GPT Image 1 specific parameters
		if isGPTImage1 {
			if !background.IsNullOrEmpty() {
				req.Background = background.GoString()
			}
			if !moderation.IsNullOrEmpty() {
				req.Moderation = moderation.GoString()
			}
			if !outputFormat.IsNullOrEmpty() {
				req.OutputFormat = outputFormat.GoString()
			}
			if outputCompression > 0 && outputCompression != 100 {
				req.OutputCompression = outputCompression
			}
		}

		// get client
		cli, err := m.getClient(model)
		if err != nil {
			return nil, err
		}

		// send request to provider
		ctx := dataconv.GetThreadContext(thread)
		var resp oai.ImageResponse
		for i := 0; i < retryTimes; i++ {
			resp, err = cli.CreateImage(ctx, req)
			// if no error, break the loop, got the response
			if err == nil {
				break
			}
			// if the error is a bad request, break the loop, no need to retry
			var ae *oai.APIError
			if errors.As(err, &ae) && ae != nil {
				if ae.HTTPStatusCode == http.StatusBadRequest {
					break
				}
			}
		}

		// handle error: if allowError is set, return None, otherwise return the error
		if err != nil {
			if allowError {
				return none, nil
			}
			return none, err
		}

		// return the response: if fullResponse is set, return the full response, otherwise return the content
		if fullResponse {
			return m.convertGoToStarlark(&resp)
		}

		// For GPT Image 1, always return base64 data since it doesn't support URL format
		// For DALL-E, check the response format
		extractImage := func(di oai.ImageResponseDataInner) (starlark.Value, error) {
			if isGPTImage1 {
				// GPT Image 1 always returns base64 data, decode it to bytes
				if di.B64JSON != "" {
					ib, err := base64.StdEncoding.DecodeString(di.B64JSON)
					if err != nil {
						return none, fmt.Errorf("failed to decode base64 image data: %w", err)
					}
					return starlark.Bytes(string(ib)), nil
				}
				return none, errors.New("no image data returned from gpt-image-1")
			}

			// DALL-E logic (existing)
			isURL := strings.ToLower(responseFormat.GoString()) == "url"
			if isURL {
				return starlark.String(di.URL), nil
			}
			ib, err := base64.StdEncoding.DecodeString(di.B64JSON)
			if err != nil {
				return none, err
			}
			r := bytes.NewReader(ib)
			img, err := png.Decode(r)
			if err != nil {
				return none, err
			}
			bf := new(bytes.Buffer)
			if err := png.Encode(bf, img); err != nil {
				return none, err
			}
			return starlark.Bytes(bf.String()), nil
		}

		if numOfChoices == 1 {
			return extractImage(resp.Data[0])
		}
		var res []starlark.Value
		for _, di := range resp.Data {
			img, err := extractImage(di)
			if err != nil {
				return none, err
			}
			res = append(res, img)
		}
		return starlark.NewList(res), nil
	})
}

// genChatFunc returns a Starlark callable for interacting with OpenAI's chat completion API
func (m *Module) genChatFunc() starlark.Callable {
	return starlark.NewBuiltin(ModuleName+".chat", func(thread *starlark.Thread, b *starlark.Builtin, args starlark.Tuple, kwargs []starlark.Tuple) (starlark.Value, error) {
		// Parse and validate parameters
		params, err := m.parseChatParams(b, args, kwargs)
		if err != nil {
			return none, err
		}

		// Get model and validate it
		model := m.getModel(configKeyGPTModel, params.userModel.GoString())
		if model == "" {
			return none, errors.New("gpt model is not set")
		}

		// Prepare messages and chat completion request
		allMsgs := m.prepareMessages(params)
		req, err := m.prepareChatRequest(allMsgs, model, params)
		if err != nil {
			return none, err
		}

		// Get client
		cli, err := m.getClient(model)
		if err != nil {
			return nil, err
		}

		// Context from Starlark thread
		ctx := dataconv.GetThreadContext(thread)

		// Handle request based on streaming mode
		if params.stream {
			return m.handleStreamingRequest(ctx, cli, req, model, thread, params)
		}

		return m.handleBlockingRequest(ctx, cli, req, params)
	})
}

// handleBlockingRequest processes a blocking (non-streaming) chat completion request
func (m *Module) handleBlockingRequest(ctx context.Context, cli *oai.Client, req oai.ChatCompletionRequest, params *chatParams) (starlark.Value, error) {
	var resp oai.ChatCompletionResponse
	var err error

	// Try the request with retries
	for i := 0; i < params.retryTimes; i++ {
		resp, err = cli.CreateChatCompletion(ctx, req)

		// If successful, break the loop
		if err == nil {
			break
		}

		// Check if this is a bad request error (no need to retry)
		var ae *oai.APIError
		if errors.As(err, &ae) && ae != nil && ae.HTTPStatusCode == http.StatusBadRequest {
			break
		}
	}

	// Handle error
	if err != nil {
		if params.allowError {
			return none, nil
		}
		return none, err
	}

	return m.formatChatResponse(&resp, params)
}

// parseChatParams parses and validates the parameters for a chat completion request
func (m *Module) parseChatParams(b *starlark.Builtin, args starlark.Tuple, kwargs []starlark.Tuple) (*chatParams, error) {
	p := &chatParams{
		// Default values
		msgText:             types.NewNullableStringOrBytesNoDefault(),
		msgImageBytes:       types.NewNullableStringOrBytesNoDefault(),
		msgImageFile:        types.NewNullableStringOrBytesNoDefault(),
		msgImageURL:         types.NewNullableStringOrBytesNoDefault(),
		messages:            types.NewOneOrManyNoDefault[*starlark.Dict](),
		userModel:           types.NewNullableStringOrBytesNoDefault(),
		numOfChoices:        1,
		maxTokens:           0, // Default to 0 (not set)
		maxCompletionTokens: 0, // Default to 0 (not set)
		temperature:         types.FloatOrInt(1.0),
		topP:                types.FloatOrInt(1.0),
		frequencyPenalty:    types.FloatOrInt(0.0),
		presencePenalty:     types.FloatOrInt(0.0),
		stopSequences:       types.NewOneOrManyNoDefault[starlark.String](),
		responseFormat:      types.NewNullableStringOrBytes("text"),
		reasoningEffort:     types.NewNullableStringOrBytesNoDefault(),
		retryTimes:          1,
		fullResponse:        false,
		allowError:          false,
		stream:              false,
	}

	if err := starlark.UnpackArgs(b.Name(), args, kwargs,
		"text?", p.msgText, "image?", p.msgImageBytes, "image_file?", p.msgImageFile, "image_url?", p.msgImageURL, "messages?", p.messages,
		"model?", p.userModel, "n?", &p.numOfChoices, "max_tokens?", &p.maxTokens, "max_completion_tokens?", &p.maxCompletionTokens, "temperature?", &p.temperature, "top_p?", &p.topP, "frequency_penalty?", &p.frequencyPenalty, "presence_penalty?", &p.presencePenalty, "stop?", p.stopSequences, "response_format?", p.responseFormat, "reasoning_effort?", p.reasoningEffort,
		"retry?", &p.retryTimes, "full_response?", &p.fullResponse, "allow_error?", &p.allowError,
		"stream?", &p.stream, "stream_callback?", &p.streamCallback,
	); err != nil {
		return nil, err
	}

	return p, nil
}

// prepareMessages constructs a list of messages for the chat completion API
func (m *Module) prepareMessages(params *chatParams) []*starlark.Dict {
	allMsgs := params.messages.Slice()

	// Create user message from parameters if provided
	usrMd := starlark.NewDict(1)
	prepared := map[string]*types.NullableStringOrBytes{
		"text":       params.msgText,
		"image":      params.msgImageBytes,
		"image_file": params.msgImageFile,
		"image_url":  params.msgImageURL,
	}

	for key, val := range prepared {
		if !val.IsNullOrEmpty() {
			usrMd.SetKey(starlark.String(key), val.StarlarkString())
		}
	}

	// Add the user message to the beginning of the list if it has content
	if usrMd.Len() > 0 {
		usrMd.SetKey(starlark.String("role"), starlark.String(oai.ChatMessageRoleUser))
		allMsgs = append([]*starlark.Dict{usrMd}, allMsgs...)
	}

	return allMsgs
}

// prepareChatRequest builds a chat completion request from messages and parameters
func (m *Module) prepareChatRequest(allMsgs []*starlark.Dict, model string, params *chatParams) (oai.ChatCompletionRequest, error) {
	// Convert Starlark messages to OpenAI chat messages
	chatMessages, err := m.messagesToChatMessages(allMsgs)
	if err != nil {
		return oai.ChatCompletionRequest{}, err
	}

	// Convert stop sequences
	var stopWords []string
	for _, s := range params.stopSequences.Slice() {
		stopWords = append(stopWords, s.GoString())
	}

	// Build request
	req := oai.ChatCompletionRequest{
		Model:            model,
		Messages:         chatMessages,
		MaxTokens:        params.maxTokens,
		Temperature:      params.temperature.GoFloat32(),
		TopP:             params.topP.GoFloat32(),
		N:                params.numOfChoices,
		Stop:             stopWords,
		PresencePenalty:  params.presencePenalty.GoFloat32(),
		FrequencyPenalty: params.frequencyPenalty.GoFloat32(),
		Stream:           params.stream,
	}

	// Set ReasoningEffort if provided
	if !params.reasoningEffort.IsNullOrEmpty() {
		req.ReasoningEffort = params.reasoningEffort.GoString()
	}

	// Set StreamOptions with IncludeUsage for streaming requests
	if params.stream {
		req.StreamOptions = &oai.StreamOptions{
			IncludeUsage: true,
		}
	}

	// Set MaxCompletionTokens if provided (for o1 series models)
	if params.maxCompletionTokens > 0 {
		req.MaxCompletionTokens = params.maxCompletionTokens
	}

	// Set response format
	if rf := params.responseFormat.GoString(); rf == "json" {
		req.ResponseFormat = &oai.ChatCompletionResponseFormat{
			Type: oai.ChatCompletionResponseFormatTypeJSONObject,
		}
	} else if rf == "text" {
		req.ResponseFormat = &oai.ChatCompletionResponseFormat{
			Type: oai.ChatCompletionResponseFormatTypeText,
		}
	} else {
		return oai.ChatCompletionRequest{}, fmt.Errorf("unsupported response format: %s", rf)
	}

	return req, nil
}

// handleStreamingRequest processes a streaming chat completion request
func (m *Module) handleStreamingRequest(ctx context.Context, cli *oai.Client, req oai.ChatCompletionRequest, model string, thread *starlark.Thread, params *chatParams) (starlark.Value, error) {
	var streamErr error
	var fullResp *oai.ChatCompletionResponse

	// Try the request with retries
	for i := 0; i < params.retryTimes; i++ {
		fullResp, streamErr = m.processStream(ctx, cli, req, model, thread, params)

		// If successful, return the result
		if streamErr == nil {
			return m.formatChatResponse(fullResp, params)
		}

		// Check if this is a bad request error (no need to retry)
		var ae *oai.APIError
		if errors.As(streamErr, &ae) && ae != nil && ae.HTTPStatusCode == http.StatusBadRequest {
			break
		}
	}

	// Handle error
	if streamErr != nil {
		if params.allowError {
			return none, nil
		}
		return none, streamErr
	}

	return none, nil
}

// processStream handles a single streaming request attempt
func (m *Module) processStream(ctx context.Context, cli *oai.Client, req oai.ChatCompletionRequest, model string, thread *starlark.Thread, params *chatParams) (*oai.ChatCompletionResponse, error) {
	// Create a stream for chat completion
	stream, err := cli.CreateChatCompletionStream(ctx, req)
	if err != nil {
		return nil, err
	}
	defer stream.Close()

	// Initialize the full response
	fullResp := &oai.ChatCompletionResponse{
		Model:   model,
		Choices: make([]oai.ChatCompletionChoice, params.numOfChoices),
	}

	// Initialize content builders for each choice
	contentBuilders := make([]strings.Builder, params.numOfChoices)

	// Track metadata from stream responses
	var lastStreamResp oai.ChatCompletionStreamResponse

	// Initialize token usage accumulator
	accumulatedUsage := oai.Usage{}
	usageFound := false

	// Process the stream
	for {
		// Receive the next response
		streamResp, streamErr := stream.Recv()
		if streamErr != nil {
			if streamErr == io.EOF {
				// End of stream is not an error
				break
			}
			return nil, streamErr
		}

		// Keep track of the last response for metadata
		lastStreamResp = streamResp

		// Store the ID from the first response
		if fullResp.ID == "" && streamResp.ID != "" {
			fullResp.ID = streamResp.ID
		}

		// Accumulate token usage from this chunk if available
		if streamResp.Usage != nil {
			usageFound = true
			accumulatedUsage.PromptTokens += streamResp.Usage.PromptTokens
			accumulatedUsage.CompletionTokens += streamResp.Usage.CompletionTokens
			accumulatedUsage.TotalTokens += streamResp.Usage.TotalTokens
		}

		// Process each choice in the stream response
		for i, choice := range streamResp.Choices {
			if i < len(contentBuilders) {
				// Append the delta content to the builder
				contentBuilders[i].WriteString(choice.Delta.Content)

				// Initialize the choice in the full response if not done yet
				if fullResp.Choices[i].Message.Role == "" {
					fullResp.Choices[i].Message.Role = choice.Delta.Role
					if choice.Delta.Role == "" {
						fullResp.Choices[i].Message.Role = oai.ChatMessageRoleAssistant
					}
				}

				// Set index and finish reason if provided
				fullResp.Choices[i].Index = choice.Index
				if choice.FinishReason != "" {
					fullResp.Choices[i].FinishReason = choice.FinishReason
				}
			}
		}

		// Call the stream callback if provided
		if params.streamCallback != nil {
			if err := m.callStreamCallback(thread, params.streamCallback, &streamResp); err != nil {
				return nil, err
			}
		}
	}

	// Combine the content from each chunk
	for i := range fullResp.Choices {
		if i < len(contentBuilders) {
			fullResp.Choices[i].Message.Content = contentBuilders[i].String()
		}
	}

	// Use accumulated token usage if we found any
	if usageFound {
		fullResp.Usage = accumulatedUsage
	} else if lastStreamResp.Usage != nil {
		// Fallback to the last response if we didn't accumulate anything
		fullResp.Usage = *lastStreamResp.Usage
	}

	// Copy any available metadata from the last response
	if lastStreamResp.Created > 0 {
		fullResp.Created = lastStreamResp.Created
	}
	if lastStreamResp.Model != "" {
		fullResp.Model = lastStreamResp.Model
	}
	if lastStreamResp.SystemFingerprint != "" {
		fullResp.SystemFingerprint = lastStreamResp.SystemFingerprint
	}

	return fullResp, nil
}

// callStreamCallback invokes the provided callback with a stream response
func (m *Module) callStreamCallback(thread *starlark.Thread, callback starlark.Callable, resp *oai.ChatCompletionStreamResponse) error {
	// Convert the stream response to Starlark
	starlarkResp, err := m.convertGoToStarlark(resp)
	if err != nil {
		return fmt.Errorf("failed to convert stream response to Starlark: %w", err)
	}

	// Call the callback with the response
	if _, err := starlark.Call(thread, callback, starlark.Tuple{starlarkResp}, nil); err != nil {
		return fmt.Errorf("stream callback error: %w", err)
	}

	return nil
}

// formatChatResponse formats the chat completion response according to parameters
func (m *Module) formatChatResponse(resp *oai.ChatCompletionResponse, params *chatParams) (starlark.Value, error) {
	// Return the full response if requested
	if params.fullResponse {
		return m.convertGoToStarlark(resp)
	}

	// Check if we have choices
	if len(resp.Choices) == 0 {
		return none, nil
	}

	// For a single choice, return the content string
	if params.numOfChoices == 1 {
		return starlark.String(resp.Choices[0].Message.Content), nil
	}

	// For multiple choices, return a list of contents
	var res []starlark.Value
	for _, ch := range resp.Choices {
		res = append(res, starlark.String(ch.Message.Content))
	}
	return starlark.NewList(res), nil
}

// SetClient sets the OpenAI client for this module.
func (m *Module) SetClient(cli *oai.Client) {
	m.cli = cli
}

// getClient retrieves the OpenAI client for this module.
func (m *Module) getClient(model string) (*oai.Client, error) {
	if m.cli != nil {
		// use the existing client
		return m.cli, nil
	}

	provider := m.ext.GetString(configKeyProvider, ProviderOpenAI)
	apiKey := m.ext.GetString(configKeyAPIKey)
	if apiKey == "" {
		return nil, fmt.Errorf("%s is not set", configKeyAPIKey)
	}

	endpointURL := m.ext.GetString(configKeyEndpointURL, "")
	apiVersion := m.ext.GetString(configKeyAPIVersion, defaultAPIVersion)

	// create client configuration
	var cfg oai.ClientConfig
	switch strings.ToLower(provider) {
	case ProviderAzure: // Azure OpenAI services
		if endpointURL == "" {
			return nil, fmt.Errorf("%s is required for Azure provider", configKeyEndpointURL) // endpointURL is required for Azure
		}
		cfg = oai.DefaultAzureConfig(apiKey, endpointURL)
		cfg.APIVersion = apiVersion
		cfg.AzureModelMapperFunc = func(_ string) string {
			return model
		}
	case ProviderAnthropic: // Anthropic Claude API
		cfg = oai.DefaultConfig(apiKey)
		cfg.APIVersion = apiVersion
		if endpointURL != "" {
			cfg.BaseURL = endpointURL
		} else {
			cfg.BaseURL = "https://api.anthropic.com"
		}
	case ProviderOpenAI, empty: // Vanilla OpenAI services
		cfg = oai.DefaultConfig(apiKey)
		if endpointURL != "" {
			cfg.BaseURL = endpointURL
		}
	default:
		return nil, fmt.Errorf("unsupported provider: %s", provider)
	}

	// create a new client
	return oai.NewClientWithConfig(cfg), nil
}

// getModel retrieves the model name.
// If modelVal is empty, it will use the modelKey to retrieve the model value from the configuration.
func (m *Module) getModel(key, val string) string {
	// use the provided model value
	if val != "" {
		return val
	}
	// or retrieve the model value from the configuration
	return m.ext.GetString(key, "")
}

// getStringFromDict retrieves a string value from a dictionary and whether the key exists
func getStringFromDict(d *starlark.Dict, key string) (string, bool) {
	v, ok, err := d.Get(starlark.String(key))
	// if the key is not found, or the value is nil, or there is an error, return an empty string
	if err != nil || !ok || v == nil {
		return empty, false
	}
	// if the value is a string, return the string
	if s, ok := v.(starlark.String); ok {
		return string(s), true
	} else if b, ok := v.(starlark.Bytes); ok {
		return string(b), true
	}
	// otherwise, return an empty string
	return empty, false
}

// imageFileToBase64 reads file and convert it to base64 data.
func imageFileToBase64(filePath string) (string, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return "", err
	}
	defer file.Close()

	fileInfo, err := file.Stat()
	if err != nil {
		return "", err
	}

	fileSize := fileInfo.Size()
	fileBuffer := make([]byte, fileSize)

	bytesRead, err := file.Read(fileBuffer)
	if err != nil {
		return "", err
	}
	if bytesRead != int(fileSize) {
		return "", fmt.Errorf("expected to read %d bytes but read %d", fileSize, bytesRead)
	}

	base64Data := base64.StdEncoding.EncodeToString(fileBuffer)
	mimeType := mime.TypeByExtension(filepath.Ext(filePath))
	return fmt.Sprintf("data:%s;base64,%s", mimeType, base64Data), nil
}

// imageDataToBase64 converts image data to base64 data.
func imageDataToBase64(data []byte) string {
	base64Data := base64.StdEncoding.EncodeToString(data)
	mimeType := http.DetectContentType(data)
	return fmt.Sprintf("data:%s;base64,%s", mimeType, base64Data)
}

// messagesToChatMessages converts a list of messages in starlark Dictionary to a list of OpenAI chat messages.
func (m *Module) messagesToChatMessages(msgs []*starlark.Dict) ([]oai.ChatCompletionMessage, error) {
	var res []oai.ChatCompletionMessage
	for i, md := range msgs {
		msg := oai.ChatCompletionMessage{}
		role, ok := getStringFromDict(md, "role")
		if !ok {
			return nil, fmt.Errorf("message %d: role is required", i+1)
		}
		msg.Role = role

		// get the content
		text, okT := getStringFromDict(md, "text")
		imageBytes, okI := getStringFromDict(md, "image")
		imageFile, okF := getStringFromDict(md, "image_file")
		imageURL, okU := getStringFromDict(md, "image_url")
		okImg := okI || okF || okU

		// if all are empty, return an error
		if !(okT || okImg) {
			return nil, fmt.Errorf("message %d: at least one of text, image, image_file, or image_url is required", i+1)
		}

		// check if text and image are both set
		if okT && !okImg {
			// only text is set, treat as one text message
			msg.Content = text
			res = append(res, msg)
			continue
		}

		// build the message parts
		var mcp []oai.ChatMessagePart
		if okT { // for text part
			mcp = append(mcp, oai.ChatMessagePart{
				Type: oai.ChatMessagePartTypeText,
				Text: text,
			})
		}
		if okU { // for image URL part
			mcp = append(mcp, oai.ChatMessagePart{
				Type: oai.ChatMessagePartTypeImageURL,
				ImageURL: &oai.ChatMessageImageURL{
					URL:    imageURL,
					Detail: oai.ImageURLDetailAuto,
				},
			})
		}
		if okI { // for image content part, convert to mime & base64
			b64 := imageDataToBase64([]byte(imageBytes))
			mcp = append(mcp, oai.ChatMessagePart{
				Type: oai.ChatMessagePartTypeImageURL,
				ImageURL: &oai.ChatMessageImageURL{
					URL:    b64,
					Detail: oai.ImageURLDetailAuto,
				},
			})
		}
		if okF { // for image file part, read and convert to mime & base64
			b64, err := imageFileToBase64(imageFile)
			if err != nil {
				return nil, fmt.Errorf("message %d: %w", i+1, err)
			}
			mcp = append(mcp, oai.ChatMessagePart{
				Type: oai.ChatMessagePartTypeImageURL,
				ImageURL: &oai.ChatMessageImageURL{
					URL:    b64,
					Detail: oai.ImageURLDetailAuto,
				},
			})
		}

		// set the message parts back to the message
		msg.MultiContent = mcp
		res = append(res, msg)
	}
	return res, nil
}

// convertGoToStarlark converts a Go struct to a Starlark value using JSON marshaling.
// When legacy mode is enabled (default), it uses ConvertJSONStruct which provides
// direct struct access. When disabled, it uses GoToStarlarkViaJSON which creates
// a JSON representation.
func (m *Module) convertGoToStarlark(v interface{}) (starlark.Value, error) {
	// Check legacy mode setting
	if m.ext.GetBool(configKeyLegacyMode, true) {
		// Legacy mode: direct struct access
		return dataconv.ConvertJSONStruct(v), nil
	}

	// Modern mode: JSON-based conversion
	return dataconv.GoToStarlarkViaJSON(v)
}
