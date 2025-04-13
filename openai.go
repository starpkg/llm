// Package llm provides a Starlark module that calls OpenAI models.
package llm

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"image/png"
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

// NewModule creates a new instance of Module with default empty configurations.
func NewModule() *Module {
	return newModuleWithOptions(
		genConfigOption(configKeyProvider, "OpenAI provider type", ProviderOpenAI),
		genConfigOption(configKeyEndpointURL, "OpenAI API endpoint URL", empty),
		genConfigOption(configKeyAPIKey, "OpenAI API key", empty).SetSecret(true),
		genConfigOption(configKeyGPTModel, "GPT model name", empty),
		genConfigOption(configKeyDALLEModel, "DALL-E model name", empty),
		genConfigOption(configKeyAPIVersion, "API version", defaultAPIVersion),
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
	)
}

// genConfigOption creates a configuration option with common settings.
// It sets up the name, description, default value, and environment variable.
func genConfigOption(name, description, defaultValue string) *base.ConfigOption[string] {
	return base.NewConfigOption(defaultValue).
		WithName(name).
		WithDescription(description).
		WithEnvVar(strings.ToUpper(ModuleName + "_" + name))
}

// newModuleWithOptions creates a Module with the given configuration options.
func newModuleWithOptions(providerOpt, endpointOpt, apiKeyOpt, gptModelOpt, dalleModelOpt, apiVersionOpt *base.ConfigOption[string]) *Module {
	cm, _ := base.NewConfigurableModuleWithConfigOptions(
		providerOpt,
		endpointOpt,
		apiKeyOpt,
		gptModelOpt,
		dalleModelOpt,
		apiVersionOpt,
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
			quality        = types.NewNullableStringOrBytes("standard")
			size           = types.NewNullableStringOrBytes("1024x1024")
			style          = types.NewNullableStringOrBytes("vivid")
			responseFormat = types.NewNullableStringOrBytes("url")
			// call
			retryTimes   = 1
			fullResponse = false
			allowError   = false
		)
		if err := starlark.UnpackArgs(b.Name(), args, kwargs,
			"prompt", prompt, "model?", userModel, "n?", &numOfChoices, "quality?", quality, "size?", size, "style?", style, "response_format?", responseFormat,
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

		// build request
		req := oai.ImageRequest{
			Prompt:         prompt.GoString(),
			Model:          model,
			N:              numOfChoices,
			Quality:        quality.GoString(),
			Size:           size.GoString(),
			Style:          style.GoString(),
			ResponseFormat: responseFormat.GoString(),
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
			return convertGoToStarlark(resp)
		}

		// if numOfChoices is 1, return the content string, otherwise return a list of contents
		isURL := strings.ToLower(responseFormat.GoString()) == "url"
		extractImage := func(di oai.ImageResponseDataInner) (starlark.Value, error) {
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

func (m *Module) genChatFunc() starlark.Callable {
	return starlark.NewBuiltin(ModuleName+".chat", func(thread *starlark.Thread, b *starlark.Builtin, args starlark.Tuple, kwargs []starlark.Tuple) (starlark.Value, error) {
		var (
			// message
			msgText       = types.NewNullableStringOrBytesNoDefault()
			msgImageBytes = types.NewNullableStringOrBytesNoDefault()
			msgImageFile  = types.NewNullableStringOrBytesNoDefault()
			msgImageURL   = types.NewNullableStringOrBytesNoDefault()
			messages      = types.NewOneOrManyNoDefault[*starlark.Dict]()
			// model request
			userModel        = types.NewNullableStringOrBytesNoDefault()
			numOfChoices     = 1
			maxTokens        = 64
			temperature      = types.FloatOrInt(1.0)
			topP             = types.FloatOrInt(1.0)
			frequencyPenalty = types.FloatOrInt(0.0)
			presencePenalty  = types.FloatOrInt(0.0)
			stopSequences    = types.NewOneOrManyNoDefault[starlark.String]()
			responseFormat   = types.NewNullableStringOrBytes("text")
			// call
			retryTimes   = 1
			fullResponse = false
			allowError   = false
		)
		if err := starlark.UnpackArgs(b.Name(), args, kwargs,
			"text?", msgText, "image?", msgImageBytes, "image_file?", msgImageFile, "image_url?", msgImageURL, "messages?", messages,
			"model?", userModel, "n?", &numOfChoices, "max_tokens?", &maxTokens, "temperature?", &temperature, "top_p?", &topP, "frequency_penalty?", &frequencyPenalty, "presence_penalty?", &presencePenalty, "stop?", stopSequences, "response_format?", responseFormat,
			"retry?", &retryTimes, "full_response?", &fullResponse, "allow_error?", &allowError,
		); err != nil {
			return none, err
		}

		// get model
		model := m.getModel(configKeyGPTModel, userModel.GoString())
		if model == "" {
			return none, errors.New("gpt model is not set")
		}

		// history messages, prepend user message if defined
		allMsgs := messages.Slice()
		usrMd := starlark.NewDict(1)
		prepared := map[string]*types.NullableStringOrBytes{
			"text":       msgText,
			"image":      msgImageBytes,
			"image_file": msgImageFile,
			"image_url":  msgImageURL,
		}
		for key, val := range prepared {
			if !val.IsNullOrEmpty() {
				usrMd.SetKey(starlark.String(key), val.StarlarkString())
			}
		}
		if usrMd.Len() > 0 {
			usrMd.SetKey(starlark.String("role"), starlark.String(oai.ChatMessageRoleUser))
			allMsgs = append([]*starlark.Dict{usrMd}, allMsgs...)
		}

		// convert to OpenAI chat messages
		chatMessages, err := messagesToChatMessages(allMsgs)
		if err != nil {
			return none, err
		}
		var stopWords []string
		for _, s := range stopSequences.Slice() {
			stopWords = append(stopWords, s.GoString())
		}
		req := oai.ChatCompletionRequest{
			Model:            model,
			Messages:         chatMessages,
			MaxTokens:        maxTokens,
			Temperature:      temperature.GoFloat32(),
			TopP:             topP.GoFloat32(),
			N:                numOfChoices,
			Stop:             stopWords,
			PresencePenalty:  presencePenalty.GoFloat32(),
			FrequencyPenalty: frequencyPenalty.GoFloat32(),
		}
		if rf := responseFormat.GoString(); rf == "json" {
			req.ResponseFormat = &oai.ChatCompletionResponseFormat{
				Type: oai.ChatCompletionResponseFormatTypeJSONObject,
			}
		} else if rf == "text" {
			req.ResponseFormat = &oai.ChatCompletionResponseFormat{
				Type: oai.ChatCompletionResponseFormatTypeText,
			}
		} else {
			return none, fmt.Errorf("unsupported response format: %s", rf)
		}

		// get client
		cli, err := m.getClient(model)
		if err != nil {
			return nil, err
		}

		// send request to provider
		ctx := dataconv.GetThreadContext(thread)
		var resp oai.ChatCompletionResponse
		for i := 0; i < retryTimes; i++ {
			resp, err = cli.CreateChatCompletion(ctx, req)
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
			return convertGoToStarlark(resp)
		}
		if len(resp.Choices) == 0 {
			return none, nil
		}
		// if numOfChoices is 1, return the content string, otherwise return a list of contents
		if numOfChoices == 1 {
			return starlark.String(resp.Choices[0].Message.Content), nil
		}
		var res []starlark.Value
		for _, ch := range resp.Choices {
			res = append(res, starlark.String(ch.Message.Content))
		}
		return starlark.NewList(res), nil
	})
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
	case ProviderOpenAI: // Vanilla OpenAI services
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
func messagesToChatMessages(msgs []*starlark.Dict) ([]oai.ChatCompletionMessage, error) {
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
func convertGoToStarlark(v interface{}) (starlark.Value, error) {
	data, err := json.Marshal(v)
	if err != nil {
		return none, err
	}

	var jsonValue interface{}
	if err := json.Unmarshal(data, &jsonValue); err != nil {
		return none, err
	}

	return convertJSONValueToStarlark(jsonValue)
}

// convertJSONValueToStarlark converts a JSON value to a Starlark value.
func convertJSONValueToStarlark(v interface{}) (starlark.Value, error) {
	switch x := v.(type) {
	case nil:
		return none, nil
	case bool:
		return starlark.Bool(x), nil
	case float64:
		return starlark.Float(x), nil
	case string:
		return starlark.String(x), nil
	case []interface{}:
		elems := make([]starlark.Value, len(x))
		for i, elem := range x {
			var err error
			elems[i], err = convertJSONValueToStarlark(elem)
			if err != nil {
				return none, err
			}
		}
		return starlark.NewList(elems), nil
	case map[string]interface{}:
		dict := starlark.NewDict(len(x))
		for k, v := range x {
			val, err := convertJSONValueToStarlark(v)
			if err != nil {
				return none, err
			}
			if err := dict.SetKey(starlark.String(k), val); err != nil {
				return none, err
			}
		}
		return dict, nil
	default:
		return none, fmt.Errorf("unsupported JSON value type: %T", v)
	}
}
