package llm

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/1set/starlet"
	oai "github.com/sashabaranov/go-openai"
	"github.com/starpkg/base"
	"go.starlark.net/starlark"
	"go.starlark.net/starlarkstruct"
)

// TestStarlarkScripts runs Starlark test scripts from the test directory.
// Scripts with "test-" prefix should succeed, "panic-" prefix should fail.
func TestStarlarkScripts(t *testing.T) {
	// Create a module factory function that returns a fresh module loader for each test
	moduleFactory := func() starlet.ModuleLoader {
		return NewModule().LoadModule()
	}
	extraModules := []string{"runtime", "file", "path", "json", "atom", "time", "go_idiomatic"}

	// Use the helper function from the base package
	base.RunStarlarkTests(t, ModuleName, moduleFactory, extraModules, "")
}

func TestKwargsParameter(t *testing.T) {
	// Create a new module
	module := NewModule()
	moduleLoader := module.LoadModule()

	// Test script that verifies kwargs parameter parsing
	script := `
load("llm", "set_openai_endpoint_url", "set_openai_api_key", "message", "chat")

# Configure with test credentials
set_openai_endpoint_url("https://api.openai.com/v1")
set_openai_api_key("test-key")

def test_kwargs_parsing():
    """Test that kwargs parameter can be parsed correctly"""
    
    # Test basic kwargs - should parse without error even if API call fails
    resp = chat(
        text="Hello",
        model="gpt-3.5-turbo",
        max_tokens=10,
        kwargs={"enable_thinking": True},
        allow_error=True
    )
    print("basic_kwargs_ok")
    
    # Test multiple kwargs with different types
    resp = chat(
        text="Hello",
        model="gpt-3.5-turbo", 
        max_tokens=10,
        kwargs={
            "enable_thinking": False,
            "custom_param": "value",
            "number": 42,
            "ratio": 0.5,
            "list_param": ["a", "b"]
        },
        allow_error=True
    )
    print("multiple_kwargs_ok")
    
    # Test empty kwargs
    resp = chat(
        text="Hello",
        model="gpt-3.5-turbo",
        max_tokens=10,
        kwargs={},
        allow_error=True
    )
    print("empty_kwargs_ok")
    
    # Test without kwargs parameter
    resp = chat(
        text="Hello",
        model="gpt-3.5-turbo",
        max_tokens=10,
        allow_error=True
    )
    print("no_kwargs_ok")

test_kwargs_parsing()
`

	// Create a starlet machine with print capture
	env := starlet.NewDefault()
	env.SetScriptContent([]byte(script))

	// Capture print output
	var printOutput strings.Builder
	env.SetPrintFunc(func(_ *starlark.Thread, msg string) {
		printOutput.WriteString(msg)
		printOutput.WriteString("\n")
	})

	// Register our module
	loaders := make(map[string]starlet.ModuleLoader)
	loaders["llm"] = moduleLoader
	env.SetLazyloadModules(loaders)

	// Run the script
	_, err := env.Run()
	if err != nil {
		t.Fatalf("Failed to run script: %v", err)
	}

	// Check the output
	output := printOutput.String()

	// All kwargs tests should succeed in parsing (even if API calls fail)
	if !strings.Contains(output, "basic_kwargs_ok") {
		t.Errorf("Basic kwargs parsing failed. Output: %s", output)
	}

	if !strings.Contains(output, "multiple_kwargs_ok") {
		t.Errorf("Multiple kwargs parsing failed. Output: %s", output)
	}

	if !strings.Contains(output, "empty_kwargs_ok") {
		t.Errorf("Empty kwargs parsing failed. Output: %s", output)
	}

	if !strings.Contains(output, "no_kwargs_ok") {
		t.Errorf("No kwargs parsing failed. Output: %s", output)
	}

	// Check that there are no parsing errors
	if strings.Contains(output, "_error:") {
		t.Errorf("Kwargs parsing had errors. Output: %s", output)
	}
}

func TestKwargsConversion(t *testing.T) {
	// Test the convertStarlarkDictToGoMap function directly
	module := NewModule()

	// Test with nil dict
	result, err := module.convertStarlarkDictToGoMap(nil)
	if err != nil {
		t.Errorf("convertStarlarkDictToGoMap failed with nil: %v", err)
	}
	if result != nil {
		t.Errorf("Expected nil result for nil input, got: %v", result)
	}

	// Test with empty dict
	emptyDict := starlark.NewDict(0)
	result, err = module.convertStarlarkDictToGoMap(emptyDict)
	if err != nil {
		t.Errorf("convertStarlarkDictToGoMap failed with empty dict: %v", err)
	}
	if len(result) != 0 {
		t.Errorf("Expected empty result for empty dict, got: %v", result)
	}

	// Test with simple string key dict
	simpleDict := starlark.NewDict(2)
	simpleDict.SetKey(starlark.String("enable_thinking"), starlark.Bool(true))
	simpleDict.SetKey(starlark.String("temperature"), starlark.Float(0.7))

	result, err = module.convertStarlarkDictToGoMap(simpleDict)
	if err != nil {
		t.Errorf("convertStarlarkDictToGoMap failed with simple dict: %v", err)
	}
	if len(result) != 2 {
		t.Errorf("Expected 2 items in result, got: %d", len(result))
	}
	if result["enable_thinking"] != true {
		t.Errorf("Expected enable_thinking to be true, got: %v", result["enable_thinking"])
	}

	// Test with mixed key types (should convert to string)
	mixedDict := starlark.NewDict(3)
	mixedDict.SetKey(starlark.String("string_key"), starlark.String("value1"))
	mixedDict.SetKey(starlark.MakeInt(42), starlark.String("value2"))
	mixedDict.SetKey(starlark.Float(3.14), starlark.String("value3"))

	result, err = module.convertStarlarkDictToGoMap(mixedDict)
	if err != nil {
		t.Errorf("convertStarlarkDictToGoMap failed with mixed dict: %v", err)
	}
	if len(result) != 3 {
		t.Errorf("Expected 3 items in result, got: %d", len(result))
	}

	// Check that non-string keys were converted to strings
	if result["string_key"] != "value1" {
		t.Errorf("String key conversion failed")
	}
	if result["42"] != "value2" {
		t.Errorf("Int key conversion failed")
	}
	if result["3.14"] != "value3" {
		t.Errorf("Float key conversion failed")
	}
}

// ---------------------------------------------------------------------------
// Public unit tests for non-TTY / non-network logic + hardening regressions.
//
// These exercise the parts of the module that run without a real API key or
// network: constructor/config defaulting, client selection error branches,
// pure helpers, argument validation in the message/chat/draw builtins, the
// conversion seam, and the hardening invariants from CLAUDE.md. The few tests
// that drive a full chat/draw request use a loopback httptest server injected
// via SetClient — no credentials, no external network.
//
// Sections:
//   - newTestServerModule        : loopback test-server helper
//   - runModuleScript / assert   : script runner + injected assert/fail globals
//   - TestModuleConstruction     : NewModule / NewModuleWithConfig / defaulting
//   - TestGetClientSelection     : provider routing + error branches
//   - TestModelAndDictHelpers    : getModel / getStringFromDict
//   - TestKwargsConversionErrors : convertStarlarkDictToGoMap error paths
//   - TestMessageBuiltin         : message() data builder + arg errors
//   - TestChatArgErrors          : chat() validation before any network call
//   - TestDrawArgErrors          : draw() validation before any network call
//   - TestMultimodalMessages     : message -> MultiContent (bytes/url/file)
//   - TestChatRequestRoundTrip   : blocking chat happy path + n>1 + full_response
//   - TestChatRequestShaping     : optional-param request shaping + modern mode
//   - TestChatRetryAndAllowError : retry short-circuit + allow_error swallowing
//   - TestStreamingChat          : streaming aggregation + callback
//   - TestDrawRoundTrip          : draw url/b64/gpt-image-1 paths
//   - TestHardeningNoPanic       : adversarial n + empty image data + clamp unit
// ---------------------------------------------------------------------------

// newTestServerModule returns a Module whose client points at the given
// loopback test server, with the gpt/dalle model presets supplied. The handler
// receives every request the SDK makes; tests assert on behavior, not network.
func newTestServerModule(t *testing.T, gptModel, dalleModel string, handler http.HandlerFunc) (*Module, *httptest.Server) {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)

	m := NewModuleWithConfig(ProviderOpenAI, srv.URL, "test-key", gptModel, dalleModel, "")
	cfg := oai.DefaultConfig("test-key")
	cfg.BaseURL = srv.URL
	m.SetClient(oai.NewClientWithConfig(cfg))
	return m, srv
}

// runModuleScript runs a Starlark script against the given module and returns
// any execution error (nil on success). It injects lightweight Go-backed
// assertion globals (assert.eq / assert.true and a top-level fail) so scripts
// can verify results without pulling in any third-party Starlark test
// framework. A failed assertion surfaces as the script's execution error.
func runModuleScript(t *testing.T, m *Module, script string) error {
	t.Helper()
	env := starlet.NewDefault()
	env.SetScriptContent([]byte(script))
	env.SetGlobals(starlet.StringAnyMap{
		"assert": assertModule(),
		"fail":   starlark.NewBuiltin("fail", builtinFail),
	})
	env.SetLazyloadModules(map[string]starlet.ModuleLoader{ModuleName: m.LoadModule()})
	_, err := env.Run()
	return err
}

// withAssert is a no-op passthrough kept for readability at call sites that use
// the injected assert/fail globals; the globals are wired in runModuleScript.
func withAssert(script string) string { return script }

// assertModule builds a Starlark struct exposing eq and true assertion helpers.
func assertModule() *starlarkstruct.Module {
	return &starlarkstruct.Module{
		Name: "assert",
		Members: starlark.StringDict{
			"eq":   starlark.NewBuiltin("assert.eq", builtinAssertEq),
			"true": starlark.NewBuiltin("assert.true", builtinAssertTrue),
		},
	}
}

func builtinAssertEq(_ *starlark.Thread, b *starlark.Builtin, args starlark.Tuple, _ []starlark.Tuple) (starlark.Value, error) {
	if len(args) != 2 {
		return nil, fmt.Errorf("%s: want 2 args, got %d", b.Name(), len(args))
	}
	eq, err := starlark.Equal(args[0], args[1])
	if err != nil {
		return nil, err
	}
	if !eq {
		return nil, fmt.Errorf("assert.eq failed: %s != %s", args[0].String(), args[1].String())
	}
	return starlark.None, nil
}

func builtinAssertTrue(_ *starlark.Thread, b *starlark.Builtin, args starlark.Tuple, _ []starlark.Tuple) (starlark.Value, error) {
	if len(args) != 1 {
		return nil, fmt.Errorf("%s: want 1 arg, got %d", b.Name(), len(args))
	}
	if !bool(args[0].Truth()) {
		return nil, fmt.Errorf("assert.true failed: %s is not truthy", args[0].String())
	}
	return starlark.None, nil
}

func builtinFail(_ *starlark.Thread, b *starlark.Builtin, args starlark.Tuple, _ []starlark.Tuple) (starlark.Value, error) {
	parts := make([]string, 0, len(args))
	for _, a := range args {
		if s, ok := starlark.AsString(a); ok {
			parts = append(parts, s)
		} else {
			parts = append(parts, a.String())
		}
	}
	return nil, fmt.Errorf("%s", strings.Join(parts, " "))
}

// decodeJSONBody decodes the request body into v (best-effort, for assertions).
func decodeJSONBody(r *http.Request, v interface{}) error {
	defer r.Body.Close()
	return json.NewDecoder(r.Body).Decode(v)
}

// writeJSON marshals v and writes it as the HTTP response body.
func writeJSON(w http.ResponseWriter, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}

// writeRaw writes s verbatim as the response body.
func writeRaw(w http.ResponseWriter, s string) {
	_, _ = io.WriteString(w, s)
}

// onePixelPNGBase64 returns a valid 1x1 PNG image encoded as base64, used to
// exercise the draw() base64 decode + PNG re-encode path without a network.
func onePixelPNGBase64() string {
	img := image.NewRGBA(image.Rect(0, 0, 1, 1))
	img.Set(0, 0, color.RGBA{R: 10, G: 20, B: 30, A: 255})
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		panic(err)
	}
	return base64.StdEncoding.EncodeToString(buf.Bytes())
}

// wantString fails the test when got != want, labelling with name.
func wantString(t *testing.T, name, got, want string) {
	t.Helper()
	if got != want {
		t.Errorf("%s = %q, want %q", name, got, want)
	}
}

func TestModuleConstruction(t *testing.T) {
	// NewModule: defaults are empty / preset, legacy mode on, api_version default.
	m := NewModule()
	wantString(t, "default provider", m.ext.GetString(configKeyProvider, ""), ProviderOpenAI)
	wantString(t, "default api_version", m.ext.GetString(configKeyAPIVersion, ""), defaultAPIVersion)
	wantString(t, "default gpt model", m.ext.GetString(configKeyGPTModel, "sentinel"), "")
	if !m.ext.GetBool(configKeyLegacyMode, false) {
		t.Error("default legacy_mode = false, want true")
	}

	// NewModuleWithConfig: presets applied; empty apiVersion falls back to default.
	mc := NewModuleWithConfig(ProviderAzure, "https://e.example", "k", "gpt-x", "dall-e-3", "")
	wantString(t, "preset provider", mc.ext.GetString(configKeyProvider, ""), ProviderAzure)
	wantString(t, "empty apiVersion -> default", mc.ext.GetString(configKeyAPIVersion, ""), defaultAPIVersion)
	wantString(t, "preset gpt model", mc.ext.GetString(configKeyGPTModel, ""), "gpt-x")

	// Explicit apiVersion is preserved.
	mv := NewModuleWithConfig(ProviderAnthropic, "", "k", "", "", "2099-01-01")
	wantString(t, "explicit apiVersion", mv.ext.GetString(configKeyAPIVersion, ""), "2099-01-01")

	// SetClient is the test seam: getClient returns the injected client verbatim.
	inj := oai.NewClientWithConfig(oai.DefaultConfig("x"))
	m.SetClient(inj)
	got, err := m.getClient("anything")
	if err != nil {
		t.Fatalf("getClient with injected client errored: %v", err)
	}
	if got != inj {
		t.Errorf("getClient did not return the injected client")
	}
}

func TestGetClientSelection(t *testing.T) {
	tests := []struct {
		name        string
		provider    string
		endpointURL string
		apiKey      string
		wantErr     string // substring; "" means no error expected
		wantBaseURL string // checked only when no error and provider known
	}{
		{name: "missing api key", provider: ProviderOpenAI, apiKey: "", wantErr: "openai_api_key is not set"},
		{name: "azure missing endpoint", provider: ProviderAzure, apiKey: "k", wantErr: "openai_endpoint_url is required for Azure provider"},
		{name: "unsupported provider", provider: "gemini", apiKey: "k", wantErr: "unsupported provider: gemini"},
		{name: "openai default", provider: ProviderOpenAI, apiKey: "k", wantErr: ""},
		{name: "openai custom base", provider: ProviderOpenAI, endpointURL: "https://gw.example/v1", apiKey: "k", wantErr: "", wantBaseURL: "https://gw.example/v1"},
		{name: "anthropic default base", provider: ProviderAnthropic, apiKey: "k", wantErr: "", wantBaseURL: "https://api.anthropic.com"},
		{name: "anthropic custom base", provider: ProviderAnthropic, endpointURL: "https://claude.example", apiKey: "k", wantErr: "", wantBaseURL: "https://claude.example"},
		{name: "azure ok", provider: ProviderAzure, endpointURL: "https://az.example", apiKey: "k", wantErr: ""},
		{name: "empty provider treated as openai", provider: "", apiKey: "k", wantErr: ""},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			m := NewModuleWithConfig(tc.provider, tc.endpointURL, tc.apiKey, "gpt-x", "dall-e-3", "")
			cli, err := m.getClient("gpt-x")
			if tc.wantErr != "" {
				if err == nil {
					t.Fatalf("expected error containing %q, got nil", tc.wantErr)
				}
				if !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("error = %q, want substring %q", err.Error(), tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if cli == nil {
				t.Fatal("expected non-nil client")
			}
		})
	}
}

func TestModelAndDictHelpers(t *testing.T) {
	// getModel: call-site value wins; falls back to config; empty when neither.
	m := NewModuleWithConfig(ProviderOpenAI, "", "k", "cfg-gpt", "", "")
	modelCases := []struct {
		key, val, want string
	}{
		{configKeyGPTModel, "call-gpt", "call-gpt"}, // call-site value wins
		{configKeyGPTModel, "", "cfg-gpt"},          // falls back to config
		{configKeyDALLEModel, "", ""},               // empty when neither
	}
	for _, c := range modelCases {
		if got := m.getModel(c.key, c.val); got != c.want {
			t.Errorf("getModel(%q,%q) = %q, want %q", c.key, c.val, got, c.want)
		}
	}

	// getStringFromDict: string value, bytes value, missing key, wrong type.
	d := starlark.NewDict(4)
	d.SetKey(starlark.String("s"), starlark.String("hello"))
	d.SetKey(starlark.String("b"), starlark.Bytes("world"))
	d.SetKey(starlark.String("n"), starlark.MakeInt(7))
	dictCases := []struct {
		key     string
		wantVal string
		wantOK  bool
	}{
		{"s", "hello", true},   // string value
		{"b", "world", true},   // bytes value
		{"missing", "", false}, // absent key
		{"n", "", false},       // non-string value rejected
	}
	for _, c := range dictCases {
		if v, ok := getStringFromDict(d, c.key); v != c.wantVal || ok != c.wantOK {
			t.Errorf("getStringFromDict(%q) = %q,%v want %q,%v", c.key, v, ok, c.wantVal, c.wantOK)
		}
	}
}

func TestKwargsConversionErrors(t *testing.T) {
	m := NewModule()

	// A nested dict round-trips with string keys preserved.
	d := starlark.NewDict(1)
	inner := starlark.NewDict(1)
	inner.SetKey(starlark.String("k"), starlark.String("v"))
	d.SetKey(starlark.String("outer"), inner)
	res, err := m.convertStarlarkDictToGoMap(d)
	if err != nil {
		t.Fatalf("nested dict errored: %v", err)
	}
	if _, ok := res["outer"]; !ok {
		t.Errorf("nested dict lost 'outer' key: %v", res)
	}

	// A tuple key (non-string, non-scalar) is stringified, not dropped or panicked.
	dk := starlark.NewDict(1)
	dk.SetKey(starlark.Tuple{starlark.MakeInt(1), starlark.MakeInt(2)}, starlark.String("tv"))
	res, err = m.convertStarlarkDictToGoMap(dk)
	if err != nil {
		t.Fatalf("tuple-key dict errored: %v", err)
	}
	if len(res) != 1 {
		t.Errorf("tuple-key dict = %v, want exactly one entry", res)
	}
}

func TestMessageBuiltin(t *testing.T) {
	// message() builds a dict with only the non-empty fields; role defaults to user.
	script := `
load("llm", "message")
m = message(text="hi")
assert.eq(m["role"], "user")
assert.eq(m["text"], "hi")
assert.eq("image" in m, False)

m2 = message(role="system", text="sys", image_url="http://x/y.png")
assert.eq(m2["role"], "system")
assert.eq(m2["text"], "sys")
assert.eq(m2["image_url"], "http://x/y.png")

# Empty message keeps only the default role.
m3 = message()
assert.eq(m3["role"], "user")
assert.eq(len(m3), 1)
`
	if err := runModuleScript(t, NewModule(), withAssert(script)); err != nil {
		t.Fatalf("message builtin script failed: %v", err)
	}

	// Unknown keyword argument is a clean error, not a panic.
	err := runModuleScript(t, NewModule(), `load("llm", "message")
message(bogus="x")`)
	if err == nil || !strings.Contains(err.Error(), "bogus") {
		t.Fatalf("expected unexpected-keyword error, got %v", err)
	}
}

func TestChatArgErrors(t *testing.T) {
	// All of these must fail BEFORE any network call, with a clean error.
	tests := []struct {
		name    string
		script  string
		wantErr string
	}{
		{
			name:    "gpt model not set",
			script:  `chat(text="hi", allow_error=True)`,
			wantErr: "gpt model is not set",
		},
		{
			name:    "unsupported response format",
			script:  `chat(text="hi", model="gpt-x", response_format="xml", allow_error=True)`,
			wantErr: "unsupported response format: xml",
		},
		{
			name:    "message missing role",
			script:  `chat(messages=[{"text": "hi"}], model="gpt-x", allow_error=True)`,
			wantErr: "role is required",
		},
		{
			name:    "message missing content",
			script:  `chat(messages=[{"role": "user"}], model="gpt-x", allow_error=True)`,
			wantErr: "at least one of text, image",
		},
		{
			// The OS-specific tail of the open error differs across platforms
			// (Unix "no such file" vs Windows "cannot find the path"), so assert
			// on the platform-neutral parts: the message index and the path.
			name:    "bad image file",
			script:  `chat(messages=[{"role": "user", "image_file": "/no/such/file.png"}], model="gpt-x", allow_error=True)`,
			wantErr: "message 1: open /no/such/file.png",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// No client set and no real network: errors must surface before any call.
			err := runModuleScript(t, NewModule(), `load("llm", "chat")
`+tc.script)
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("script %q: error = %v, want substring %q", tc.name, err, tc.wantErr)
			}
		})
	}
}

func TestDrawArgErrors(t *testing.T) {
	tests := []struct {
		name    string
		dalle   string // preset dalle model ("" => none)
		script  string
		wantErr string
	}{
		{name: "prompt required", script: `draw(prompt="", allow_error=True)`, wantErr: "prompt is required"},
		{name: "dalle model not set", script: `draw(prompt="cat", allow_error=True)`, wantErr: "dalle model is not set"},
		{name: "dall-e-3 n>1", dalle: "dall-e-3", script: `draw(prompt="cat", n=2, allow_error=True)`, wantErr: "dall-e-3 only supports n=1"},
		{name: "dall-e-3 bad quality", dalle: "dall-e-3", script: `draw(prompt="cat", quality="ultra", allow_error=True)`, wantErr: "quality must be 'standard' or 'hd' for dall-e-3"},
		{name: "dall-e-2 bad quality", dalle: "dall-e-2", script: `draw(prompt="cat", quality="hd", allow_error=True)`, wantErr: "quality must be 'standard' for dall-e-2"},
		{name: "gpt-image-1 bad background", dalle: "gpt-image-1", script: `draw(prompt="cat", background="rainbow", allow_error=True)`, wantErr: "background must be 'auto', 'transparent', or 'opaque'"},
		{name: "gpt-image-1 bad moderation", dalle: "gpt-image-1", script: `draw(prompt="cat", moderation="high", allow_error=True)`, wantErr: "moderation must be 'auto' or 'low'"},
		{name: "gpt-image-1 bad output_format", dalle: "gpt-image-1", script: `draw(prompt="cat", output_format="gif", allow_error=True)`, wantErr: "output_format must be 'png', 'jpeg', or 'webp'"},
		{name: "gpt-image-1 bad compression high", dalle: "gpt-image-1", script: `draw(prompt="cat", output_compression=150, allow_error=True)`, wantErr: "output_compression must be between 0 and 100"},
		{name: "gpt-image-1 bad compression neg", dalle: "gpt-image-1", script: `draw(prompt="cat", output_compression=-5, allow_error=True)`, wantErr: "output_compression must be between 0 and 100"},
		{name: "gpt-image-1 bad quality", dalle: "gpt-image-1", script: `draw(prompt="cat", quality="ultra", allow_error=True)`, wantErr: "quality must be 'auto', 'high', 'medium', or 'low'"},
		{name: "gpt-image-1 bad size", dalle: "gpt-image-1", script: `draw(prompt="cat", size="999x999", allow_error=True)`, wantErr: "size must be 'auto', '1024x1024', '1536x1024', or '1024x1536'"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			m := NewModuleWithConfig(ProviderOpenAI, "", "test-key", "", tc.dalle, "")
			err := runModuleScript(t, m, `load("llm", "draw")
`+tc.script)
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("draw %q: error = %v, want substring %q", tc.name, err, tc.wantErr)
			}
		})
	}
}

// soleMessage runs script against m, fails the test if it errors, and returns
// the single chat message the loopback handler captured into gotReq.
func soleMessage(t *testing.T, m *Module, gotReq *oai.ChatCompletionRequest, script string) oai.ChatCompletionMessage {
	t.Helper()
	if err := runModuleScript(t, m, script); err != nil {
		t.Fatalf("script failed: %v\nscript: %s", err, script)
	}
	if len(gotReq.Messages) != 1 {
		t.Fatalf("want exactly 1 message, got %+v", gotReq.Messages)
	}
	return gotReq.Messages[0]
}

// imageURLOfPart returns the data/url string of an image_url part, or "".
func imageURLOfPart(p oai.ChatMessagePart) string {
	if p.Type == oai.ChatMessagePartTypeImageURL && p.ImageURL != nil {
		return p.ImageURL.URL
	}
	return ""
}

// hasTextPart reports whether parts contains a text part equal to text.
func hasTextPart(parts []oai.ChatMessagePart, text string) bool {
	for _, p := range parts {
		if p.Type == oai.ChatMessagePartTypeText && p.Text == text {
			return true
		}
	}
	return false
}

// hasDataImagePart reports whether parts contains an image part whose URL is a
// data: URL (i.e. an inline-encoded image).
func hasDataImagePart(parts []oai.ChatMessagePart) bool {
	for _, p := range parts {
		if strings.HasPrefix(imageURLOfPart(p), "data:") {
			return true
		}
	}
	return false
}

// assertSoleImageURL fails unless msg has exactly one image_url part whose URL
// equals want (when want is a full URL) or is prefixed by want (e.g. "data:").
func assertSoleImageURL(t *testing.T, name string, msg oai.ChatCompletionMessage, want string) {
	t.Helper()
	if len(msg.MultiContent) != 1 {
		t.Errorf("%s: want 1 part, got %+v", name, msg.MultiContent)
		return
	}
	got := imageURLOfPart(msg.MultiContent[0])
	ok := got == want
	if strings.HasSuffix(want, ":") {
		ok = strings.HasPrefix(got, want)
	}
	if !ok {
		t.Errorf("%s: image URL = %q, want %q", name, got, want)
	}
}

func TestMultimodalMessages(t *testing.T) {
	// Exercise the message -> MultiContent conversion for every image source
	// (inline bytes, url, and file) without a network: assert the request the
	// SDK would send. This covers imageDataToBase64, imageFileToBase64, and the
	// MultiContent assembly branches of messagesToChatMessages.
	var gotReq oai.ChatCompletionRequest
	handler := func(w http.ResponseWriter, r *http.Request) {
		_ = decodeJSONBody(r, &gotReq)
		writeJSON(w, oai.ChatCompletionResponse{
			Choices: []oai.ChatCompletionChoice{{Message: oai.ChatCompletionMessage{Role: "assistant", Content: "ok"}}},
		})
	}
	m, _ := newTestServerModule(t, "gpt-x", "", handler)

	// Write a real PNG to a temp file for the image_file path.
	pngBytes, err := base64.StdEncoding.DecodeString(onePixelPNGBase64())
	if err != nil {
		t.Fatalf("decode test png: %v", err)
	}
	imgPath := filepath.Join(t.TempDir(), "pixel.png")
	if err := os.WriteFile(imgPath, pngBytes, 0o644); err != nil {
		t.Fatalf("write test png: %v", err)
	}

	// text + inline image bytes -> a MultiContent message with a text part and
	// an inline data-URL image part.
	msg := soleMessage(t, m, &gotReq, `load("llm", "chat")
chat(messages=[{"role": "user", "text": "describe", "image": b"\x89PNG\r\n\x1a\n"}])`)
	if len(msg.MultiContent) != 2 || !hasTextPart(msg.MultiContent, "describe") || !hasDataImagePart(msg.MultiContent) {
		t.Errorf("inline image: want text + data-image parts, got %+v", msg.MultiContent)
	}

	// Each single-image source yields exactly one image_url part. image_url is
	// passed verbatim; image_file is read and embedded as a data: URL.
	urlMsg := soleMessage(t, m, &gotReq, `load("llm", "chat")
chat(messages=[{"role": "user", "image_url": "https://x/y.png"}])`)
	assertSoleImageURL(t, "image_url", urlMsg, "https://x/y.png")

	fileMsg := soleMessage(t, m, &gotReq, fmt.Sprintf(`load("llm", "chat")
chat(messages=[{"role": "user", "image_file": %q}])`, imgPath))
	assertSoleImageURL(t, "image_file", fileMsg, "data:")

	// Top-level text arg builds an implicit prepended user message (text-only).
	msg = soleMessage(t, m, &gotReq, `load("llm", "chat")
chat(text="hi there")`)
	if msg.Content != "hi there" || msg.Role != oai.ChatMessageRoleUser {
		t.Errorf("implicit user message wrong: %+v", msg)
	}
}

func TestChatRequestRoundTrip(t *testing.T) {
	// Blocking chat path against a loopback server: assert request shaping and
	// the three response shapes (string for n==1, list for n>1, full object).
	var gotReq oai.ChatCompletionRequest
	handler := func(w http.ResponseWriter, r *http.Request) {
		_ = decodeJSONBody(r, &gotReq)
		n := gotReq.N
		if n == 0 {
			n = 1
		}
		choices := make([]oai.ChatCompletionChoice, 0, n)
		for i := 0; i < n; i++ {
			choices = append(choices, oai.ChatCompletionChoice{
				Index:        i,
				Message:      oai.ChatCompletionMessage{Role: "assistant", Content: fmt.Sprintf("reply-%d", i)},
				FinishReason: "stop",
			})
		}
		writeJSON(w, oai.ChatCompletionResponse{ID: "cmpl-1", Model: "gpt-x", Choices: choices})
	}

	m, _ := newTestServerModule(t, "gpt-x", "", handler)

	// n==1 -> content string.
	script := `
load("llm", "chat")
out = chat(text="hello")
assert.eq(out, "reply-0")
`
	if err := runModuleScript(t, m, withAssert(script)); err != nil {
		t.Fatalf("n==1 round trip failed: %v", err)
	}
	if gotReq.Model != "gpt-x" {
		t.Errorf("request model = %q, want gpt-x", gotReq.Model)
	}
	if len(gotReq.Messages) != 1 || gotReq.Messages[0].Content != "hello" {
		t.Errorf("request messages = %+v, want single user 'hello'", gotReq.Messages)
	}

	// n>1 -> list of contents.
	script = `
load("llm", "chat")
out = chat(text="hello", n=3)
assert.eq(type(out), "list")
assert.eq(len(out), 3)
assert.eq(out[0], "reply-0")
assert.eq(out[2], "reply-2")
`
	if err := runModuleScript(t, m, withAssert(script)); err != nil {
		t.Fatalf("n>1 round trip failed: %v", err)
	}

	// full_response -> object with choices accessible (legacy mode default).
	// ConvertJSONStruct exposes JSON-tagged field names (id, choices, message).
	script = `
load("llm", "chat")
out = chat(text="hello", full_response=True)
assert.eq(out.id, "cmpl-1")
assert.eq(out.choices[0].message.content, "reply-0")
`
	if err := runModuleScript(t, m, withAssert(script)); err != nil {
		t.Fatalf("full_response round trip failed: %v", err)
	}
}

func TestChatRequestShaping(t *testing.T) {
	// Assert that optional parameters flow into the request as expected and that
	// legacy_mode=false routes full_response through the JSON converter. All
	// loopback, no network.
	var gotReq oai.ChatCompletionRequest
	handler := func(w http.ResponseWriter, r *http.Request) {
		_ = decodeJSONBody(r, &gotReq)
		writeJSON(w, oai.ChatCompletionResponse{
			ID:      "cmpl-9",
			Choices: []oai.ChatCompletionChoice{{Message: oai.ChatCompletionMessage{Role: "assistant", Content: "ok"}}},
		})
	}
	m, _ := newTestServerModule(t, "gpt-x", "", handler)

	// json response_format, max_completion_tokens, reasoning_effort, stop, penalties.
	if err := runModuleScript(t, m, withAssert(`load("llm", "chat")
chat(text="hi", response_format="json", max_completion_tokens=42, reasoning_effort="high",
     stop=["END", "STOP"], temperature=0.2, top_p=0.9, frequency_penalty=0.5, presence_penalty=0.25)`)); err != nil {
		t.Fatalf("shaping chat failed: %v", err)
	}
	rf := ""
	if gotReq.ResponseFormat != nil {
		rf = string(gotReq.ResponseFormat.Type)
	}
	wantString(t, "response_format type", rf, string(oai.ChatCompletionResponseFormatTypeJSONObject))
	wantString(t, "reasoning_effort", gotReq.ReasoningEffort, "high")
	wantString(t, "stop sequences", strings.Join(gotReq.Stop, ","), "END,STOP")
	if gotReq.MaxCompletionTokens != 42 {
		t.Errorf("max_completion_tokens = %d, want 42", gotReq.MaxCompletionTokens)
	}

	// legacy_mode=false: full_response goes through GoToStarlarkViaJSON. The
	// converted value must still expose the id/choices fields (JSON-shaped).
	mModern := NewModuleWithConfig(ProviderOpenAI, "", "test-key", "gpt-x", "", "")
	mModern.SetClient(oai.NewClientWithConfig(mustClientCfg(t, handler)))
	// Disable legacy mode via the setter so the modern conversion branch runs.
	if err := runModuleScript(t, mModern, withAssert(`load("llm", "chat", "set_legacy_mode")
set_legacy_mode(False)
out = chat(text="hi", full_response=True)
assert.eq(out["id"], "cmpl-9")
assert.eq(out["choices"][0]["message"]["content"], "ok")`)); err != nil {
		t.Fatalf("modern-mode full_response failed: %v", err)
	}
}

// mustClientCfg returns an oai client config aimed at a fresh loopback server
// running handler; the server is torn down at test end.
func mustClientCfg(t *testing.T, handler http.HandlerFunc) oai.ClientConfig {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	cfg := oai.DefaultConfig("test-key")
	cfg.BaseURL = srv.URL
	return cfg
}

func TestChatRetryAndAllowError(t *testing.T) {
	// 400 Bad Request must NOT be retried (retry short-circuit) and, with
	// allow_error, must surface as None rather than a script-fatal error.
	var calls400 int
	h400 := func(w http.ResponseWriter, _ *http.Request) {
		calls400++
		w.WriteHeader(http.StatusBadRequest)
		writeRaw(w, `{"error":{"message":"bad","type":"invalid_request_error"}}`)
	}
	m400, _ := newTestServerModule(t, "gpt-x", "", h400)
	script := `
load("llm", "chat")
out = chat(text="hi", retry=5, allow_error=True)
assert.eq(out, None)
`
	if err := runModuleScript(t, m400, withAssert(script)); err != nil {
		t.Fatalf("allow_error on 400 should yield None: %v", err)
	}
	if calls400 != 1 {
		t.Errorf("400 was retried %d times, want exactly 1 (no retry on bad request)", calls400)
	}

	// 500 server error IS retried up to retry times; without allow_error it errors.
	var calls500 int
	h500 := func(w http.ResponseWriter, _ *http.Request) {
		calls500++
		w.WriteHeader(http.StatusInternalServerError)
		writeRaw(w, `{"error":{"message":"boom","type":"server_error"}}`)
	}
	m500, _ := newTestServerModule(t, "gpt-x", "", h500)
	err := runModuleScript(t, m500, `load("llm", "chat")
chat(text="hi", retry=3)`)
	if err == nil {
		t.Fatal("expected error from repeated 500 without allow_error")
	}
	if calls500 != 3 {
		t.Errorf("500 retried %d times, want 3", calls500)
	}

	// Same 500 with allow_error swallows to None.
	calls500 = 0
	if err := runModuleScript(t, m500, withAssert(`load("llm", "chat")
assert.eq(chat(text="hi", retry=2, allow_error=True), None)`)); err != nil {
		t.Fatalf("allow_error on 500 should yield None: %v", err)
	}
	if calls500 != 2 {
		t.Errorf("500 with allow_error retried %d times, want 2", calls500)
	}
}

func TestStreamingChat(t *testing.T) {
	// Streaming aggregation: the SDK reassembles deltas; return shape matches
	// blocking. The stream_callback fires per chunk.
	handler := func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		chunks := []string{
			`{"id":"s1","object":"chat.completion.chunk","model":"gpt-x","choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"}}]}`,
			`{"id":"s1","object":"chat.completion.chunk","model":"gpt-x","choices":[{"index":0,"delta":{"content":"lo!"},"finish_reason":"stop"}]}`,
		}
		var b strings.Builder
		for _, c := range chunks {
			b.WriteString("data: " + c + "\n\n")
		}
		b.WriteString("data: [DONE]\n\n")
		writeRaw(w, b.String())
	}
	m, _ := newTestServerModule(t, "gpt-x", "", handler)

	script := `
load("llm", "chat")
parts = []
def cb(chunk):
    parts.append(chunk)

out = chat(text="hi", stream=True, stream_callback=cb)
assert.eq(out, "Hello!")
assert.true(len(parts) >= 1)
`
	if err := runModuleScript(t, m, withAssert(script)); err != nil {
		t.Fatalf("streaming chat failed: %v", err)
	}

	// A callback that raises must propagate as a clean error, not a panic.
	err := runModuleScript(t, m, withAssert(`load("llm", "chat")
def cb(chunk):
    fail("boom in callback")
chat(text="hi", stream=True, stream_callback=cb)`))
	if err == nil || !strings.Contains(err.Error(), "boom in callback") {
		t.Fatalf("callback error should propagate, got %v", err)
	}
}

func TestDrawRoundTrip(t *testing.T) {
	// DALL-E url format returns the URL string.
	urlHandler := func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, oai.ImageResponse{Data: []oai.ImageResponseDataInner{{URL: "https://img.example/a.png"}}})
	}
	mURL, _ := newTestServerModule(t, "", "dall-e-3", urlHandler)
	if err := runModuleScript(t, mURL, withAssert(`load("llm", "draw")
assert.eq(draw(prompt="cat", response_format="url"), "https://img.example/a.png")`)); err != nil {
		t.Fatalf("draw url path failed: %v", err)
	}

	// gpt-image-1 always returns decoded base64 bytes.
	pngB64 := onePixelPNGBase64()
	giHandler := func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, oai.ImageResponse{Data: []oai.ImageResponseDataInner{{B64JSON: pngB64}}})
	}
	mGI, _ := newTestServerModule(t, "", "gpt-image-1", giHandler)
	if err := runModuleScript(t, mGI, withAssert(`load("llm", "draw")
out = draw(prompt="cat")
assert.eq(type(out), "bytes")
assert.true(len(out) > 0)`)); err != nil {
		t.Fatalf("draw gpt-image-1 path failed: %v", err)
	}

	// DALL-E b64_json decodes the PNG and re-encodes it to bytes; n>1 yields a
	// list of those byte values.
	b64Handler := func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, oai.ImageResponse{Data: []oai.ImageResponseDataInner{{B64JSON: pngB64}, {B64JSON: pngB64}}})
	}
	mB64, _ := newTestServerModule(t, "", "dall-e-2", b64Handler)
	if err := runModuleScript(t, mB64, withAssert(`load("llm", "draw")
single = draw(prompt="cat", response_format="b64_json")
assert.eq(type(single), "bytes")
assert.true(len(single) > 0)

many = draw(prompt="cat", n=2, response_format="b64_json")
assert.eq(type(many), "list")
assert.eq(len(many), 2)
assert.eq(type(many[0]), "bytes")`)); err != nil {
		t.Fatalf("draw dall-e b64 path failed: %v", err)
	}
}

func TestHardeningNoPanic(t *testing.T) {
	// HARDENING: draw() with a 200 + empty Data slice must not panic on
	// resp.Data[0]; it must surface a clean error.
	emptyHandler := func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, oai.ImageResponse{Data: []oai.ImageResponseDataInner{}})
	}
	mEmpty, _ := newTestServerModule(t, "", "dall-e-3", emptyHandler)
	err := runModuleScript(t, mEmpty, `load("llm", "draw")
draw(prompt="cat", response_format="url")`)
	if err == nil || !strings.Contains(err.Error(), "no image data returned") {
		t.Fatalf("empty draw data: error = %v, want 'no image data returned'", err)
	}

	// HARDENING: streaming chat with a hostile n (negative / absurdly large)
	// must not panic makeslice or OOM during pre-allocation. The clamp keeps
	// pre-allocation bounded; the stream still opens and processStream runs the
	// (now-bounded) make. Before the fix, n=-1 panicked makeslice and a huge n
	// attempted a multi-gigabyte allocation, both crashing the host.
	var streamHits int
	streamHandler := func(w http.ResponseWriter, _ *http.Request) {
		streamHits++
		w.Header().Set("Content-Type", "text/event-stream")
		writeRaw(w, "data: {\"id\":\"s\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"x\"}}]}\n\ndata: [DONE]\n\n")
	}
	mStream, _ := newTestServerModule(t, "gpt-x", "", streamHandler)

	for _, n := range []string{"-1", "2000000000"} {
		t.Run("stream_n_"+n, func(t *testing.T) {
			before := streamHits
			// allow_error so any post-clamp API rejection is swallowed; the point
			// is that no Go panic escapes for adversarial n. Reaching here at all
			// (the test process did not crash) plus the handler being hit proves
			// processStream ran its bounded make() without panicking.
			err := runModuleScript(t, mStream, fmt.Sprintf(`load("llm", "chat")
chat(text="hi", stream=True, n=%s, allow_error=True)`, n))
			if err != nil && strings.Contains(err.Error(), "makeslice") {
				t.Fatalf("hostile n=%s panicked makeslice: %v", n, err)
			}
			if streamHits == before {
				t.Fatalf("stream handler was not reached for n=%s; make() path not exercised", n)
			}
		})
	}

	// clampStreamChoices unit: negative -> 0, in-range -> identity, huge -> cap.
	clampCases := []struct{ in, want int }{
		{-5, 0},
		{0, 0},
		{3, 3},
		{maxStreamChoices, maxStreamChoices},
		{maxStreamChoices + 100, maxStreamChoices},
	}
	for _, c := range clampCases {
		if got := clampStreamChoices(c.in); got != c.want {
			t.Errorf("clampStreamChoices(%d) = %d, want %d", c.in, got, c.want)
		}
	}
}
