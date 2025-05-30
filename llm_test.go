package llm

import (
	"strings"
	"testing"

	"github.com/1set/starlet"
	"github.com/starpkg/base"
	"go.starlark.net/starlark"
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
