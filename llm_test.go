package llm

import (
	"testing"

	"github.com/starpkg/base"
)

// TestStarlarkScripts runs Starlark test scripts from the test directory.
// Scripts with "test-" prefix should succeed, "panic-" prefix should fail.
func TestStarlarkScripts(t *testing.T) {
	module := NewModule()
	moduleLoader := module.LoadModule()
	extraModules := []string{"runtime", "file", "path", "json", "time", "go_idiomatic"}

	// Use the helper function from the base package
	base.RunStarlarkTests(t, ModuleName, moduleLoader, extraModules, "")
}
