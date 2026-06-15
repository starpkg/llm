# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`starpkg/llm` is an **L4 domain module** of the Star\* ecosystem: it exposes large-language-model and image-generation services to Starlark scripts. A script imports the module, optionally sets credentials/model, and calls `chat` (text/multimodal completions) or `draw` (image generation), with Go data marshalled to and from Starlark values.

Per the starpkg positioning — *"support for necessary LOCAL operations + simple abstractions over common ONLINE services, for ease of use"* — `llm` sits squarely on the **online-service side**: it is a thin, ergonomic wrapper over a remote SaaS API. There is no local model runtime here; every call reaches out over HTTP. The wrapped surface is OpenAI's API and anything OpenAI-compatible — **OpenAI**, **Azure OpenAI**, the **Anthropic Claude** API, and self-hosted/compatible gateways — selected by the `openai_provider` config.

Layer position: depends downward on `starpkg/base` (the configurable-module/config-option system), `1set/starlet` (the Machine + `dataconv` for Go⇄Starlark conversion and thread context), and transitively `1set/starlight` + `go.starlark.net`. The single third-party SDK is `github.com/sashabaranov/go-openai` (imported as `oai`). Nothing in the ecosystem depends on this module.

## Dev commands

Pure Go library with a Makefile. From this repo:

```bash
make test                                  # -race -cover, the working bar
make ci                                    # -race -cover profile + bench compile (what CI runs)
go test ./... -run TestKwargsConversion    # a single test
gofmt -l . && go vet ./...                 # must be clean before commit
go run github.com/1set/meta/doccov@master . # doc-coverage gate (exit 0 = every builtin documented)
```

**Verify on the go floor in Docker** — this repo's floor is **go 1.19** (its `go.mod`), which may be older than the local toolchain. Behavior on the floor must be checked in a container:

```bash
docker run --rm -v "$PWD":/src -v "$HOME/go/pkg/mod":/go/pkg/mod -w /src golang:1.19 go test -race -count=1 ./...
```

Live API tests need real credentials and a reachable endpoint; they are not part of the unit suite. Integration scripts under `../test/llm/*.star` live in the **private `starpkg/test` repo** and auto-skip when that directory is absent (e.g. in CI). The in-repo `TestKwargsParameter` only exercises *parameter parsing* with `allow_error=True`, so it does not need a live API.

## Architecture (the part that spans files)

The module is small and single-file by design — one SDK, one object surface, no per-connection state. Everything is in **`openai.go`**; `llm_test.go` holds the tests.

- **Module construction.** `Module` wraps a `base.ConfigurableModule` (`cfgMod`) plus its `Extend()` accessor (`ext`) and an optional injected `*oai.Client` (`cli`). `NewModule()` registers the seven config options with empty/preset defaults; `NewModuleWithConfig(provider, endpointURL, apiKey, gptModel, dalleModel, apiVersion)` presets them. Both route through `newModuleWithOptions`. `genConfigOption` is the shared builder that wires each option's name, description, default, and `LLM_<NAME>` env var; the API-key option is marked `SetSecret(true)`.
- **`LoadModule()`** registers three additional builtins — **`message`**, **`chat`**, **`draw`** — alongside the `set_<option>` / `get_<option>` builtins that `base.ConfigurableModule.LoadModule` generates from the config options. The API key is secret, so `set_openai_api_key` exists but `get_openai_api_key` does **not** (secrets are write-only from a script).
- **`message`** (`newMessageStruct`) builds a plain Starlark dict with the non-empty subset of `role`/`text`/`image`/`image_file`/`image_url`. It is data only — no API call. `chat`'s `messages=[...]` consumes exactly these dicts.
- **`chat`** (`genChatFunc`) is the text path. `parseChatParams` unpacks ~24 keyword args into `chatParams`; `prepareMessages` prepends an implicit user message built from the top-level `text`/`image*` args; `messagesToChatMessages` converts each dict to an `oai.ChatCompletionMessage` (text-only → `Content`; any image → `MultiContent` parts, with `image`/`image_file` base64-encoded into a `data:` URL via `imageDataToBase64`/`imageFileToBase64`); `prepareChatRequest` assembles the `oai.ChatCompletionRequest` (response_format text/json, reasoning_effort, kwargs → `ChatTemplateKwargs`). Then either `handleBlockingRequest` or `handleStreamingRequest`/`processStream` runs, both honoring `retry` and short-circuiting on HTTP 400. `formatChatResponse` returns a content string for `n==1`, a list for `n>1`, or the full converted response for `full_response=True`.
- **`draw`** (`genDrawFunc`) is the image path. It branches on model: **`dall-e-2`**, **`dall-e-3`**, **`gpt-image-1`**, each with its own validated parameter set and defaults. DALL-E `url` format returns the URL string; `b64_json` decodes → re-encodes PNG → returns bytes; `gpt-image-1` always returns decoded base64 bytes. `n==1` returns one value, `n>1` a list.
- **Streaming** (`processStream`) accumulates per-choice content builders and token usage across chunks, invoking `stream_callback` (if any) with each chunk converted to Starlark, then synthesizes one `*oai.ChatCompletionResponse` so the streaming and blocking return shapes are identical.
- **Conversion seam.** `convertGoToStarlark` is the one place SDK structs become Starlark values; `legacy_mode` (default `true`) selects `dataconv.ConvertJSONStruct` (direct struct-field access) vs `dataconv.GoToStarlarkViaJSON` (JSON-shaped). `convertStarlarkDictToGoMap` turns the `kwargs` dict into a Go map for `ChatTemplateKwargs`.
- **Client selection.** `getClient(model)` returns the injected `cli` if `SetClient` was used (the test seam), else builds an `oai.ClientConfig` per provider: Azure (`DefaultAzureConfig` + version + a model-mapper), Anthropic (`DefaultConfig` + `https://api.anthropic.com` base unless overridden), or vanilla OpenAI (`DefaultConfig`, optional base URL). `getModel(key, val)` prefers the call-site model, falling back to config.

## Invariants / hardening (preserve when editing)

This module talks to a paid, rate-limited, occasionally-failing network service, so its safety properties are about *not crashing the host* and *not surprising old scripts*:

1. **No host panics / no host crash from script input or API failure.** `chat` and `draw` return `(none, err)` (or `(none, nil)` under `allow_error=True`) on every failure path; they never panic. The base module defers config validation into the loader so a misconfiguration surfaces as a loader error, not a panic (PKG-03). Keep new error paths returning errors, not panicking.
2. **Retry only when it can help.** Both paths retry up to `retry` times but break immediately on an `*oai.APIError` with HTTP 400 (a bad request will never succeed on retry). Preserve this short-circuit; don't retry deterministic client errors.
3. **`allow_error` means "swallow to None".** When set, any request error becomes `None` instead of a script-fatal error — used by callers that prefer to branch on a falsy result. Don't bypass it on new error paths.
4. **Secrets are write-only from scripts.** `openai_api_key` is `SetSecret(true)`, so no `get_openai_api_key` builtin is generated. Never add a getter for a secret option, and never log the key.
5. **Backward compatibility (the iron rule).** Existing scripts must keep working unchanged. `legacy_mode` defaults to `true`; `max_tokens` defaults to `0` (unset → API default); `n` defaults to `1`. Any new option or behavior must default to the historical behavior. The return shape (string for `n==1`, list for `n>1`, full object for `full_response`) is API; do not change it.
6. **Deterministic conversion seam.** All SDK-struct → Starlark conversion goes through `convertGoToStarlark` so `legacy_mode` is honored uniformly; new response surfaces must route through it, not call `dataconv` directly.

## Test organization

Group by functional goal — **do not add one `*_test.go` per fix.** `llm_test.go` is the home: `TestStarlarkScripts` (the `../test/llm` integration harness via `base.RunStarlarkTests`, auto-skips when absent), `TestKwargsParameter` (kwargs parsing end-to-end with `allow_error=True`, no live API), and `TestKwargsConversion` (the `convertStarlarkDictToGoMap` unit). Add a new test as a **section here**, not a new file. Tests are table/example-driven; no third-party test framework. Anything that needs a live API key or network goes in the private `starpkg/test` repo as a `../test/llm/*.star` script, not in this repo's unit suite.

## Documentation

Three layers must stay in sync (enforced by the doc standard, `plan/starpkg文档标准（DOC-STD）`):

- **`README.md`** — every script-facing builtin (`message`, `chat`, `draw`) and every config setter/getter (`set_openai_api_key`, `get_openai_gpt_model`, …) documented as a backtick whole-word, with correct names, signatures, args, returns, and behavior verified against the code. Examples must be valid Starlark (no Python f-strings, no `print(..., end=...)`).
- **GoDoc** — package comment + a doc comment on every exported symbol (`ModuleName`, the `Provider*` consts, `Module`, `NewModule`, `NewModuleWithConfig`, `LoadModule`, `SetClient`), first word = symbol name (gated by `revive`'s `exported` rule in CI).
- **doc-coverage gate** — `go run github.com/1set/meta/doccov@master .` must exit 0. doccov statically scans the package for `starlark.NewBuiltin("<name>", …)` literals and checks each shows up as a backtick word in the README. The `set_*`/`get_*` builtins are generated in `base` with computed names, so doccov does not see them — document them anyway for accuracy (it is the right thing, just not gate-enforced here).

## Release discipline

- **Floor = go 1.19** (this repo's `go.mod`). A repo's floor only rises in its own pin PR.
- **CI** runs via the centralized reusable workflow in `1set/meta` (`.github/workflows/go-ci.yml@<sha>`), with `go-floor: "1.19"` and `doc-coverage: true`. Pin the reusable workflow to a full commit SHA; bump the pin deliberately.
- **Pin upgrade is the last PR** of a repo's series (`go.starlark.net` baseline + 1set deps + go floor), as one isolated PR — never folded into a feature/doc PR.
- **Bumping the version, the go floor, or tagging are user-confirmed actions** — never tag autonomously; default to patch bumps; published tags are immutable in the Go module proxy.
- **Open-source boundary.** This repo is public: code, comments, commits, PRs, and issues must not contain internal/business names.
