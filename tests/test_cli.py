"""Tests for CLI argument parsing and provider support."""

import pytest

from tgb.cli import parse_model_spec, SUPPORTED_PROVIDERS, _normalize_provider


class TestParseModelSpec:
    def test_ollama(self):
        provider, model = parse_model_spec("ollama:qwen2.5:32b")
        assert provider == "ollama"
        assert model == "qwen2.5:32b"

    def test_openai(self):
        provider, model = parse_model_spec("openai:gpt-4o")
        assert provider == "openai"
        assert model == "gpt-4o"

    def test_claude(self):
        provider, model = parse_model_spec("claude:opus")
        assert provider == "claude"
        assert model == "opus"

    def test_gemini(self):
        provider, model = parse_model_spec("gemini:pro")
        assert provider == "gemini"
        assert model == "pro"

    def test_codex(self):
        provider, model = parse_model_spec("codex:latest")
        assert provider == "codex"
        assert model == "latest"

    def test_opencode(self):
        provider, model = parse_model_spec("opencode:default")
        assert provider == "opencode"
        assert model == "default"

    def test_no_colon_raises(self):
        with pytest.raises(ValueError, match="Invalid model spec"):
            parse_model_spec("justmodel")


class TestSupportedProviders:
    def test_all_providers_listed(self):
        assert "ollama" in SUPPORTED_PROVIDERS
        assert "openai" in SUPPORTED_PROVIDERS
        assert "claude" in SUPPORTED_PROVIDERS
        assert "gemini" in SUPPORTED_PROVIDERS
        assert "codex" in SUPPORTED_PROVIDERS
        assert "opencode" in SUPPORTED_PROVIDERS

    def test_aliases_listed(self):
        assert "codex-cli" in SUPPORTED_PROVIDERS
        assert "opencode_cli" in SUPPORTED_PROVIDERS


class TestNormalizeProvider:
    def test_codex_cli_alias(self):
        assert _normalize_provider("codex-cli") == "codex"

    def test_opencode_cli_alias(self):
        assert _normalize_provider("opencode_cli") == "opencode"

    def test_canonical_unchanged(self):
        assert _normalize_provider("ollama") == "ollama"
        assert _normalize_provider("codex") == "codex"
        assert _normalize_provider("opencode") == "opencode"
