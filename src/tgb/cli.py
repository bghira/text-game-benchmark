"""CLI entry point for text-game-benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tgb.checks.registry import list_checks
from tgb.config import load_scenarios
from tgb.results import BenchmarkRun, print_summary
from tgb.rubric import Rubric, builtin_rubric_dir, load_rubrics


def parse_model_spec(spec: str) -> tuple[str, str]:
    """Parse 'provider:model_name' into (provider, model).

    Examples:
        'ollama:qwen2.5:32b' -> ('ollama', 'qwen2.5:32b')
        'openai:GLM-5' -> ('openai', 'GLM-5')
    """
    parts = spec.split(":", 1)
    if len(parts) < 2:
        raise ValueError(
            f"Invalid model spec '{spec}'. "
            "Expected format: provider:model_name (e.g., ollama:qwen2.5:32b)"
        )
    return parts[0], parts[1]


SUPPORTED_PROVIDERS = (
    "ollama", "openai", "claude", "gemini",
    "codex", "codex-cli",
    "opencode", "opencode_cli",
)

# Normalize aliases to canonical provider names
_PROVIDER_ALIASES: dict[str, str] = {
    "codex-cli": "codex",
    "opencode_cli": "opencode",
}


def _normalize_provider(provider: str) -> str:
    """Normalize provider alias to canonical name."""
    return _PROVIDER_ALIASES.get(provider, provider)


def build_client(
    provider: str,
    model: str,
    ollama_url: str,
    openai_url: str,
    openai_api_key: str,
):
    """Build a completion client for the given provider."""
    provider = _normalize_provider(provider)

    if provider == "ollama":
        from tgb.clients.ollama_client import OllamaClient
        return OllamaClient(model=model, base_url=ollama_url)
    elif provider == "openai":
        from tgb.clients.openai_compat import OpenAICompatClient
        client = OpenAICompatClient(
            base_url=openai_url,
            api_key=openai_api_key,
            model=model,
        )
        # Wrap to match CompletionClient protocol
        return _OpenAIClientAdapter(client)
    elif provider == "claude":
        from tgb.clients.cli_backends import ClaudeCLIClient
        return ClaudeCLIClient(model=model or None)
    elif provider == "gemini":
        from tgb.clients.cli_backends import GeminiCLIClient
        return GeminiCLIClient(model=model or None)
    elif provider == "codex":
        from tgb.clients.cli_backends import CodexCLIClient
        return CodexCLIClient(model=model or None)
    elif provider == "opencode":
        from tgb.clients.cli_backends import OpenCodeCLIClient
        return OpenCodeCLIClient(model=model or None)
    else:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Supported: {', '.join(SUPPORTED_PROVIDERS)}"
        )


class _OpenAIClientAdapter:
    """Adapts OpenAICompatClient to the CompletionClient protocol."""

    def __init__(self, client):
        self._client = client

    def complete(self, system_prompt, user_prompt, **opts):
        return self._client.complete_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            **opts,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tgb",
        description="text-game-benchmark — LLM compliance benchmark for text-game-engine",
    )
    parser.add_argument(
        "-s", "--scenario",
        action="append",
        dest="scenarios",
        metavar="PATH",
        help="Scenario YAML file or directory (can be repeated)",
    )
    parser.add_argument(
        "-m", "--model",
        action="append",
        dest="models",
        metavar="SPEC",
        help="Model spec as provider:name (e.g., ollama:qwen2.5:32b). Can be repeated.",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://127.0.0.1:11434",
        help="Ollama API base URL (default: http://127.0.0.1:11434)",
    )
    parser.add_argument(
        "--judge-model",
        default="",
        metavar="SPEC",
        help="Judge model spec (e.g., openai:GLM-5)",
    )
    parser.add_argument(
        "--judge-url",
        default="",
        help="Judge API base URL",
    )
    parser.add_argument(
        "--judge-api-key",
        default="",
        help="Judge API key",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip all judge checks",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="results",
        help="Output directory for results JSON (default: results/)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--list-checks",
        action="store_true",
        help="List all available check IDs and exit",
    )
    parser.add_argument(
        "--openai-url",
        default="",
        help="OpenAI-compatible API base URL for test subjects",
    )
    parser.add_argument(
        "--openai-api-key",
        default="",
        help="API key for OpenAI-compatible test subjects",
    )
    parser.add_argument(
        "--rubric-dir",
        action="append",
        dest="rubric_dirs",
        metavar="DIR",
        help="Additional rubric directory (can be repeated). Built-in rubrics always loaded.",
    )
    parser.add_argument(
        "--no-rubrics",
        action="store_true",
        help="Skip all rubric grading",
    )
    parser.add_argument(
        "--list-rubrics",
        action="store_true",
        help="List all available rubrics and exit",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_checks:
        checks = list_checks()
        print(f"Available checks ({len(checks)}):")
        for check_id in checks:
            print(f"  {check_id}")
        sys.exit(0)

    # Load rubrics
    rubric_dirs = [str(builtin_rubric_dir())]
    if args.rubric_dirs:
        rubric_dirs.extend(args.rubric_dirs)
    all_rubrics = load_rubrics(rubric_dirs) if not args.no_rubrics else {}

    if args.list_rubrics:
        print(f"Available rubrics ({len(all_rubrics)}):")
        for rid, rubric in sorted(all_rubrics.items()):
            print(f"  {rid}: {rubric.name} [{rubric.category}, {rubric.scope}]")
            if rubric.computed_metric:
                print(f"    computed_metric: {rubric.computed_metric}")
        sys.exit(0)

    if not args.scenarios:
        parser.error("At least one scenario (-s) is required")
    if not args.models:
        parser.error("At least one model (-m) is required")

    # Load scenarios
    scenarios = load_scenarios(args.scenarios)
    if not scenarios:
        print("No scenarios found.", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"Loaded {len(scenarios)} scenario(s)", file=sys.stderr)

    # Build judge if needed
    judge = None
    judge_model_name = ""
    judge_client = None
    if not args.no_judge and args.judge_model:
        from tgb.clients.openai_compat import OpenAICompatClient
        from tgb.judge import JudgeEvaluator

        judge_provider, judge_model_name = parse_model_spec(args.judge_model)
        judge_url = args.judge_url
        judge_key = args.judge_api_key

        if judge_provider == "ollama":
            judge_url = judge_url or args.ollama_url
            judge_url = f"{judge_url}/v1" if not judge_url.endswith("/v1") else judge_url
        elif not judge_url:
            parser.error("--judge-url required for non-ollama judge models")

        judge_client = OpenAICompatClient(
            base_url=judge_url,
            api_key=judge_key,
            model=judge_model_name,
        )
        judge = JudgeEvaluator(client=judge_client)

    # Build rubric grader if we have rubrics
    rubric_grader = None
    if all_rubrics and not args.no_rubrics:
        from tgb.rubric import RubricGrader
        # Reuse judge client for rubric grading; None means only computed metrics run
        rubric_grader = RubricGrader(client=judge_client)

    # Run benchmark
    from tgb.runner import run_scenario

    run = BenchmarkRun(judge_model=judge_model_name)
    run.start()

    for model_spec in args.models:
        provider, model_name = parse_model_spec(model_spec)

        if args.verbose:
            print(f"\nModel: {provider}:{model_name}", file=sys.stderr)

        client = build_client(
            provider=provider,
            model=model_name,
            ollama_url=args.ollama_url,
            openai_url=args.openai_url,
            openai_api_key=args.openai_api_key,
        )

        for scenario in scenarios:
            if args.verbose:
                print(f"\nScenario: {scenario.name}", file=sys.stderr)

            # Filter rubrics for this scenario
            scenario_rubrics: list[Rubric] = []
            if all_rubrics and rubric_grader:
                if scenario.rubrics:
                    # Scenario specifies which rubrics to apply
                    scenario_rubrics = [
                        all_rubrics[rid] for rid in scenario.rubrics
                        if rid in all_rubrics
                    ]
                else:
                    # No filter — apply all loaded rubrics
                    scenario_rubrics = list(all_rubrics.values())

            result = run_scenario(
                scenario=scenario,
                client=client,
                provider=provider,
                model=model_name,
                judge=judge,
                rubric_grader=rubric_grader,
                rubrics=scenario_rubrics or None,
                verbose=args.verbose,
            )
            run.results.append(result)

    run.finish()

    # Save results
    output_path = run.save(args.output_dir)
    print_summary(run, verbose=args.verbose)
    print(f"\nResults saved to: {output_path}")
