"""Test runner — executes quality + latency tests against models."""

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .prompts import Prompt, PROMPTS, get_prompt
from .scorer import SCORERS
from .providers import ModelConfig, ProviderConfig, ModelResponse, call_model, check_health


@dataclass
class TestResult:
    prompt_name: str
    model_name: str
    score: int
    response: ModelResponse


def run_test(
    model: ModelConfig,
    provider: ProviderConfig,
    prompt: Prompt,
    max_tokens: int = 4096,
    temperature: float = 0.3,
) -> TestResult:
    """Run a single prompt against a single model and score it."""
    messages = [{"role": "user", "content": prompt.text}]
    response = call_model(provider, model, messages, max_tokens, temperature)

    # Score
    scorer = SCORERS.get(prompt.name)
    score = scorer(response.content) if scorer and not response.error else 0

    return TestResult(
        prompt_name=prompt.name,
        model_name=model.name,
        score=score,
        response=response,
    )


def run_tests(
    models: List[ModelConfig],
    providers: Dict[str, ProviderConfig],
    prompt_names: Optional[List[str]] = None,
    max_tokens: int = 4096,
    temperature: float = 0.3,
) -> Dict[str, List[TestResult]]:
    """
    Run selected prompts against selected models.
    Returns {model_name: [TestResult, ...]}.
    """
    prompts = [get_prompt(n) for n in (prompt_names or [p.name for p in PROMPTS])]
    results: Dict[str, List[TestResult]] = {}

    for model in models:
        provider = providers.get(model.provider)
        if not provider:
            print(f"  [skip] {model.name}: unknown provider '{model.provider}'")
            continue

        # Check health for local models
        if model.type == "local":
            healthy = check_health(provider, model)
            if not healthy:
                print(f"  [skip] {model.name}: server not responding")
                continue

        # Check API key for cloud models
        if model.type == "cloud":
            from .providers import resolve_api_key
            key = resolve_api_key(provider)
            if not key or key == "not-needed":
                print(f"  [skip] {model.name}: no API key ({provider.api_key_env})")
                continue

        model_results = []
        print(f"\n  Testing: {model.name} ({model.description})")

        for prompt in prompts:
            print(f"    {prompt.name}...", end=" ", flush=True)
            result = run_test(model, provider, prompt, max_tokens, temperature)
            model_results.append(result)

            if result.response.error:
                print(f"ERROR: {result.response.error[:80]}")
            else:
                tok_note = f" (+{result.response.reasoning_tokens} thinking)" if result.response.reasoning_tokens else ""
                print(f"{result.score}/10  ({result.response.output_tokens} tok{tok_note}, {result.response.elapsed:.1f}s, {result.response.tok_per_sec:.1f} tok/s)")

        results[model.name] = model_results

    return results


def save_results(
    results: Dict[str, List[TestResult]],
    results_dir: str,
):
    """Save raw results and generate scorecard."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(results_dir) / f"quality_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save raw JSON
    raw = {}
    for model_name, model_results in results.items():
        model_dir = run_dir / model_name
        model_dir.mkdir(exist_ok=True)
        raw[model_name] = []
        for r in model_results:
            entry = {
                "prompt": r.prompt_name,
                "score": r.score,
                "content": r.response.content[:3000],
                "reasoning": r.response.reasoning_content[:2000],
                "elapsed": r.response.elapsed,
                "tok_per_sec": r.response.tok_per_sec,
                "output_tokens": r.response.output_tokens,
                "reasoning_tokens": r.response.reasoning_tokens,
                "error": r.response.error,
            }
            raw[model_name].append(entry)
            with open(model_dir / f"{r.prompt_name}.json", "w") as f:
                json.dump(entry, f, indent=2)

    return run_dir
