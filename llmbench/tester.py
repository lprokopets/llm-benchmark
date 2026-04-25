"""Test runner — executes quality + latency tests against models."""

import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .prompts import Prompt, PROMPTS, get_prompt
from .scorer import SCORERS
from .providers import ModelConfig, ProviderConfig, ModelResponse, call_model, check_health

PORT_TO_SERVE_TARGET = {
    11435: "primary",
    8091: "secondary",
    8092: "ternary",
    8093: "supergemma",
    11436: "gemma4",
    11437: "qwen3-tiny",
    8094: "reap",
    8095: "qwopus",
    8096: "paro",
    8097: "qwen-paro",
    8098: "qwen36",
}


def _serve_cmd(base_dir: Path, *args: str, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(base_dir / "serve.sh"), *args],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout,
    )


def _start_server(base_dir: Path, model: ModelConfig) -> bool:
    target = PORT_TO_SERVE_TARGET.get(model.port)
    if not target:
        return False
    try:
        r = _serve_cmd(base_dir, "start", target, timeout=600)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"    [timeout] {model.name} did not start in 600s")
        return False


def _stop_server(base_dir: Path, model: ModelConfig) -> None:
    target = PORT_TO_SERVE_TARGET.get(model.port)
    if not target:
        return
    try:
        _serve_cmd(base_dir, "stop", target, timeout=30)
    except subprocess.TimeoutExpired:
        pass


def _stop_all_managed(base_dir: Path) -> None:
    try:
        _serve_cmd(base_dir, "stop", timeout=30)
    except subprocess.TimeoutExpired:
        pass
    time.sleep(2)


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
    timeout: float = 300,
) -> TestResult:
    """Run a single prompt against a single model and score it."""
    messages = [{"role": "user", "content": prompt.text}]
    response = call_model(provider, model, messages, max_tokens, temperature, timeout=timeout)

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
    timeout: float = 300,
    base_dir: Optional[str] = None,
) -> Dict[str, List[TestResult]]:
    """
    Run selected prompts against selected models.
    Local models are loaded one at a time to fit in memory.
    Returns {model_name: [TestResult, ...]}.
    """
    prompts = [get_prompt(n) for n in (prompt_names or [p.name for p in PROMPTS])]
    results: Dict[str, List[TestResult]] = {}

    local_models = [m for m in models if m.type == "local"]
    cloud_models = [m for m in models if m.type != "local"]
    managed_ports = {m.port for m in local_models if m.port in PORT_TO_SERVE_TARGET}
    _base = Path(base_dir) if base_dir else None

    if managed_ports and _base:
        print("  Stopping all local servers for sequential loading...")
        _stop_all_managed(_base)

    for model in cloud_models:
        provider = providers.get(model.provider)
        if not provider:
            print(f"  [skip] {model.name}: unknown provider '{model.provider}'")
            continue
        from .providers import resolve_api_key
        key = resolve_api_key(provider)
        if not key or key == "not-needed":
            print(f"  [skip] {model.name}: no API key ({provider.api_key_env})")
            continue
        results[model.name] = _test_model(model, provider, prompts, max_tokens, temperature, timeout)

    for model in local_models:
        provider = providers.get(model.provider)
        if not provider:
            print(f"  [skip] {model.name}: unknown provider '{model.provider}'")
            continue

        is_managed = model.port in managed_ports
        if is_managed and _base:
            _stop_all_managed(_base)
            print(f"  Starting {model.name}...")
            if not _start_server(_base, model):
                print(f"  [skip] {model.name}: failed to start server")
                continue
        elif model.type == "local":
            healthy = check_health(provider, model)
            if not healthy:
                print(f"  [skip] {model.name}: server not responding")
                continue

        results[model.name] = _test_model(model, provider, prompts, max_tokens, temperature, timeout)

        if is_managed and _base:
            _stop_server(_base, model)

    return results


def _test_model(
    model: ModelConfig,
    provider: ProviderConfig,
    prompts: List[Prompt],
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> List[TestResult]:
    model_results = []
    print(f"\n  Testing: {model.name} ({model.description})")

    for prompt in prompts:
        print(f"    {prompt.name}...", end=" ", flush=True)
        result = run_test(model, provider, prompt, max_tokens, temperature, timeout=timeout)
        model_results.append(result)

        if result.response.error:
            print(f"ERROR: {result.response.error[:80]}")
        else:
            tok_note = f" (+{result.response.reasoning_tokens} thinking)" if result.response.reasoning_tokens else ""
            spec_note = ""
            if result.response.speculation_stats:
                acc = result.response.speculation_stats.get("acceptance_rate")
                if acc:
                    spec_note = f" | spec_acc: {acc:.2f}"
                elif "accepted_tokens" in result.response.speculation_stats:
                    spec_note = f" | spec_acc: {result.response.speculation_stats['accepted_tokens']}"

            # Elapsed time is already wall-clock time from call_model
            print(f"{result.score}/10  ({result.response.output_tokens} tok{tok_note}, wall: {result.response.elapsed:.2f}s, {result.response.tok_per_sec:.1f} tok/s{spec_note})")

    return model_results


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
