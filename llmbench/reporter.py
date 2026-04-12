"""Report generation — quality + latency scorecard."""

import json
from pathlib import Path
from typing import Dict, List, Optional

from .prompts import PROMPTS
from .tester import TestResult


def load_all_results(results_dir: str) -> Dict[str, Dict[str, dict]]:
    """
    Load results from all quality_* runs, keyed by model name.
    Returns {model_name: {prompt_name: {score, elapsed, tok_per_sec, ...}}}
    Uses the LATEST score for each model+prompt combination.
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return {}

    all_results: Dict[str, Dict[str, dict]] = {}

    for run_dir in sorted(results_dir.glob("quality_*")):
        for model_dir in run_dir.iterdir():
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name
            if model_name not in all_results:
                all_results[model_name] = {}

            for json_file in model_dir.glob("*.json"):
                try:
                    data = json.loads(json_file.read_text())
                    prompt_name = data.get("prompt", json_file.stem)
                    all_results[model_name][prompt_name] = data
                except Exception:
                    pass

    return all_results


def generate_running_scorecard(results_dir: str) -> str:
    """
    Generate a running scorecard across ALL results to date.
    Each model gets its best/latest scores from any run.
    """
    all_data = load_all_results(results_dir)
    if not all_data:
        return "No results found."

    prompt_names = [p.name for p in PROMPTS]
    model_names = sorted(all_data.keys())

    lines = []
    lines.append("# Running Scorecard (All Results to Date)")
    lines.append("")
    lines.append("Scores are from the most recent run for each model+prompt combination.")
    lines.append("")

    # ── Quality Table ──
    lines.append("## Quality Scores (0-10)")
    lines.append("")

    header = "| Test | " + " | ".join(model_names) + " |"
    sep = "|" + "------|" * (len(model_names) + 1)
    lines.append(header)
    lines.append(sep)

    for pname in prompt_names:
        row = f"| {pname} |"
        for mname in model_names:
            entry = all_data.get(mname, {}).get(pname)
            score = entry.get("score", "-") if entry else "-"
            if isinstance(score, int):
                if score >= 9:
                    row += f" **{score}** |"
                elif score <= 3:
                    row += f" *{score}* |"
                else:
                    row += f" {score} |"
            else:
                row += " - |"
        lines.append(row)

    total_row = "| **Total** |"
    for mname in model_names:
        scores = [all_data[mname].get(p, {}).get("score", 0) for p in prompt_names]
        total = sum(s for s in scores if isinstance(s, int))
        max_total = len(prompt_names) * 10
        total_row += f" **{total}/{max_total}** |"
    lines.append(total_row)
    lines.append("")

    # ── Speed Table ──
    lines.append("## Speed & Latency")
    lines.append("")
    lines.append("| Model | Avg tok/s | Avg Latency (s) | Total Tokens | Tests Run |")
    lines.append("|-------|-----------|-----------------|--------------|-----------|")

    for mname in model_names:
        model_data = all_data[mname]
        valid = [d for d in model_data.values() if d.get("elapsed", 0) > 0 and not d.get("error")]
        if valid:
            avg_tps = sum(d.get("tok_per_sec", 0) for d in valid) / len(valid)
            avg_lat = sum(d.get("elapsed", 0) for d in valid) / len(valid)
            total_tok = sum(d.get("output_tokens", 0) for d in valid)
            lines.append(f"| {mname} | {avg_tps:.1f} | {avg_lat:.1f} | {total_tok} | {len(valid)}/{len(prompt_names)} |")
        else:
            lines.append(f"| {mname} | - | - | - | 0/{len(prompt_names)} |")
    lines.append("")

    # ── Per-Model Detail ──
    lines.append("## Per-Model Detail")
    lines.append("")
    for mname in model_names:
        model_data = all_data[mname]
        scores = {p: model_data.get(p, {}).get("score", "-") for p in prompt_names}
        valid_scores = [s for s in scores.values() if isinstance(s, int)]
        total = sum(valid_scores) if valid_scores else 0
        lines.append(f"### {mname} — {total}/{len(prompt_names)*10}")
        lines.append("")
        for pname in prompt_names:
            entry = model_data.get(pname, {})
            score = entry.get("score", "-")
            elapsed = entry.get("elapsed", 0)
            tps = entry.get("tok_per_sec", 0)
            tok = entry.get("output_tokens", 0)
            error = entry.get("error", "")
            if error:
                lines.append(f"- **{pname}**: *error* — {error[:100]}")
            else:
                lines.append(f"- **{pname}**: {score}/10 ({tok} tok, {elapsed:.1f}s, {tps:.1f} tok/s)")
        lines.append("")

    return "\n".join(lines)


def generate_scorecard(
    results: Dict[str, List[TestResult]],
    model_descriptions: Dict[str, str] = None,
) -> str:
    """Generate a markdown scorecard from test results."""
    model_descriptions = model_descriptions or {}
    prompt_names = [p.name for p in PROMPTS]
    model_names = list(results.keys())

    lines = []
    lines.append("# LLM Quality + Latency Scorecard")
    lines.append(f"**Date:** $DATE")
    lines.append("")

    # ── Quality Scores Table ──
    lines.append("## Quality Scores (0-10 per test)")
    lines.append("")

    header = "| Test | " + " | ".join(model_names) + " |"
    sep = "|" + "------|" * (len(model_names) + 1)
    lines.append(header)
    lines.append(sep)

    for pname in prompt_names:
        row = f"| {pname} |"
        for mname in model_names:
            model_results = results.get(mname, [])
            score = next((r.score for r in model_results if r.prompt_name == pname), None)
            if score is None:
                row += " - |"
            elif score >= 9:
                row += f" **{score}** |"
            elif score <= 3:
                row += f" *{score}* |"
            else:
                row += f" {score} |"
        lines.append(row)

    # Totals
    total_row = "| **Total** |"
    for mname in model_names:
        model_results = results.get(mname, [])
        total = sum(r.score for r in model_results)
        max_total = len(prompt_names) * 10
        total_row += f" **{total}/{max_total}** |"
    lines.append(total_row)
    lines.append("")

    # ── Latency Table ──
    lines.append("## Latency & Speed")
    lines.append("")
    lines.append("| Model | Avg tok/s | Avg Latency (s) | Type |")
    lines.append("|-------|-----------|-----------------|------|")

    for mname in model_names:
        model_results = results.get(mname, [])
        valid = [r for r in model_results if r.response.error is None and r.response.elapsed > 0]
        if valid:
            avg_tps = sum(r.response.tok_per_sec for r in valid) / len(valid)
            avg_lat = sum(r.response.elapsed for r in valid) / len(valid)
            desc = model_descriptions.get(mname, "")
            lines.append(f"| {mname} | {avg_tps:.1f} | {avg_lat:.1f} | {desc} |")
        else:
            lines.append(f"| {mname} | ERROR | ERROR | |")
    lines.append("")

    # ── Full Responses ──
    lines.append("## Full Responses")
    lines.append("")

    for pname in prompt_names:
        lines.append(f"### {pname.upper()}")
        lines.append("")
        for mname in model_names:
            model_results = results.get(mname, [])
            r = next((x for x in model_results if x.prompt_name == pname), None)
            if not r:
                continue

            tok = r.response.output_tokens
            s = r.score
            elapsed = r.response.elapsed
            tps = r.response.tok_per_sec

            lines.append(f"<details><summary><b>{mname}</b> — score: {s}/10, {tok} tokens, {elapsed:.1f}s, {tps:.1f} tok/s</summary>")
            lines.append("")
            lines.append("```")
            content = r.response.content
            if r.response.error:
                content = f"ERROR: {r.response.error}"
            lines.append(content[:1500] + ("..." if len(content) > 1500 else ""))
            lines.append("```")
            lines.append("</details>")
            lines.append("")

    return "\n".join(lines)
