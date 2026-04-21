#!/usr/bin/env python3
"""DFlash MLX Benchmark — uses dflash_mlx Metal kernels for Apple Silicon"""
import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(__file__))

PROMPTS = [
    "Explain what this Python function does and suggest improvements:\n\ndef fib(n):\n    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)\n",
    "Write a Rust function that finds the two numbers in a slice that add up to a target value. Include tests.",
    "Review this SQL query for performance issues and suggest indexes:\n\nSELECT u.name, COUNT(o.id) as order_count, SUM(o.total) as total_spent\nFROM users u\nLEFT JOIN orders o ON u.id = o.user_id\nWHERE o.created_at > DATE_SUB(NOW(), INTERVAL 30 DAY)\nGROUP BY u.name\nORDER BY total_spent DESC;",
]


def run_baseline(model_ref, prompt, max_tokens=512):
    """Baseline inference using dflash_mlx's optimized runtime."""
    from dflash_mlx.runtime import load_target_bundle, stream_baseline_generate

    model, tokenizer, _ = load_target_bundle(model_ref, lazy=True)

    t0 = time.perf_counter()
    summary = None
    for event in stream_baseline_generate(
        target_model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_tokens,
        use_chat_template=True,
    ):
        if event.get("event") == "summary":
            summary = event

    elapsed = time.perf_counter() - t0
    if summary:
        gen_tokens = int(summary.get("generation_tokens", 0))
        elapsed_us = float(summary.get("elapsed_us", 0))
        gen_us = elapsed_us - float(summary.get("prefill_us", 0))
        tps = gen_tokens / (gen_us / 1e6) if gen_us > 0 else 0
        return {"tokens": gen_tokens, "elapsed_s": elapsed, "gen_s": gen_us / 1e6, "tok_per_sec": tps}
    return {"tokens": 0, "elapsed_s": elapsed, "gen_s": 0, "tok_per_sec": 0}


def run_dflash(model_ref, draft_ref, prompt, max_tokens=512, block_size=16):
    """DFlash speculative decoding with Metal verify kernels."""
    from dflash_mlx.runtime import load_target_bundle, load_draft_bundle, stream_dflash_generate

    model, tokenizer, _ = load_target_bundle(model_ref, lazy=True)
    draft_model, _ = load_draft_bundle(draft_ref, lazy=True)

    t0 = time.perf_counter()
    summary = None
    for event in stream_dflash_generate(
        target_model=model,
        tokenizer=tokenizer,
        draft_model=draft_model,
        prompt=prompt,
        max_new_tokens=max_tokens,
        use_chat_template=True,
        block_tokens=block_size,
    ):
        if event.get("event") == "summary":
            summary = event

    elapsed = time.perf_counter() - t0
    if summary:
        gen_tokens = int(summary.get("generation_tokens", 0))
        elapsed_us = float(summary.get("elapsed_us", 0))
        gen_us = elapsed_us - float(summary.get("prefill_us", 0))
        tps = gen_tokens / (gen_us / 1e6) if gen_us > 0 else 0
        acceptance = float(summary.get("acceptance_ratio", 0)) * 100
        return {"tokens": gen_tokens, "elapsed_s": elapsed, "gen_s": gen_us / 1e6, "tok_per_sec": tps, "acceptance_pct": acceptance}
    return {"tokens": 0, "elapsed_s": elapsed, "gen_s": 0, "tok_per_sec": 0, "acceptance_pct": 0}


def main():
    parser = argparse.ArgumentParser(description="DFlash MLX Benchmark (Metal kernels)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft", default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=16)
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"DFlash MLX Benchmark (Metal Verify Kernels)")
    print(f"Model: {args.model}")
    print(f"Draft: {args.draft or '(baseline only)'}")
    print(f"Max tokens: {args.max_tokens}, Runs: {args.runs}")
    print(f"{'='*60}\n")

    # --- Baseline ---
    print("--- BASELINE (mlx_lm, no speculation) ---")
    baseline_results = []
    for i, prompt in enumerate(PROMPTS[:args.runs]):
        print(f"  Run {i+1}...", end=" ", flush=True)
        result = run_baseline(args.model, prompt, args.max_tokens)
        baseline_results.append(result)
        print(f"{result['tok_per_sec']:.1f} tok/s ({result['tokens']} tokens in {result['gen_s']:.2f}s)")

    avg_baseline_tps = sum(r['tok_per_sec'] for r in baseline_results) / len(baseline_results)
    avg_baseline_time = sum(r['gen_s'] for r in baseline_results) / len(baseline_results)
    print(f"  >> Avg: {avg_baseline_tps:.1f} tok/s, {avg_baseline_time:.2f}s\n")

    # --- DFlash ---
    if args.draft:
        print(f"--- DFLASH (draft={args.draft}, block={args.block_size}) ---")
        dflash_results = []
        for i, prompt in enumerate(PROMPTS[:args.runs]):
            print(f"  Run {i+1}...", end=" ", flush=True)
            try:
                result = run_dflash(args.model, args.draft, prompt, args.max_tokens, args.block_size)
                dflash_results.append(result)
                acc = result.get('acceptance_pct', 0)
                print(f"{result['tok_per_sec']:.1f} tok/s ({result['tokens']} tokens in {result['gen_s']:.2f}s, {acc:.1f}% accepted)")
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
                dflash_results.append({"tokens": 0, "elapsed_s": 0, "gen_s": 0, "tok_per_sec": 0, "acceptance_pct": 0})

        if dflash_results:
            valid = [r for r in dflash_results if r['tok_per_sec'] > 0]
            if valid:
                avg_dflash_tps = sum(r['tok_per_sec'] for r in valid) / len(valid)
                avg_dflash_time = sum(r['gen_s'] for r in valid) / len(valid)
                avg_acc = sum(r.get('acceptance_pct', 0) for r in valid) / len(valid)
                speedup = avg_dflash_tps / avg_baseline_tps if avg_baseline_tps > 0 else 0
                print(f"  >> Avg: {avg_dflash_tps:.1f} tok/s, {avg_dflash_time:.2f}s, {avg_acc:.1f}% accepted")
                print(f"  >> Speedup: {speedup:.2f}x\n")

    # --- Summary ---
    print(f"{'='*60}")
    print(f"SUMMARY")
    print(f"  Baseline: {avg_baseline_tps:.1f} tok/s")
    if args.draft and dflash_results:
        valid = [r for r in dflash_results if r['tok_per_sec'] > 0]
        if valid:
            avg_dflash_tps = sum(r['tok_per_sec'] for r in valid) / len(valid)
            print(f"  DFlash:   {avg_dflash_tps:.1f} tok/s ({avg_dflash_tps/avg_baseline_tps:.2f}x)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
