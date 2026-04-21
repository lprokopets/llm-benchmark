#!/usr/bin/env python3
"""
Secondary LLM Server — Qwen3.5-27B + TriAttention KV Compression
Port: 8091 | OpenAI-compatible API
=================================================================

A lightweight OpenAI-compatible server wrapping mlx-lm with TriAttention
KV cache compression for Apple Silicon.

Usage:
    python serve-secondary.py [--port 8091] [--kv-budget 512] [--model MODEL_ID]

Environment:
    TRIATTN_KV_BUDGET  — KV cache budget per layer (default: 512)
"""

import argparse
import json
import os
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple
import functools

import mlx.core as mx

# Add script directory to path for local imports
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from triattention_mlx import TriAttentionCompressor, TriAttentionConfig

# ───────────────────────────── Generation with TriAttention ──────────────────

generation_stream = mx.new_stream(mx.default_device())


def triattention_generate_step(
    prompt: mx.array,
    model,
    tokenizer,
    *,
    max_tokens: int = 512,
    temperature: float = 0.3,
    kv_budget: int = 512,
    prefill_step_size: int = 2048,
):
    """
    Generator yielding (token_id, text_segment) tuples with TriAttention
    KV cache compression applied during generation.
    """
    from mlx_lm.models.cache import make_prompt_cache

    prompt_cache = make_prompt_cache(model)
    compressor = TriAttentionCompressor(
        TriAttentionConfig(kv_budget=kv_budget)
    )

    # Build sampler
    if temperature > 0:
        def sampler(logits):
            return mx.random.categorical(logits * (1.0 / temperature))
    else:
        def sampler(logits):
            return mx.argmax(logits, axis=-1)

    def _model_call(tokens):
        return model(tokens, cache=prompt_cache)

    def _step(tokens):
        with mx.stream(generation_stream):
            logits = _model_call(tokens[None])
            logits = logits[:, -1, :]
            logprobs = logits - mx.logsumexp(logits, keepdims=True)
            sampled = sampler(logprobs)
            return sampled, logprobs.squeeze(0)

    # Prefill
    with mx.stream(generation_stream):
        total = len(prompt)
        processed = 0
        while total - processed > 1:
            remaining = total - processed - 1
            n = min(prefill_step_size, remaining)
            _model_call(prompt[:n][None])
            mx.eval([c.state for c in prompt_cache])
            processed += n
            prompt = prompt[n:]

        # Mark prefill for TriAttention
        compressor.step(prompt_cache, is_prefill=True)

        y, logprobs = _step(prompt)

    mx.async_eval(y, logprobs)

    detokenizer = tokenizer.detokenizer
    n = 0
    while True:
        if n != max_tokens:
            next_y, next_logprobs = _step(y)
            mx.async_eval(next_y, next_logprobs)

            # TriAttention compression after each decode step
            compressor.step(prompt_cache, is_prefill=False)

        if n == 0:
            mx.eval(y)

        if n == max_tokens:
            break

        token_id = y.item()
        if token_id in tokenizer.eos_token_ids:
            break

        detokenizer.add_token(token_id)
        text = detokenizer.last_segment

        yield token_id, text

        y, logprobs = next_y, next_logprobs
        n += 1


def generate_completion(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    max_tokens: int = 512,
    temperature: float = 0.3,
    kv_budget: int = 512,
) -> Dict[str, Any]:
    """Generate a chat completion response."""
    # Apply chat template
    if hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages
        ) + "\nassistant: "

    prompt_tokens = tokenizer.encode(prompt_text)
    prompt_array = mx.array(prompt_tokens)

    gen_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    full_text = ""
    token_count = 0
    prompt_toks = len(prompt_tokens)

    start = time.time()
    first_token_time = None

    for token_id, text in triattention_generate_step(
        prompt_array, model, tokenizer,
        max_tokens=max_tokens,
        temperature=temperature,
        kv_budget=kv_budget,
    ):
        if first_token_time is None:
            first_token_time = time.time()
        full_text += text
        token_count += 1

    elapsed = time.time() - start
    tps = token_count / elapsed if elapsed > 0 else 0

    print(f"[{gen_id}] {token_count} tok in {elapsed:.1f}s = {tps:.1f} tok/s")

    return {
        "id": gen_id,
        "object": "chat.completion",
        "created": created,
        "model": "qwen35-27b-triattention",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": full_text,
            },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_toks,
            "completion_tokens": token_count,
            "total_tokens": prompt_toks + token_count,
        },
        "_timing": {
            "elapsed_s": round(elapsed, 2),
            "tok_per_sec": round(tps, 1),
        },
    }


# ───────────────────────────── HTTP Server ──────────────────────────────────

class TriAttentionHandler(BaseHTTPRequestHandler):
    """OpenAI-compatible HTTP handler."""

    # Set by main() before server starts
    model = None
    tokenizer = None
    kv_budget = 512

    def log_message(self, format, *args):
        """Quieter logging."""
        pass

    def send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, message, status=400):
        self.send_json({"error": {"message": message, "type": "invalid_request_error"}}, status)

    def do_GET(self):
        if self.path == "/health":
            self.send_json({"status": "ok", "model": "qwen35-27b-triattention"})
        elif self.path == "/v1/models":
            self.send_json({
                "object": "list",
                "data": [{
                    "id": "qwen35-27b-triattention",
                    "object": "model",
                    "owned_by": "local",
                }],
            })
        else:
            self.send_error_json("Not found", 404)

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            self._handle_chat_completions()
        else:
            self.send_error_json("Not found", 404)

    def _handle_chat_completions(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, ValueError) as e:
            self.send_error_json(f"Invalid JSON: {e}")
            return

        messages = body.get("messages")
        if not messages:
            self.send_error_json("messages is required")
            return

        max_tokens = body.get("max_tokens", 512)
        temperature = body.get("temperature", 0.3)
        request_timeout = body.get("timeout", 300)

        # Stream mode — not supported, return full response
        stream = body.get("stream", False)

        try:
            import threading
            result_holder = [None]
            error_holder = [None]

            def _generate():
                try:
                    result_holder[0] = generate_completion(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        kv_budget=self.kv_budget,
                    )
                except Exception as e:
                    error_holder[0] = e

            gen_thread = threading.Thread(target=_generate, daemon=True)
            gen_thread.start()
            gen_thread.join(timeout=request_timeout)

            if gen_thread.is_alive():
                # Generation is still running — client will have timed out
                print(f"WARNING: Generation timed out after {request_timeout}s, abandoning")
                try:
                    self.send_error_json(f"Generation timeout after {request_timeout}s", 504)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return

            if error_holder[0]:
                raise error_holder[0]

            result = result_holder[0]

            # Strip internal timing from response
            timing = result.pop("_timing", None)
            if timing:
                print(f"  -> {timing['tok_per_sec']} tok/s")

            self.send_json(result)

        except (BrokenPipeError, ConnectionResetError):
            print("WARNING: Client disconnected during generation")
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            try:
                self.send_error_json(f"Generation failed: {e}", 500)
            except (BrokenPipeError, ConnectionResetError):
                pass


# ───────────────────────────── Main ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TriAttention MLX Server")
    parser.add_argument("--port", type=int, default=8091, help="Server port (default: 8091)")
    parser.add_argument("--model", default="mlx-community/Qwen3.5-27B-4bit", help="Model ID or path")
    parser.add_argument("--kv-budget", type=int, default=None, help="KV cache budget (default: 512)")
    parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    args = parser.parse_args()

    kv_budget = args.kv_budget or int(os.environ.get("TRIATTN_KV_BUDGET", "512"))

    print(f"Loading model: {args.model}")
    import mlx_lm
    model, tokenizer = mlx_lm.load(args.model)
    print(f"Model loaded.")
    print(f"TriAttention config: kv_budget={kv_budget}")
    print(f"Starting server on {args.host}:{args.port}")
    print()

    # Configure handler
    TriAttentionHandler.model = model
    TriAttentionHandler.tokenizer = tokenizer
    TriAttentionHandler.kv_budget = kv_budget

    server = ThreadingHTTPServer((args.host, args.port), TriAttentionHandler)
    print(f"Secondary server: http://localhost:{args.port}/v1/chat/completions")
    print(f"Health check:     http://localhost:{args.port}/health")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
