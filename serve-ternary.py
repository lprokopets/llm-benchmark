#!/usr/bin/env python3
"""
Ternary Bonsai 8B MLX Server
Port: 8092 | OpenAI-compatible API
=================================================================

MLX server for prism-ml/Ternary-Bonsai-8B-mlx-2bit (1.58-bit quantization)

Usage:
    python serve-ternary.py [--port 8092]
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
from typing import Any, Dict, List

import mlx.core as mx
import mlx_lm
from mlx_lm.sample_utils import make_sampler, make_logits_processors

generation_stream = mx.new_stream(mx.default_device())


def generate_completion(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    max_tokens: int = 512,
    temperature: float = 0.3,
) -> Dict[str, Any]:
    """Generate a chat completion response."""
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

    prompt_toks = len(prompt_tokens)

    start = time.time()

    sampler = make_sampler(temp=temperature) if temperature > 0 else None
    logits_processors = make_logits_processors(
        repetition_penalty=1.0,
        repetition_context_size=256,
    )

    full_text = mlx_lm.generate(
        model,
        tokenizer,
        prompt=prompt_array,
        max_tokens=max_tokens,
        sampler=sampler,
        logits_processors=logits_processors,
    )

    token_count = len(tokenizer.encode(full_text))
    elapsed = time.time() - start
    tps = token_count / elapsed if elapsed > 0 else 0

    print(f"[{gen_id}] {token_count} tok in {elapsed:.1f}s = {tps:.1f} tok/s")

    return {
        "id": gen_id,
        "object": "chat.completion",
        "created": created,
        "model": "ternary-bonsai-8b",
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


class MLXHandler(BaseHTTPRequestHandler):
    """OpenAI-compatible HTTP handler."""

    model = None
    tokenizer = None

    def log_message(self, format, *args):
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
            self.send_json({"status": "ok", "model": "ternary-bonsai-8b"})
        elif self.path == "/v1/models":
            self.send_json({
                "object": "list",
                "data": [{
                    "id": "ternary-bonsai-8b",
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
        request_timeout = body.get("timeout", 600)

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
                    )
                except Exception as e:
                    error_holder[0] = e

            gen_thread = threading.Thread(target=_generate, daemon=True)
            gen_thread.start()
            gen_thread.join(timeout=request_timeout)

            if gen_thread.is_alive():
                print(f"WARNING: Generation timed out after {request_timeout}s")
                try:
                    self.send_error_json(f"Generation timeout after {request_timeout}s", 504)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return

            if error_holder[0]:
                raise error_holder[0]

            result = result_holder[0]
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


def main():
    parser = argparse.ArgumentParser(description="Ternary Bonsai MLX Server")
    parser.add_argument("--port", type=int, default=8092, help="Server port (default: 8092)")
    parser.add_argument("--model", default="models/ternary-bonsai-8b", help="Model ID or local path")
    parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model, tokenizer = mlx_lm.load(args.model)
    print(f"Model loaded.")

    MLXHandler.model = model
    MLXHandler.tokenizer = tokenizer

    server = ThreadingHTTPServer((args.host, args.port), MLXHandler)
    print(f"Ternary Bonsai server: http://localhost:{args.port}/v1/chat/completions")
    print(f"Health check:         http://localhost:{args.port}/health")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
