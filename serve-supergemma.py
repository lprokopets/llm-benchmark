#!/usr/bin/env python3
"""
SuperGemma4 26B multimodal server (MLX 4-bit)
Port: 8093 | OpenAI-compatible API
"""

import argparse
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List

import mlx.core as mx
import mlx_vlm
from mlx_vlm import apply_chat_template, generate


def generate_completion(
    model,
    processor,
    config,
    messages: List[Dict[str, str]],
    max_tokens: int = 512,
    temperature: float = 0.3,
) -> Dict[str, Any]:
    prompt = apply_chat_template(processor, config, messages)

    gen_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    start = time.time()

    result = generate(
        model,
        processor,
        prompt=prompt,
        image=None,
        max_tokens=max_tokens,
        temperature=temperature,
        verbose=False,
    )

    full_text = result.text
    prompt_toks = result.prompt_tokens
    token_count = result.generation_tokens
    elapsed = time.time() - start
    tps = result.generation_tps

    print(f"[{gen_id}] {token_count} tok in {elapsed:.1f}s = {tps:.1f} tok/s")

    return {
        "id": gen_id,
        "object": "chat.completion",
        "created": created,
        "model": "supergemma4-26b-abliterated-multimodal",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": full_text},
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


class GemmaHandler(BaseHTTPRequestHandler):
    model = None
    processor = None
    config = None

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
            self.send_json({"status": "ok", "model": "supergemma4-26b-abliterated-multimodal"})
        elif self.path == "/v1/models":
            self.send_json({
                "object": "list",
                "data": [{"id": "supergemma4-26b-abliterated-multimodal", "object": "model", "owned_by": "local"}],
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
                        processor=self.processor,
                        config=self.config,
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
                self.send_error_json(f"Generation timeout after {request_timeout}s", 504)
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
    parser = argparse.ArgumentParser(description="SuperGemma4 MLX Server")
    parser.add_argument("--port", type=int, default=8093, help="Server port (default: 8093)")
    parser.add_argument("--model", default="models/supergemma4-26b-mlx-4bit", help="Model path or HF repo ID")
    parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model, processor = mlx_vlm.load(args.model)
    config = model.config
    print("Model loaded.")

    GemmaHandler.model = model
    GemmaHandler.processor = processor
    GemmaHandler.config = config

    server = ThreadingHTTPServer((args.host, args.port), GemmaHandler)
    print(f"SuperGemma server: http://localhost:{args.port}/v1/chat/completions")
    print(f"Health check:      http://localhost:{args.port}/health")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
