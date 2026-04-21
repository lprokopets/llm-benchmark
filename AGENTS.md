# AGENTS.md

## Running Benchmarks

```bash
./bench                    # Interactive menu
./bench models             # List models
./bench prompts            # List prompts
./bench test -m 1,2 -p all    # Test specific models
./bench test -m 8,9,12 -p all  # Test new local models
python -m llmbench models  # Alternative via Python
```

Local models are loaded **one at a time** by the test runner — no need to start servers manually for benchmarking.

## Local Models

**Servers (for manual use):**
```bash
bash serve.sh start                # Start all
bash serve.sh start primary        # Qwen3.5-35B-A3B (port 11435, llama.cpp)
bash serve.sh start secondary      # Qwen3.5-27B TriAttention (port 8091, MLX)
bash serve.sh start ternary        # Ternary Bonsai 8B (port 8092, MLX)
bash serve.sh start supergemma     # SuperGemma4 26B (port 8093, mlx-vlm)
bash serve.sh start gemma4         # Gemma 4 26B A4B IT (port 11436, llama.cpp)
bash serve.sh start qwen3-tiny     # Qwen3 0.6B (port 11437, llama.cpp)
bash serve.sh start reap           # Gemma 4 19B REAP (port 8094, mlx-vlm)
bash serve.sh start qwopus         # Qwopus 3.5-27B-v3 (port 8095, MLX)
bash serve.sh start qwen36         # Qwen3.6-35B-A3B (port 8098, MLX)
bash serve.sh test                 # Connectivity test
bash serve.sh stop                 # Stop all
```

### Model Weights

Local models store weights under `models/`:
- `models/Qwen3.5-35B-A3B-Q4_K_M.gguf` — Primary (21 GB, GGUF, llama.cpp)
- `models/Qwen3.5-27B-Q4_K_M.gguf` — Secondary (GGUF, llama.cpp, also MLX)
- `models/Qwen3-0.6B-Q4_K_M.gguf` — Tiny baseline (378 MB, GGUF)
- `models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf` — Gemma 4 26B (16 GB, GGUF)
- `models/ternary-bonsai-8b/` — Ternary (2 GB, MLX safetensors)
- `models/supergemma4-26b-mlx-4bit/` — SuperGemma (14.5 GB, MLX 4-bit)
- `models/supergemma4-26b-abliterated-multimodal/` — SuperGemma abliterated (15 GB)
- `models/gemma-4-19b-reap-mlx-4bit/` — REAP pruned (12.6 GB, MLX 4-bit)
- `models/qwopus35-27b-v3-mlx-4bit/` — Qwopus (16.5 GB, MLX 4-bit)
- `models/qwen36-35b-a3b-mlx-4bit/` — Qwen3.6 (19 GB, MLX 4-bit)
- `models/gemma-4-31b-paro/` — Gemma 4 PARO (18 GB, disabled — tokenizer issue)
- `models/qwen35-35b-a3b-paro/` — Qwen PARO (19.2 GB, disabled — load timeout)

### Prerequisites

- **llama.cpp**: `brew install llama.cpp` (Primary, Gemma4, Qwen3-tiny)
- **Python venv**: `.venv/` with `mlx-lm>=0.31`, `mlx-vlm>=0.4`, `huggingface_hub`
- Cloud models need `.env` with API keys (copy from `.env.example`)

## Project Structure

- `llmbench/` - Python package (providers, tester, scorer)
- `config.yaml` - Model definitions (PARO models commented out)
- `serve.sh` - Master server manager (all 10 active models)
- `serve-primary.sh` - llama.cpp server (GGUF)
- `serve-secondary.py` - MLX + TriAttention server
- `serve-ternary.py` - MLX server (Ternary Bonsai)
- `serve-supergemma.py` - mlx-vlm server (SuperGemma4, REAP)
- `serve-qwopus.py` - MLX server (Qwopus, Qwen3.6)
- `triattention_mlx.py` - TriAttention KV compression module
- `models/` - Model weight files

## Test Prompts (21 total)

**Original (8):** `reasoning`, `strawberry`, `widgets`, `coding`, `haiku`, `bat_ball`, `truth_liar`, `sql_injection`

**Coding challenges (13):** `coding_parser`, `coding_concurrent`, `coding_refactor`, `coding_lru_cache`, `coding_min_heap`, `coding_async_debug`, `coding_producer_consumer`, `coding_api_design`, `coding_middleware`, `coding_regex`, `coding_template_engine`, `coding_binary_search`, `coding_merge_intervals`

## Latest Benchmark Results

| Model | Score | tok/s | Type |
|-------|-------|-------|------|
| kimi-k2.5-or | 200/210 | 10.2 | Cloud (OpenRouter) |
| gemma-4-19b-reap | 209/210 | 55.7 | Local (MLX 4-bit, pruned) |
| qwen36-35b-a3b | 201/210 | 70.5 | Local (MLX 4-bit) |
| ternary-bonsai-8b | 192/210 | 94.5 | Local (MLX 1.58-bit) |
| supergemma4-26b | 207/210 | 52.4 | Local (MLX 4-bit) |
| qwen35-35b-a3b | 195/210 | 50.7 | Local (llama.cpp) |
| qwopus35-27b-v3 | 149/210 | varies | Local (MLX 4-bit) |
| gemma-4-26b-a4b-it | 188/210 | 41.4 | Local (llama.cpp) |
| glm-5.1 | 141/210 | 25.5 | Cloud (thinking, exhausts tokens) |
| glm-4.7-flash | 132/210 | 14.6 | Cloud (thinking, exhausts tokens) |

## Known Issues

- **GLM thinking models**: Use all `max_tokens` on reasoning, producing 0 visible output on complex prompts. Known limitation.
- **PARO models**: Disabled — Gemma 4 PARO outputs pad tokens (tokenizer incompatibility), Qwen PARO hangs on load.
- **Verbose models**: Qwen3.6, REAP hit `max_tokens: 16384` on simple prompts like `haiku`, wasting ~240s.

## Adding Models

Edit `config.yaml`. Each entry needs: `name`, `provider`, `model_id` (cloud) or `port` (local), `type`.
Add the serve target name to `PORT_TO_SERVE_TARGET` in `llmbench/tester.py` and add `start_<name>()` to `serve.sh`.
