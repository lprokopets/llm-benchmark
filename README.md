# LLM Benchmark Suite

Quality + latency benchmarking for local and cloud LLMs on Apple Silicon.

## Quick Start

```bash
cd ~/dev/llm-benchmark

# 1. Set up API keys (optional — only needed for cloud models)
cp .env.example .env
# Edit .env with your OpenRouter / Moonshot / Hugging Face keys

# 2. Run the CLI (local models are loaded/unloaded automatically)
./bench
```

### First-Time Setup

Local models need their weights downloaded before first use. Set your Hugging Face token in `.env`:

```
HUGGINGFACE_TOKEN=hf_...
```

Then download any missing models:

```bash
# Download SuperGemma4 26B MLX 4-bit (~14.5 GB)
.venv/bin/python -c "
from dotenv import load_dotenv; import os; load_dotenv()
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Jiunsong/supergemma4-26b-abliterated-multimodal-mlx-4bit',
    local_dir='models/supergemma4-26b-mlx-4bit',
    token=os.getenv('HUGGINGFACE_TOKEN'),
)
print('Done.')
"
```

Other local models (Qwen, Bonsai) download automatically on first serve via `huggingface_hub` or are already present in `models/`.

## CLI Commands

```bash
# Interactive mode (menu-driven)
./bench

# List models and their status
./bench models

# List available test prompts (21 total)
./bench prompts

# Run tests — specific models and prompts
./bench test -m 1,3,5 -p reasoning,strawberry,coding

# Run all prompts on all available models
./bench test -m all -p all

# Run just the local models
./bench test -m 1,2 -p all

# View last scorecard
./bench results
```

### Model Selection (`-m`)

Use numbers from the `models` command, comma-separated:
- `-m 1` — just model #1
- `-m 1,2` — models 1 and 2
- `-m all` — all models

### Prompt Selection (`-p`)

Use prompt names or `all`:
- `-p reasoning,strawberry,coding` — specific prompts
- `-p all` — all 21 prompts

**Original prompts (8):** `reasoning`, `strawberry`, `widgets`, `coding`, `haiku`, `bat_ball`, `truth_liar`, `sql_injection`

**Coding prompts (13):** `coding_parser`, `coding_concurrent`, `coding_refactor`, `coding_lru_cache`, `coding_min_heap`, `coding_async_debug`, `coding_producer_consumer`, `coding_api_design`, `coding_middleware`, `coding_regex`, `coding_template_engine`, `coding_binary_search`, `coding_merge_intervals`

## Server Management

Local models are loaded **one at a time** by the test runner. For manual use:

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
bash serve.sh stop                 # Stop all
bash serve.sh status               # Show running servers
bash serve.sh test                 # Quick connectivity test
```

| Server | Model | Port | Runtime | Speed | Weight |
|--------|-------|------|---------|-------|--------|
| Primary | Qwen3.5-35B-A3B (21B active/128 experts) | 11435 | llama.cpp | ~50 tok/s | 21 GB |
| Secondary | Qwen3.5-27B + TriAttention | 8091 | MLX | ~24 tok/s | GGUF+MLX |
| Ternary | Ternary Bonsai 8B | 8092 | MLX | ~94 tok/s | 2 GB |
| SuperGemma | SuperGemma4 26B Abliterated | 8093 | mlx-vlm | ~52 tok/s | 14.5 GB |
| Gemma4 | Gemma 4 26B A4B IT | 11436 | llama.cpp | ~41 tok/s | 16 GB |
| Qwen3-Tiny | Qwen3 0.6B baseline | 11437 | llama.cpp | ~196 tok/s | 378 MB |
| REAP | Gemma 4 19B (30% pruned) | 8094 | mlx-vlm | ~55 tok/s | 12.6 GB |
| Qwopus | Qwopus 3.5-27B-v3 | 8095 | MLX | varies | 16.5 GB |
| Qwen3.6 | Qwen3.6-35B-A3B | 8098 | MLX | ~70 tok/s | 19 GB |

### Prerequisites

- **llama.cpp** — install via `brew install llama.cpp` (needed for GGUF models)
- **Python venv** — `.venv/` with `mlx-lm>=0.31`, `mlx-vlm>=0.4`, `huggingface_hub`
- **Model files** — GGUF files in `models/`, MLX safetensors in `models/<name>/`

## Cloud Providers

Cloud models are accessed via OpenRouter and Moonshot APIs. All are OpenAI-compatible.

### Setup

1. Copy the env template: `cp .env.example .env`
2. Add your API keys to `.env`:
   ```
   OPENROUTER_API_KEY=sk-or-...
   MOONSHOT_API_KEY=sk-...
   HUGGINGFACE_TOKEN=hf_...
   ```
3. Verify: `./bench models`

### Available Cloud Models

| Model | Provider | Notes |
|-------|----------|-------|
| DeepSeek V3.2 | OpenRouter | Cheap, strong coding |
| Claude Sonnet 4.6 | OpenRouter | Top-tier coding/agents |
| GPT-5.4 | OpenRouter | Frontier |
| Qwen3 Coder Plus | OpenRouter | 1M context, autonomous coding |
| GLM-5.1 (Z.AI) | OpenRouter | Thinking model, exhausts tokens on complex prompts |
| GLM-4.7 Flash (Z.AI) | OpenRouter | Fast/cheap, thinking model |
| Kimi K2.5 | OpenRouter | Multimodal, visual coding |
| Kimi K2.5 | Moonshot direct | Direct API access |
| Kimi K2 Thinking | Moonshot direct | Deep reasoning |

## Latest Benchmark Results

| Model | Score | tok/s | Type |
|-------|-------|-------|------|
| gemma-4-19b-reap | 209/210 | 55.7 | Local (MLX 4-bit, 30% pruned) |
| supergemma4-26b | 207/210 | 52.4 | Local (MLX 4-bit) |
| kimi-k2.5-or | 200/210 | 10.2 | Cloud (OpenRouter) |
| qwen36-35b-a3b | 201/210 | 70.5 | Local (MLX 4-bit) |
| qwen35-35b-a3b | 195/210 | 50.7 | Local (llama.cpp) |
| ternary-bonsai-8b | 192/210 | 94.5 | Local (MLX 1.58-bit) |
| gemma-4-26b-a4b-it | 188/210 | 41.4 | Local (llama.cpp) |
| qwopus35-27b-v3 | 149/210 | varies | Local (MLX 4-bit) |
| glm-5.1 | 141/210 | 25.5 | Cloud (thinking, exhausts tokens) |
| glm-4.7-flash | 132/210 | 14.6 | Cloud (thinking, exhausts tokens) |

## Known Issues

- **GLM thinking models**: Use all `max_tokens` on reasoning, producing 0 visible output on complex prompts
- **PARO models**: Disabled — Gemma 4 PARO outputs pad tokens (tokenizer incompatibility), Qwen PARO hangs on load
- **Verbose models**: Qwen3.6 and REAP hit `max_tokens: 16384` on simple prompts like `haiku`, wasting ~240s

## Project Structure

```
llm-benchmark/
├── .env                    # API keys (not checked in)
├── .env.example            # API key template
├── config.yaml             # Model + provider definitions
├── llmbench/               # Python package
│   ├── cli.py              # Interactive CLI
│   ├── providers.py        # Local + cloud API abstraction
│   ├── tester.py           # Test runner (sequential local model loading)
│   ├── scorer.py           # Quality scoring functions
│   ├── prompts.py          # Test prompts (21 total)
│   └── reporter.py         # Scorecard generation
├── serve.sh                # Server manager (all 10 active models)
├── serve-primary.sh        # llama.cpp server (GGUF)
├── serve-secondary.py      # MLX + TriAttention server
├── serve-ternary.py        # MLX server (Ternary Bonsai)
├── serve-supergemma.py     - mlx-vlm server (SuperGemma4, REAP)
├── serve-qwopus.py         - MLX server (Qwopus, Qwen3.6)
├── triattention_mlx.py     - TriAttention KV compression module
├── models/                 # Model weights (not checked in)
│   ├── Qwen3.5-35B-A3B-Q4_K_M.gguf
│   ├── gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
│   ├── Qwen3-0.6B-Q4_K_M.gguf
│   ├── supergemma4-26b-mlx-4bit/
│   ├── supergemma4-26b-abliterated-multimodal/
│   ├── ternary-bonsai-8b/
│   ├── gemma-4-19b-reap-mlx-4bit/
│   ├── qwopus35-27b-v3-mlx-4bit/
│   ├── qwen36-35b-a3b-mlx-4bit/
│   └── ...
└── results/                # Benchmark output
    └── quality_YYYYMMDD_HHMMSS/
        ├── scorecard.md
        └── model_name/
            └── prompt.json
```

## Hardware

Mac Studio M1 Ultra, 64 GB — benchmarks were calibrated on this machine. Local models use:
- llama.cpp with Metal GPU (`-ngl 99 -fa on --mlock -ctk q8_0 -ctv q8_0`)
- MLX 0.31 via mlx-lm 0.31 (secondary, ternary, qwopus, qwen36)
- mlx-vlm 0.4 (supergemma, REAP — multimodal Gemma 4)
- TriAttention KV compression (norm-only scoring, kv_budget=512)
- Local models are loaded **one at a time** to fit within 64 GB RAM

## Adding/Removing Models

Edit `config.yaml`. Each model entry:

```yaml
- name: my-model
  provider: openrouter        # must match a provider section
  model_id: "org/model-name"  # API model ID
  description: "Description"
  type: cloud                 # or "local"
```

For local models, also add `port: 11XXX`, add the serve target to `PORT_TO_SERVE_TARGET` in `llmbench/tester.py`, and add `start_<name>()` to `serve.sh`.
