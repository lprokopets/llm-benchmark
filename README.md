# LLM Benchmark Suite

Quality + latency benchmarking for local and cloud LLMs on Apple Silicon.

## Quick Start

```bash
cd ~/dev/llm-benchmark

# 1. Set up API keys (optional — only needed for cloud models)
cp .env.example .env
# Edit .env with your OpenRouter / Moonshot keys

# 2. Start local servers
bash serve.sh start

# 3. Run the CLI
./bench
```

## CLI Commands

```bash
# Interactive mode (menu-driven)
./bench

# List models and their status
./bench models

# List available test prompts
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
- `-p all` — all 8 prompts

Available prompts: `reasoning`, `strawberry`, `widgets`, `coding`, `haiku`, `bat_ball`, `truth_liar`, `sql_injection`

## Server Management

```bash
bash serve.sh start                # Start both local servers
bash serve.sh start primary        # Start Qwen3.5-35B-A3B only (port 11435)
bash serve.sh start secondary      # Start Qwen3.5-27B + TriAttention only (port 8091)
bash serve.sh stop                 # Stop all
bash serve.sh status               # Show running servers
bash serve.sh test                 # Quick connectivity test
bash serve.sh logs                 # Show server logs
```

| Server | Model | Port | Speed | Best For |
|--------|-------|------|-------|----------|
| Primary | Qwen3.5-35B-A3B (llama.cpp) | 11435 | ~46 tok/s | Everyday use |
| Secondary | Qwen3.5-27B + TriAttention (MLX) | 8091 | ~24 tok/s | Dense model fallback |

## Cloud Providers

Cloud models are accessed via OpenRouter and Moonshot APIs. All are OpenAI-compatible.

### Setup

1. Copy the env template:
   ```bash
   cp .env.example .env
   ```

2. Add your API keys to `.env`:
   ```
   OPENROUTER_API_KEY=sk-or-...
   MOONSHOT_API_KEY=sk-...
   ```

3. Verify they show as READY:
   ```bash
   python -m llmbench models
   ```

### Available Cloud Models

| # | Model | Provider | Notes |
|---|-------|----------|-------|
| 3 | DeepSeek V3.2 | OpenRouter | Cheap, strong coding |
| 4 | Claude Sonnet 4.6 | OpenRouter | Top-tier coding/agents |
| 5 | GPT-5.4 | OpenRouter | Frontier |
| 6 | Qwen3 Coder Plus | OpenRouter | 1M context, autonomous coding |
| 7 | GLM-5.1 (Z.AI) | OpenRouter | Long-horizon agent tasks |
| 8 | Kimi K2.5 | OpenRouter | Multimodal, visual coding |
| 9 | Kimi K2.5 | Moonshot direct | Direct API access |
| 10 | Kimi K2 Thinking | Moonshot direct | Deep reasoning |

### Adding/Removing Models

Edit `config.yaml`. Each model entry:

```yaml
- name: my-model
  provider: openrouter        # must match a provider section
  model_id: "org/model-name"  # API model ID
  description: "Description"
  type: cloud                 # or "local"
```

For local models, also add `port: 11XXX`.

## Project Structure

```
llm-benchmark/
├── .env                    # API keys (not checked in)
├── .env.example            # API key template
├── config.yaml             # Model + provider definitions
├── llmbench/               # Python package
│   ├── cli.py              # Interactive CLI
│   ├── providers.py        # Local + cloud API abstraction
│   ├── tester.py           # Test runner
│   ├── scorer.py           # Quality scoring functions
│   ├── prompts.py          # Test prompts
│   └── reporter.py         # Scorecard generation
├── serve.sh                # Server manager
├── serve-primary.sh        # Primary server (llama.cpp)
├── serve-secondary.py      # Secondary server (MLX + TriAttention)
├── triattention_mlx.py     # TriAttention KV compression
└── results/                # Benchmark output
    └── quality_YYYYMMDD_HHMMSS/
        ├── scorecard.md
        └── model_name/
            └── prompt.json
```

## Hardware

Mac Studio M1 Ultra, 64GB — benchmarks were calibrated on this machine. Local models use:
- llama.cpp v8680 with Metal GPU (`-ngl 99 -fa on --mlock -ctk q8_0 -ctv q8_0`)
- MLX 0.31 via mlx-lm 0.31
- TriAttention KV compression (norm-only scoring, kv_budget=512)
