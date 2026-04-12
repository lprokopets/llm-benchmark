#!/usr/bin/env bash
# =============================================================================
# Local LLM Benchmark Harness for Mac Studio M1 Ultra 64GB
# Tests: Gemma 4 26B-A4B, Qwen3.5-35B-A3B, Qwen3.5-27B, Qwen3.5-27B-MLX, Gemma 4 31B-MLX
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/models"
RESULTS_DIR="$SCRIPT_DIR/results"
VENV_DIR="$SCRIPT_DIR/.venv"
mkdir -p "$RESULTS_DIR"

# --- Config ---
PORT=11435           # Base port; each candidate offsets by 1
TEMP=0.6
MAX_TOKENS=512
PARALLEL_PROMPTS=3

# 3 standardized prompts for coding/agentic use
PROMPT_1='Explain what this Python function does and suggest improvements:\n\ndef fib(n):\n    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)\n'
PROMPT_2='Write a Rust function that finds the two numbers in a slice that add up to a target value. Include tests.'
PROMPT_3='Review this SQL query for performance issues and suggest indexes:\n\nSELECT u.name, COUNT(o.id) as order_count, SUM(o.total) as total_spent\nFROM users u\nLEFT JOIN orders o ON u.id = o.user_id\nWHERE o.created_at > DATE_SUB(NOW(), INTERVAL 30 DAY)\nGROUP BY u.name\nORDER BY total_spent DESC;'

# --- Helpers ---
log()  { echo "[$(date +%H:%M:%S)] $*"; }
die()  { log "ERROR: $*"; exit 1; }

check_port_free() {
    if lsof -iTCP:"$1" -sTCP:LISTEN &>/dev/null; then
        die "Port $1 is already in use. Stop existing server first."
    fi
}

wait_for_server() {
    local port="$1" max_wait="${2:-120}" elapsed=0
    log "Waiting for server on :${port} ..."
    while ! curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; do
        sleep 2
        elapsed=$((elapsed + 2))
        if (( elapsed >= max_wait )); then
            die "Server on :${port} did not start within ${max_wait}s"
        fi
    done
    log "Server on :${port} is ready (${elapsed}s)"
}

kill_server() {
    local port="$1"
    local pid
    pid=$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)
    if [[ -n "$pid" ]]; then
        log "Stopping server on :${port} (PID ${pid})"
        kill "$pid" 2>/dev/null || true
        sleep 2
        # Force kill if still running
        pid=$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)
        [[ -n "$pid" ]] && kill -9 "$pid" 2>/dev/null || true
        sleep 1
    fi
}

run_single_benchmark() {
    local label="$1" port="$2" prompt="$3" run_num="$4"

    local json_payload
    json_payload=$(cat <<ENDJSON
{
  "messages": [{"role": "user", "content": "${prompt}"}],
  "max_tokens": ${MAX_TOKENS},
  "temperature": ${TEMP},
  "stream": false
}
ENDJSON
    )

    log "  [${label}] Run ${run_num}: sending prompt..."
    local start_ns end_ns ttft_ms total_ms output tok_count
    start_ns=$(date +%s%N)

    # Use curl with write-out to capture timing
    local tmpfile
    tmpfile=$(mktemp)
    local curl_out
    curl_out=$(curl -sf \
        -o "$tmpfile" \
        -w '{"ttfb_ms": %{time_starttransfer}, "total_ms": %{time_total}, "speed_upload": %{speed_upload}, "speed_download": %{speed_download}}' \
        -X POST "http://localhost:${port}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$json_payload" 2>&1) || {
            log "  [${label}] Run ${run_num}: curl FAILED"
            rm -f "$tmpfile"
            return 1
        }

    end_ns=$(date +%s%N)

    # Parse response
    local content usage ttfb total_time
    content=$(python3 -c "
import json, sys
try:
    d = json.load(open('$tmpfile'))
    c = d.get('choices',[{}])[0].get('message',{}).get('content','')
    u = d.get('usage',{})
    print(json.dumps({'content_len': len(c), 'prompt_tokens': u.get('prompt_tokens',0), 'completion_tokens': u.get('completion_tokens',0), 'total_tokens': u.get('total_tokens',0)}))
except Exception as e:
    print(json.dumps({'content_len': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'error': str(e)}))
" 2>/dev/null) || content='{"content_len":0,"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}'

    ttfb=$(echo "$curl_out" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('ttfb_ms',0))")
    total_time=$(echo "$curl_out" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('total_ms',0))")

    local comp_tokens
    comp_tokens=$(echo "$content" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('completion_tokens',0))")

    local tok_per_sec=0
    if [[ -n "$total_time" ]] && [[ "$total_time" != "0" ]] && [[ -n "$comp_tokens" ]] && [[ "$comp_tokens" != "0" ]]; then
        tok_per_sec=$(awk "BEGIN {printf \"%.2f\", $comp_tokens / ($total_time)}" 2>/dev/null || echo "0")
    fi

    rm -f "$tmpfile"

    log "  [${label}] Run ${run_num}: ${comp_tokens} tokens in ${total_time}ms (${tok_per_sec} tok/s, TTFT ${ttfb}ms)"

    # Return CSV line
    echo "${label},${run_num},${comp_tokens},${ttfb},${total_time},${tok_per_sec}"
}

benchmark_candidate() {
    local label="$1" port="$2"
    local results=()
    log "Benchmarking: ${label} on :${port}"

    for run in $(seq 1 $PARALLEL_PROMPTS); do
        local prompt_var="PROMPT_${run}"
        local prompt="${!prompt_var}"
        local result
        result=$(run_single_benchmark "$label" "$port" "$prompt" "$run") || {
            results+=("${label},${run},ERROR,ERROR,ERROR,ERROR")
            continue
        }
        results+=("$result")
        sleep 2  # Brief pause between runs
    done

    printf '%s\n' "${results[@]}"
}

# =============================================================================
# Main
# =============================================================================
CSV_FILE="$RESULTS_DIR/benchmark_$(date +%Y%m%d_%H%M%S).csv"
MD_FILE="$RESULTS_DIR/benchmark_$(date +%Y%m%d_%H%M%S).md"

echo "label,run,completion_tokens,ttft_ms,total_ms,tok_per_sec" > "$CSV_FILE"

log "========================================"
log "Local LLM Benchmark — $(date)"
log "Hardware: Mac Studio M1 Ultra, 64GB"
log "========================================"

# --- Phase 0: Install dependencies ---
log "Checking dependencies..."

if ! command -v llama-server &>/dev/null; then
    die "llama-server not found. Install: brew install llama.cpp"
fi

# Create venv and install mlx-lm (Homebrew Python blocks pip install to system)
if [[ ! -f "$VENV_DIR/bin/python3" ]]; then
    log "Creating Python venv and installing mlx-lm..."
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --upgrade pip 2>/dev/null
    "$VENV_DIR/bin/pip" install mlx-lm huggingface_hub 2>/dev/null || die "Failed to install mlx-lm into venv"
fi
# Export for mlx-lm server usage
export VENV_PYTHON="$VENV_DIR/bin/python3"

# --- Phase 1: Download models ---
log ""
log "=== Phase 1: Model Downloads ==="

# Model 1: Gemma 4 26B-A4B Q4_K_M (already exists)
GEMMA_GGUF="$MODELS_DIR/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
if [[ -f "/Users/lenprokopets/.local/share/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf" ]]; then
    log "Gemma 4 26B-A4B: linking existing file"
    ln -sf "/Users/lenprokopets/.local/share/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf" "$GEMMA_GGUF"
elif [[ ! -f "$GEMMA_GGUF" ]]; then
    log "Gemma 4 26B-A4B: downloading..."
    curl -L -o "$GEMMA_GGUF" \
        "https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
fi

# Model 2: Qwen3.5-35B-A3B Q4_K_M
QWEN35_MOE_GGUF="$MODELS_DIR/Qwen3.5-35B-A3B-Q4_K_M.gguf"
if [[ ! -f "$QWEN35_MOE_GGUF" ]]; then
    log "Qwen3.5-35B-A3B: downloading (~22GB)..."
    curl -L -o "$QWEN35_MOE_GGUF" \
        "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/main/Qwen3.5-35B-A3B-Q4_K_M.gguf"
fi

# Model 3: Qwen3.5-27B Q4_K_M
QWEN35_27B_GGUF="$MODELS_DIR/Qwen3.5-27B-Q4_K_M.gguf"
if [[ ! -f "$QWEN35_27B_GGUF" ]]; then
    log "Qwen3.5-27B: downloading (~17GB)..."
    curl -L -o "$QWEN35_27B_GGUF" \
        "https://huggingface.co/unsloth/Qwen3.5-27B-GGUF/resolve/main/Qwen3.5-27B-Q4_K_M.gguf"
fi

# Model 4: Qwen3.5-27B MLX 4-bit — pre-download so server startup is fast
log "Qwen3.5-27B MLX: pre-downloading (if not cached)..."
"$VENV_PYTHON" -c "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/Qwen3.5-27B-4bit')" || log "WARNING: Qwen3.5-27B MLX download issue (will retry on serve)"

# Model 5: Gemma 4 31B MLX 4-bit (no compatible draft model exists for speculative decoding)
log "Gemma 4 31B MLX 4-bit: pre-downloading (if not cached)..."
"$VENV_PYTHON" -c "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/gemma-4-31b-it-4bit')" || log "WARNING: Gemma 4 31B MLX download issue (will retry on serve)"

log "Model downloads complete."

# --- Phase 2: Run benchmarks ---
log ""
log "=== Phase 2: Benchmarks ==="

declare -a ALL_RESULTS

# ---- Candidate 1: Gemma 4 26B-A4B via llama.cpp ----
CANDIDATE="gemma4-26b-a4b-llamacpp"
PORT1=11435
kill_server "$PORT1"
check_port_free "$PORT1"

log "Starting ${CANDIDATE}..."
llama-server \
    -m "$GEMMA_GGUF" \
    -ngl 99 \
    -c 8192 \
    -b 2048 \
    -ub 512 \
    -t 8 \
    -fa on \
    --mlock \
    -ctk q8_0 -ctv q8_0 \
    --port "$PORT1" \
    --host 127.0.0.1 \
    --metrics \
    2>"$RESULTS_DIR/${CANDIDATE}.log" &

wait_for_server "$PORT1"
RESULTS=()
while IFS= read -r line; do RESULTS+=("$line"); done < <(benchmark_candidate "$CANDIDATE" "$PORT1")
ALL_RESULTS+=("${RESULTS[@]}")
kill_server "$PORT1"

# ---- Candidate 2: Qwen3.5-35B-A3B MoE via llama.cpp ----
CANDIDATE="qwen35-35b-a3b-llamacpp"
PORT2=11436
kill_server "$PORT2"
check_port_free "$PORT2"

log "Starting ${CANDIDATE}..."
llama-server \
    -m "$QWEN35_MOE_GGUF" \
    -ngl 99 \
    -c 8192 \
    -b 2048 \
    -ub 512 \
    -t 8 \
    -fa on \
    --mlock \
    -ctk q8_0 -ctv q8_0 \
    --port "$PORT2" \
    --host 127.0.0.1 \
    --metrics \
    2>"$RESULTS_DIR/${CANDIDATE}.log" &

wait_for_server "$PORT2"
RESULTS=()
while IFS= read -r line; do RESULTS+=("$line"); done < <(benchmark_candidate "$CANDIDATE" "$PORT2")
ALL_RESULTS+=("${RESULTS[@]}")
kill_server "$PORT2"

# ---- Candidate 3: Qwen3.5-27B via llama.cpp ----
CANDIDATE="qwen35-27b-llamacpp"
PORT3=11437
kill_server "$PORT3"
check_port_free "$PORT3"

log "Starting ${CANDIDATE}..."
llama-server \
    -m "$QWEN35_27B_GGUF" \
    -ngl 99 \
    -c 8192 \
    -b 2048 \
    -ub 512 \
    -t 8 \
    -fa on \
    --mlock \
    -ctk q8_0 -ctv q8_0 \
    --port "$PORT3" \
    --host 127.0.0.1 \
    --metrics \
    2>"$RESULTS_DIR/${CANDIDATE}.log" &

wait_for_server "$PORT3"
RESULTS=()
while IFS= read -r line; do RESULTS+=("$line"); done < <(benchmark_candidate "$CANDIDATE" "$PORT3")
ALL_RESULTS+=("${RESULTS[@]}")
kill_server "$PORT3"

# ---- Candidate 4: Qwen3.5-27B via mlx-lm ----
CANDIDATE="qwen35-27b-mlx"
PORT4=11438
kill_server "$PORT4"
check_port_free "$PORT4"

log "Starting ${CANDIDATE} (mlx-lm serve)..."
"$VENV_PYTHON" -m mlx_lm server \
    --model mlx-community/Qwen3.5-27B-4bit \
    --host 127.0.0.1 \
    --port "$PORT4" \
    2>"$RESULTS_DIR/${CANDIDATE}.log" &

wait_for_server "$PORT4" 300
RESULTS=()
while IFS= read -r line; do RESULTS+=("$line"); done < <(benchmark_candidate "$CANDIDATE" "$PORT4")
ALL_RESULTS+=("${RESULTS[@]}")
kill_server "$PORT4"

# ---- Candidate 5: Gemma 4 31B via mlx-lm ----
CANDIDATE="gemma4-31b-mlx"
PORT5=11439
kill_server "$PORT5"
check_port_free "$PORT5"

log "Starting ${CANDIDATE} (mlx-lm serve)..."
"$VENV_PYTHON" -m mlx_lm server \
    --model mlx-community/gemma-4-31b-it-4bit \
    --host 127.0.0.1 \
    --port "$PORT5" \
    2>"$RESULTS_DIR/${CANDIDATE}.log" &

wait_for_server "$PORT5" 300
RESULTS=()
while IFS= read -r line; do RESULTS+=("$line"); done < <(benchmark_candidate "$CANDIDATE" "$PORT5")
ALL_RESULTS+=("${RESULTS[@]}")
kill_server "$PORT5"

# --- Phase 3: Compile results ---
log ""
log "=== Phase 3: Results ==="

# Write CSV
for line in "${ALL_RESULTS[@]}"; do
    echo "$line" >> "$CSV_FILE"
done

# Generate markdown report
python3 << 'PYEOF'
import csv, os, sys
from datetime import datetime

results_dir = os.environ.get("RESULTS_DIR", "results")
csv_files = sorted([f for f in os.listdir(results_dir) if f.endswith(".csv")])
if not csv_files:
    print("No CSV results found")
    sys.exit(1)

csv_path = os.path.join(results_dir, csv_files[-1])
md_path = csv_path.replace(".csv", ".md")

with open(csv_path) as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Group by label
labels = {}
for r in rows:
    lab = r["label"]
    if lab not in labels:
        labels[lab] = []
    labels[lab].append(r)

md = []
md.append(f"# LLM Benchmark Results — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
md.append(f"**Hardware:** Mac Studio M1 Ultra, 64GB")
md.append(f"**Config:** context=8192, max_tokens=512, temp=0.6, flash-attn=on, KV quant=q8_0")
md.append("")

# Per-model table
for label, runs in labels.items():
    md.append(f"## {label}")
    md.append("")
    md.append("| Run | Tokens | TTFT (ms) | Total (ms) | tok/s |")
    md.append("|-----|--------|-----------|------------|-------|")
    for r in runs:
        md.append(f"| {r['run']} | {r['completion_tokens']} | {r['ttft_ms']} | {r['total_ms']} | {r['tok_per_sec']} |")

    # Averages
    try:
        avg_tokens = sum(int(r['completion_tokens']) for r in runs if r['completion_tokens'] != 'ERROR') / max(1, len([r for r in runs if r['completion_tokens'] != 'ERROR']))
        avg_ttft = sum(float(r['ttft_ms']) for r in runs if r['ttft_ms'] != 'ERROR') / max(1, len([r for r in runs if r['ttft_ms'] != 'ERROR']))
        avg_total = sum(float(r['total_ms']) for r in runs if r['total_ms'] != 'ERROR') / max(1, len([r for r in runs if r['total_ms'] != 'ERROR']))
        avg_tps = sum(float(r['tok_per_sec']) for r in runs if r['tok_per_sec'] != 'ERROR') / max(1, len([r for r in runs if r['tok_per_sec'] != 'ERROR']))
        md.append(f"| **Avg** | **{avg_tokens:.0f}** | **{avg_ttft:.0f}** | **{avg_total:.0f}** | **{avg_tps:.1f}** |")
    except (ValueError, ZeroDivisionError):
        pass
    md.append("")

# Summary comparison table
md.append("## Summary Comparison")
md.append("")
md.append("| Model | Avg tok/s | Avg TTFT (ms) | Avg Total (ms) |")
md.append("|-------|-----------|---------------|----------------|")
for label, runs in labels.items():
    valid = [r for r in runs if r['tok_per_sec'] != 'ERROR']
    if valid:
        avg_tps = sum(float(r['tok_per_sec']) for r in valid) / len(valid)
        avg_ttft = sum(float(r['ttft_ms']) for r in valid) / len(valid)
        avg_total = sum(float(r['total_ms']) for r in valid) / len(valid)
        md.append(f"| {label} | {avg_tps:.1f} | {avg_ttft:.0f} | {avg_total:.0f} |")
    else:
        md.append(f"| {label} | ERROR | ERROR | ERROR |")
md.append("")

content = "\n".join(md)
with open(md_path, "w") as f:
    f.write(content)
print(content)
PYEOF

log "========================================"
log "Benchmark complete!"
log "CSV: $CSV_FILE"
log "MD:  $MD_FILE"
log "========================================"
