#!/usr/bin/env bash
# =============================================================================
# LLM Quality + Speed Scorecard (bash 3.2 compatible)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python3"
PRISM_SERVER="$HOME/.local/share/prism-llama-cpp/build/bin/llama-server"
RESULTS_DIR="$SCRIPT_DIR/results/quality_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# Kill any leftover servers
for p in 11450 11451 11452 11453 11454; do
    lsof -tiTCP:$p -sTCP:LISTEN 2>/dev/null | xargs kill 2>/dev/null || true
done
sleep 2

wait_ready() {
    local port=$1 max=90 elapsed=0
    while ! curl -sf http://localhost:$port/health >/dev/null 2>&1; do
        sleep 2; elapsed=$((elapsed+2))
        (( elapsed >= max )) && { echo "TIMEOUT"; return 1; }
    done
    echo "${elapsed}s"
}

kill_server() {
    local port=$1
    lsof -tiTCP:$port -sTCP:LISTEN 2>/dev/null | xargs kill 2>/dev/null || true
    sleep 2
}

# --- Models: name|port|serve_command ---
MODELS="
bonsai-8b|11450|$PRISM_SERVER -m $HOME/.local/share/models/Bonsai-8B.gguf -ngl 99 -c 4096 --port 11450 --host 127.0.0.1
gemma4-26b-a4b|11451|llama-server -m $HOME/.local/share/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf -ngl 99 -c 8192 -fa on --mlock -ctk q8_0 -ctv q8_0 -t 8 --port 11451 --host 127.0.0.1
qwen35-35b-a3b|11452|llama-server -m $SCRIPT_DIR/models/Qwen3.5-35B-A3B-Q4_K_M.gguf -ngl 99 -c 8192 -fa on --mlock -ctk q8_0 -ctv q8_0 -t 8 --port 11452 --host 127.0.0.1
qwen35-27b-mlx|11453|$VENV_PYTHON -m mlx_lm server --model mlx-community/Qwen3.5-27B-4bit --host 127.0.0.1 --port 11453
gemma4-31b-mlx|11454|$VENV_PYTHON -m mlx_lm server --model mlx-community/gemma-4-31b-it-4bit --host 127.0.0.1 --port 11454
"

# --- Test each model ---
log "========================================"
log "Quality Test — $(date)"
log "========================================"

echo "$MODELS" | while read -r line; do
    [[ -z "$line" ]] && continue
    name=$(echo "$line" | cut -d'|' -f1)
    port=$(echo "$line" | cut -d'|' -f2)
    serve_cmd=$(echo "$line" | cut -d'|' -f3-)

    log ""
    log "=== $name (:$port) ==="

    kill_server "$port"
    log "Starting server..."
    $serve_cmd 2>/dev/null &
    ready=$(wait_ready "$port")
    if [[ "$ready" == "TIMEOUT" ]]; then
        log "FAILED to start $name"
        kill_server "$port"
        continue
    fi
    log "Server ready ($ready)"

    model_dir="$RESULTS_DIR/$name"
    mkdir -p "$model_dir"

    # Run all 5 prompts via a single python script that handles JSON safely
    "$VENV_PYTHON" << PYEOF
import json, urllib.request, sys, os

model_dir = "$model_dir"
port = $port
base_url = f"http://localhost:{port}"

prompts = [
    ("reasoning", "I have a meeting at 3pm. It's currently 2:45pm. The meeting room is a 10-minute walk away, but it's raining heavily outside and I don't have an umbrella. Should I walk to the meeting room, or should I call in remotely? Explain your reasoning briefly."),
    ("strawberry", "How many r's are in the word 'strawberry'? Think step by step and count each one."),
    ("widgets", "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets? Think carefully."),
    ("coding", "Write a Python function that takes a list of integers and returns the length of the longest consecutive elements sequence. For example, input [100, 4, 200, 1, 3, 2] should return 4 because the longest consecutive sequence is [1, 2, 3, 4]. The algorithm must run in O(n) time. Include the full function with type hints and a test case."),
    ("haiku", "Write a haiku about programming. Output ONLY the haiku — no title, no explanation, no additional text."),
]

for pname, prompt in prompts:
    print(f"  Testing: {pname}")
    payload = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4096,
        "temperature": 0.3
    }).encode()

    try:
        req = urllib.request.Request(
            f"{base_url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        import time
        start = time.time()
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read()
        elapsed = time.time() - start

        data = json.loads(body)
        msg = data.get("choices", [{}])[0].get("message", {})
        content = msg.get("content", "") or ""
        # Some models put thinking in reasoning_content field
        thinking = msg.get("reasoning_content", "") or ""
        usage = data.get("usage", {})
        tokens = usage.get("completion_tokens", 0)
        completion_tokens_details = usage.get("completion_tokens_details", {})
        reasoning_tokens = completion_tokens_details.get("reasoning_tokens", 0)
        output_tokens = tokens - reasoning_tokens if reasoning_tokens else tokens
        tps = output_tokens / elapsed if elapsed > 0 and output_tokens > 0 else (tokens / elapsed if elapsed > 0 else 0)

        # Save full response
        with open(os.path.join(model_dir, f"{pname}.json"), "w") as f:
            json.dump({
                "content": content,
                "thinking": thinking[:2000] if thinking else "",
                "usage": usage,
                "elapsed": elapsed,
                "tps": tps,
                "output_tokens": output_tokens,
                "reasoning_tokens": reasoning_tokens,
            }, f, indent=2)

        think_note = f" (+{reasoning_tokens} thinking)" if reasoning_tokens else ""
        print(f"    [{output_tokens} tok{think_note}, {elapsed:.1f}s, {tps:.1f} tok/s]")
        if content:
            print(f"    {content[:200]}")
        elif thinking:
            print(f"    (thinking only, no visible output)")
        else:
            print(f"    (empty response)")
    except Exception as e:
        print(f"    FAILED: {e}")
        with open(os.path.join(model_dir, f"{pname}.json"), "w") as f:
            json.dump({"error": str(e)}, f)
PYEOF

    kill_server "$port"
done

# --- Generate Scorecard ---
log ""
log "Generating scorecard..."

"$VENV_PYTHON" << 'PYEOF'
import json, os, sys

# Find results dir (latest)
base = "/Users/lenprokopets/dev/llm-benchmark/results"
dirs = sorted([d for d in os.listdir(base) if d.startswith("quality_")])
if not dirs:
    print("ERROR: No quality results found"); sys.exit(1)
results_dir = os.path.join(base, dirs[-1])

models = ["bonsai-8b", "gemma4-26b-a4b", "qwen35-35b-a3b", "qwen35-27b-mlx", "gemma4-31b-mlx"]
speed_data = {"bonsai-8b": 81.9, "gemma4-26b-a4b": 50.0, "qwen35-35b-a3b": 52.7, "qwen35-27b-mlx": 21.9, "gemma4-31b-mlx": 16.7}
prompts = ["reasoning", "strawberry", "widgets", "coding", "haiku"]

def load_response(model, prompt):
    path = os.path.join(results_dir, model, f"{prompt}.json")
    try:
        d = json.load(open(path))
        return d.get("content", ""), d.get("usage", {}), d.get("elapsed", 0), d.get("tps", 0)
    except:
        return "(no response)", {}, 0, 0

def score_reasoning(text):
    t = text.lower()
    has_remote = any(k in t for k in ["remote", "call in", "virtual", "video call", "zoom", "teams"])
    acknowledges_rain = any(k in t for k in ["rain", "wet", "umbrella", "shelter", "pouring"])
    score = 0
    if has_remote: score += 4
    if acknowledges_rain: score += 2
    if "walk" in t and has_remote: score += 2  # discusses both options
    if len(text) > 50: score += 1
    if "time" in t or "10-minute" in t or "2:55" in t: score += 1  # considers timing
    return min(score, 10)

def score_strawberry(text):
    import re
    t = text.lower()
    # Must find the final/answer section, not step numbers
    # Look for conclusive statements
    answer_patterns = [
        r'(?:answer|total|there are|result|final)\s*(?:is|:)?\s*\*{0,2}?3\b',
        r'3\s*r[\'"]?s',
        r'three\s*r[\'"]?s',
        r'\b3\b.{0,5}(?:r[\'"]?s|in\s+strawberry)',
    ]
    for pat in answer_patterns:
        if re.search(pat, t):
            # But reject if they also say "2 r's" as final answer nearby
            if re.search(r'(?:final answer|answer.*?:|there are)\s*\*{0,2}2\s*r', t):
                return 0
            return 10
    # Also check: did they explicitly say "2 r's" as their answer?
    if re.search(r'(?:final answer|answer.*?:|there are|so,? there are)\s*\*{0,2}2\b', t):
        return 0
    if "three" in t and ("r" in t) and "strawberry" in t:
        # Check "three" is used as the count, not just the word
        if re.search(r'three\s+r', t):
            return 10
    if any(n in re.findall(r'\b(\d+)\b', text) for n in ["4","5","6"]):
        return 2
    return 0

def score_widgets(text):
    import re
    t = text.lower()
    # Must find "5 minutes" as THE answer, not just quoted from the prompt
    # Look for conclusive answer patterns
    answer_patterns = [
        r'(?:answer|result|take|would take|it would take)\s*\*{0,2}:?\s*\*{0,2}5\s*(?:minute|min\b)',
        r'5\s*minutes?\s*(?:to make|to produce|to complete|\.$|\*\*$)',
    ]
    for pat in answer_patterns:
        if re.search(pat, t):
            return 10
    if "same time" in t and ("5 minute" in t or "original" in t):
        return 10
    # Check for wrong answer "1 minute"
    if re.search(r'(?:final answer|answer|would take|it would take)\s*\*{0,2}:?\s*\*{0,2}1\s*(?:minute|min\b)', t):
        return 0
    return 0

def score_coding(text):
    t = text.lower()
    has_set = "set(" in text
    has_func = "def " in text
    has_while_or_for = "while" in text or "for " in text
    has_consecutive = "consecutive" in t
    has_o_n = "o(n)" in t or "linear" in t or "hash" in t
    has_test = "assert" in t or "test" in t or "print(" in t
    has_type = "-> int" in text or "-> " in text or "List[" in text or ": List" in text

    # Correct logic: uses set, iterates, checks neighbors
    correct_logic = has_set and has_while_or_for and has_consecutive

    score = 0
    if has_func: score += 1
    if has_set: score += 2
    if correct_logic: score += 3
    if has_o_n: score += 1
    if has_test: score += 1
    if has_type: score += 1
    if "100, 4, 200, 1, 3, 2" in text or "[100, 4, 200, 1, 3, 2]" in text: score += 1
    return min(score, 10)

def score_haiku(text):
    lines = [l.strip() for l in text.strip().split('\n') if l.strip() and not l.strip().startswith('#')]
    non_empty = len(lines)
    t = text.lower()

    score = 0
    if non_empty == 3: score += 6
    elif non_empty == 4: score += 3  # maybe title line
    elif non_empty <= 2: score += 1

    # Programming related
    prog_words = ["code", "bug", "function", "loop", "variable", "debug", "compile", "syntax",
                  "program", "keyboard", "screen", "terminal", "git", "stack", "array", "byte",
                  "pixel", "cursor", "logic", "data", "type", "class"]
    if any(w in t for w in prog_words): score += 2

    # No extra text (just the haiku)
    if non_empty == 3 and "here" not in t.split('\n')[0].lower() and "haiku" not in t.split('\n')[0].lower():
        score += 2

    return min(score, 10)

scorers = {
    "reasoning": score_reasoning,
    "strawberry": score_strawberry,
    "widgets": score_widgets,
    "coding": score_coding,
    "haiku": score_haiku,
}

# Score everything
all_scores = {}
all_responses = {}
for model in models:
    model_scores = {}
    model_responses = {}
    for prompt in prompts:
        text, usage, elapsed, tps = load_response(model, prompt)
        s = scorers[prompt](text)
        model_scores[prompt] = s
        model_responses[prompt] = (text, usage, elapsed, tps)
    all_scores[model] = model_scores
    all_responses[model] = model_responses

# Build markdown
md = []
md.append("# LLM Quality + Speed Scorecard")
md.append("**Date:** 2026-04-11 | **Hardware:** Mac Studio M1 Ultra, 64GB")
md.append("")

# Quality table
md.append("## Quality Scores (0-10 per test)")
md.append("")
header = "| Test | " + " | ".join(models) + " |"
md.append(header)
md.append("|" + "------|" * (len(models) + 1))

for prompt in prompts:
    row = f"| {prompt} |"
    for model in models:
        s = all_scores[model][prompt]
        # Highlight 10s and low scores
        if s >= 9: row += f" **{s}** |"
        elif s <= 3: row += f" *{s}* |"
        else: row += f" {s} |"
    md.append(row)

# Totals
total_row = "| **Total** |"
for model in models:
    total = sum(all_scores[model].values())
    total_row += f" **{total}/50** |"
md.append(total_row)
md.append("")

# Combined ranking
md.append("## Combined Ranking (60% Quality, 40% Speed)")
md.append("")
md.append("| Rank | Model | Quality | Speed (tok/s) | Combined |")
md.append("|------|-------|---------|---------------|----------|")

max_s = max(speed_data.values()) or 1
ranked = []
for model in models:
    q = sum(all_scores[model].values())
    spd = speed_data[model]
    q_norm = (q / 50) * 100
    s_norm = (spd / max_s) * 100
    combined = q_norm * 0.6 + s_norm * 0.4
    ranked.append((model, q, spd, combined))

ranked.sort(key=lambda x: x[3], reverse=True)
medals = {0: "🥇", 1: "🥈", 2: "🥉"}

for i, (model, q, spd, cs) in enumerate(ranked):
    medal = medals.get(i, f"{i+1}.")
    md.append(f"| {medal} | {model} | {q}/50 | {spd:.1f} | **{cs:.0f}** |")

md.append("")

# Full responses
md.append("## Full Responses")
md.append("")

for prompt in prompts:
    md.append(f"### {prompt.upper()}")
    md.append("")
    for model in models:
        text, usage, elapsed, tps = all_responses[model][prompt]
        tok = usage.get("completion_tokens", "?")
        s = all_scores[model][prompt]
        md.append(f"<details><summary><b>{model}</b> — score: {s}/10, {tok} tokens, {elapsed:.1f}s, {tps:.1f} tok/s</summary>")
        md.append(f"")
        md.append(f"```")
        md.append(text[:1500] + ("..." if len(text) > 1500 else ""))
        md.append(f"```")
        md.append(f"</details>")
        md.append("")

content = "\n".join(md)
outpath = os.path.join(results_dir, "scorecard.md")
with open(outpath, "w") as f:
    f.write(content)
print(content)
PYEOF

log ""
log "Scorecard written to $RESULTS_DIR/scorecard.md"
