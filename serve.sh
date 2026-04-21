#!/usr/bin/env bash
# =============================================================================
# LLM Server Manager — Start/stop primary and secondary local models
# =============================================================================
# Primary:   Qwen3.5-35B-A3B via llama.cpp  (port 11435)
# Secondary: Qwen3.5-27B + TriAttention MLX (port 8091)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python3"
LOG_DIR="$SCRIPT_DIR/logs"
LISTEN_HOST="${LISTEN_HOST:-0.0.0.0}"
mkdir -p "$LOG_DIR"

PRIMARY_PORT=11435
SECONDARY_PORT=8091
TERNARY_PORT=8092
SUPERGEMMA_PORT=8093
GEMMA4_PORT=11436
QWEN3_TINY_PORT=11437
REAP_PORT=8094
QWOPUS_PORT=8095
PARO_PORT=8096
QWEN_PARO_PORT=8097
QWEN36_PORT=8098

log() { echo "[$(date +%H:%M:%S)] $*"; }

is_running() {
    local port=$1
    lsof -tiTCP:$port -sTCP:LISTEN 2>/dev/null | head -1
}

start_primary() {
    if is_running "$PRIMARY_PORT" >/dev/null; then
        log "Primary already running on :$PRIMARY_PORT"
        return 0
    fi
    log "Starting primary (Qwen3.5-35B-A3B) on :$PRIMARY_PORT..."
    bash "$SCRIPT_DIR/serve-primary.sh" start
}

start_secondary() {
    if is_running "$SECONDARY_PORT" >/dev/null; then
        log "Secondary already running on :$SECONDARY_PORT"
        return 0
    fi
    log "Starting secondary (Qwen3.5-27B + TriAttention) on :$SECONDARY_PORT..."
    local kv_budget="${TRIATTN_KV_BUDGET:-512}"
    "$VENV_PYTHON" "$SCRIPT_DIR/serve-secondary.py" \
        --host "$LISTEN_HOST" \
        --port "$SECONDARY_PORT" \
        --kv-budget "$kv_budget" \
        >"$LOG_DIR/secondary.log" 2>&1 &

    # Wait for server to be ready
    local elapsed=0 max=180
    while ! curl -sf "http://localhost:$SECONDARY_PORT/health" >/dev/null 2>&1; do
        sleep 2; elapsed=$((elapsed+2))
        if (( elapsed >= max )); then
            log "TIMEOUT waiting for secondary server (>${max}s)"
            log "Check logs: $LOG_DIR/secondary.log"
            return 1
        fi
    done
    log "Secondary server ready (${elapsed}s)"
}

start_ternary() {
    if is_running "$TERNARY_PORT" >/dev/null; then
        log "Ternary already running on :$TERNARY_PORT"
        return 0
    fi
    log "Starting Ternary Bonsai 8B on :$TERNARY_PORT..."
    "$VENV_PYTHON" "$SCRIPT_DIR/serve-ternary.py" \
        --host "$LISTEN_HOST" \
        --port "$TERNARY_PORT" \
        --model "$SCRIPT_DIR/models/ternary-bonsai-8b" \
        >"$LOG_DIR/ternary.log" 2>&1 &

    local elapsed=0 max=180
    while ! curl -sf "http://localhost:$TERNARY_PORT/health" >/dev/null 2>&1; do
        sleep 2; elapsed=$((elapsed+2))
        if (( elapsed >= max )); then
            log "TIMEOUT waiting for ternary server (>${max}s)"
            log "Check logs: $LOG_DIR/ternary.log"
            return 1
        fi
    done
    log "Ternary server ready (${elapsed}s)"
}

start_supergemma() {
    if is_running "$SUPERGEMMA_PORT" >/dev/null; then
        log "SuperGemma already running on :$SUPERGEMMA_PORT"
        return 0
    fi
    log "Starting SuperGemma4 26B on :$SUPERGEMMA_PORT..."
    "$VENV_PYTHON" "$SCRIPT_DIR/serve-supergemma.py" \
        --host "$LISTEN_HOST" \
        --port "$SUPERGEMMA_PORT" \
        --model "$SCRIPT_DIR/models/supergemma4-26b-mlx-4bit" \
        >"$LOG_DIR/supergemma.log" 2>&1 &

    local elapsed=0 max=180
    while ! curl -sf "http://localhost:$SUPERGEMMA_PORT/health" >/dev/null 2>&1; do
        sleep 2; elapsed=$((elapsed+2))
        if (( elapsed >= max )); then
            log "TIMEOUT waiting for supergemma server (>${max}s)"
            log "Check logs: $LOG_DIR/supergemma.log"
            return 1
        fi
    done
    log "SuperGemma server ready (${elapsed}s)"
}

start_gemma4() {
    if is_running "$GEMMA4_PORT" >/dev/null; then
        log "Gemma 4 already running on :$GEMMA4_PORT"
        return 0
    fi
    log "Starting Gemma 4 26B A4B IT on :$GEMMA4_PORT..."
    llama-server \
        -m "$SCRIPT_DIR/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf" \
        -ngl 99 -c 8192 -fa on --mlock \
        -ctk q8_0 -ctv q8_0 -t 8 \
        --port "$GEMMA4_PORT" --host "${LISTEN_HOST:-0.0.0.0}" \
        >>"$LOG_DIR/gemma4.log" 2>&1 &
    wait_for_server "Gemma 4" "$GEMMA4_PORT" 120
}

start_qwen3_tiny() {
    if is_running "$QWEN3_TINY_PORT" >/dev/null; then
        log "Qwen3 0.6B already running on :$QWEN3_TINY_PORT"
        return 0
    fi
    log "Starting Qwen3 0.6B on :$QWEN3_TINY_PORT..."
    llama-server \
        -m "$SCRIPT_DIR/models/Qwen3-0.6B-Q4_K_M.gguf" \
        -ngl 99 -c 4096 -fa on \
        -t 4 \
        --port "$QWEN3_TINY_PORT" --host "${LISTEN_HOST:-0.0.0.0}" \
        >>"$LOG_DIR/qwen3-tiny.log" 2>&1 &
    wait_for_server "Qwen3 0.6B" "$QWEN3_TINY_PORT" 60
}

wait_for_server() {
    local name=$1 port=$2 max=${3:-180}
    local elapsed=0
    while ! curl -sf "http://localhost:$port/health" >/dev/null 2>&1; do
        sleep 2; elapsed=$((elapsed+2))
        if (( elapsed >= max )); then
            log "TIMEOUT waiting for $name server (>${max}s)"
            return 1
        fi
    done
    log "$name server ready (${elapsed}s)"
}

start_reap() {
    if is_running "$REAP_PORT" >/dev/null; then
        log "REAP already running on :$REAP_PORT"
        return 0
    fi
    log "Starting Gemma 4 19B REAP on :$REAP_PORT..."
    "$VENV_PYTHON" "$SCRIPT_DIR/serve-supergemma.py" \
        --host "$LISTEN_HOST" \
        --port "$REAP_PORT" \
        --model "$SCRIPT_DIR/models/gemma-4-19b-reap-mlx-4bit" \
        >"$LOG_DIR/reap.log" 2>&1 &
    wait_for_server "REAP" "$REAP_PORT" 120
}

start_qwopus() {
    if is_running "$QWOPUS_PORT" >/dev/null; then
        log "Qwopus already running on :$QWOPUS_PORT"
        return 0
    fi
    log "Starting Qwopus3.5-27B-v3 on :$QWOPUS_PORT..."
    "$VENV_PYTHON" "$SCRIPT_DIR/serve-qwopus.py" \
        --host "$LISTEN_HOST" \
        --port "$QWOPUS_PORT" \
        --model "$SCRIPT_DIR/models/qwopus35-27b-v3-mlx-4bit" \
        >"$LOG_DIR/qwopus.log" 2>&1 &
    wait_for_server "Qwopus" "$QWOPUS_PORT" 180
}

start_paro() {
    if is_running "$PARO_PORT" >/dev/null; then
        log "PARO already running on :$PARO_PORT"
        return 0
    fi
    log "Starting Gemma 4 31B PARO on :$PARO_PORT..."
    "$VENV_PYTHON" -m paroquant.cli.serve \
        --model "$SCRIPT_DIR/models/gemma-4-31b-paro" \
        --port "$PARO_PORT" \
        >"$LOG_DIR/paro.log" 2>&1 &
    wait_for_server "PARO" "$PARO_PORT" 180
}

start_qwen_paro() {
    if is_running "$QWEN_PARO_PORT" >/dev/null; then
        log "Qwen PARO already running on :$QWEN_PARO_PORT"
        return 0
    fi
    log "Starting Qwen3.5-35B-A3B PARO on :$QWEN_PARO_PORT..."
    "$VENV_PYTHON" -m paroquant.cli.serve \
        --model "$SCRIPT_DIR/models/qwen35-35b-a3b-paro" \
        --port "$QWEN_PARO_PORT" \
        >"$LOG_DIR/qwen-paro.log" 2>&1 &
    wait_for_server "Qwen PARO" "$QWEN_PARO_PORT" 180
}

start_qwen36() {
    if is_running "$QWEN36_PORT" >/dev/null; then
        log "Qwen3.6 already running on :$QWEN36_PORT"
        return 0
    fi
    log "Starting Qwen3.6-35B-A3B on :$QWEN36_PORT..."
    "$VENV_PYTHON" "$SCRIPT_DIR/serve-qwopus.py" \
        --host "$LISTEN_HOST" \
        --port "$QWEN36_PORT" \
        --model "$SCRIPT_DIR/models/qwen36-35b-a3b-mlx-4bit" \
        --model-name "qwen36-35b-a3b" \
        >"$LOG_DIR/qwen36.log" 2>&1 &
    wait_for_server "Qwen3.6" "$QWEN36_PORT" 180
}

start_dflash() {
    log "DFlash removed — speculative decoding was slower on M1 Ultra"
    return 1
}

stop_server() {
    local name=$1 port=$2
    local pid
    pid=$(is_running "$port") || true
    if [[ -n "$pid" ]]; then
        log "Stopping $name (PID $pid, port :$port)..."
        kill "$pid" 2>/dev/null || true
        sleep 2
        # Force kill if still running
        pid=$(is_running "$port") || true
        [[ -n "$pid" ]] && kill -9 "$pid" 2>/dev/null || true
        log "$name stopped"
    else
        log "$name not running"
    fi
}

show_status() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║              Local LLM Server Status                       ║"
    echo "╠══════════════════════════════════════════════════════════════╣"

    local p1 status1
    p1=$(is_running "$PRIMARY_PORT") || true
    if [[ -n "$p1" ]]; then
        status1="RUNNING (PID $p1)"
    else
        status1="STOPPED"
    fi
    printf "║ Primary   :%-5s  %-20s  %-20s ║\n" "$PRIMARY_PORT" "Qwen3.5-35B-A3B" "$status1"

    local p2 status2
    p2=$(is_running "$SECONDARY_PORT") || true
    if [[ -n "$p2" ]]; then
        status2="RUNNING (PID $p2)"
    else
        status2="STOPPED"
    fi
    printf "║ Secondary :%-5s  %-20s  %-20s ║\n" "$SECONDARY_PORT" "Qwen3.5-27B+TriAttn" "$status2"

    local p3 status3
    p3=$(is_running "$TERNARY_PORT") || true
    if [[ -n "$p3" ]]; then
        status3="RUNNING (PID $p3)"
    else
        status3="STOPPED"
    fi
    printf "║ Ternary   :%-5s  %-20s  %-20s ║\n" "$TERNARY_PORT" "Ternary-Bonsai-8B" "$status3"

    local p4 status4
    p4=$(is_running "$SUPERGEMMA_PORT") || true
    if [[ -n "$p4" ]]; then
        status4="RUNNING (PID $p4)"
    else
        status4="STOPPED"
    fi
    printf "║ Gemma4    :%-5s  %-20s  %-20s ║\n" "$SUPERGEMMA_PORT" "SuperGemma4-26B" "$status4"

    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "  Local:       http://localhost:$PRIMARY_PORT/v1/chat/completions"
    echo "  Tailscale:   http://100.92.163.109:$PRIMARY_PORT/v1/chat/completions"
    echo ""
}

test_endpoint() {
    local name=$1 port=$2
    local url="http://localhost:$port/v1/chat/completions"
    log "Testing $name ($url)..."
    local resp
    resp=$(curl -sf "$url" \
        -H "Content-Type: application/json" \
        -d '{"messages":[{"role":"user","content":"Say hello in one word."}],"max_tokens":20,"temperature":0.3}' \
        2>&1) || {
        log "  FAILED: $name not responding"
        return 1
    }
    local content
    content=$(echo "$resp" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'][:100])" 2>/dev/null) || {
        log "  FAILED: could not parse response"
        return 1
    }
    log "  OK: $content"
}

case "${1:-status}" in
    start)
        if [[ "${2:-}" == "primary" ]]; then
            start_primary
        elif [[ "${2:-}" == "secondary" ]]; then
            start_secondary
        elif [[ "${2:-}" == "ternary" ]]; then
            start_ternary
        elif [[ "${2:-}" == "supergemma" ]]; then
            start_supergemma
        elif [[ "${2:-}" == "gemma4" ]]; then
            start_gemma4
        elif [[ "${2:-}" == "qwen3-tiny" ]]; then
            start_qwen3_tiny
        elif [[ "${2:-}" == "reap" ]]; then
            start_reap
        elif [[ "${2:-}" == "qwopus" ]]; then
            start_qwopus
        elif [[ "${2:-}" == "paro" ]]; then
            start_paro
        elif [[ "${2:-}" == "qwen-paro" ]]; then
            start_qwen_paro
        elif [[ "${2:-}" == "qwen36" ]]; then
            start_qwen36
        else
            start_primary
            start_secondary
            start_ternary
            start_supergemma
            start_gemma4
            start_qwen3_tiny
            start_reap
            start_qwopus
            start_paro
            start_qwen_paro
            start_qwen36
        fi
        show_status
        ;;
    stop)
        if [[ "${2:-}" == "primary" ]]; then
            stop_server "Primary" "$PRIMARY_PORT"
        elif [[ "${2:-}" == "secondary" ]]; then
            stop_server "Secondary" "$SECONDARY_PORT"
        elif [[ "${2:-}" == "ternary" ]]; then
            stop_server "Ternary" "$TERNARY_PORT"
        elif [[ "${2:-}" == "supergemma" ]]; then
            stop_server "SuperGemma" "$SUPERGEMMA_PORT"
        elif [[ "${2:-}" == "gemma4" ]]; then
            stop_server "Gemma 4" "$GEMMA4_PORT"
        elif [[ "${2:-}" == "qwen3-tiny" ]]; then
            stop_server "Qwen3 0.6B" "$QWEN3_TINY_PORT"
        elif [[ "${2:-}" == "reap" ]]; then
             stop_server "REAP" "$REAP_PORT"
        elif [[ "${2:-}" == "qwopus" ]]; then
            stop_server "Qwopus" "$QWOPUS_PORT"
        elif [[ "${2:-}" == "paro" ]]; then
            stop_server "PARO" "$PARO_PORT"
        elif [[ "${2:-}" == "qwen-paro" ]]; then
            stop_server "Qwen PARO" "$QWEN_PARO_PORT"
        elif [[ "${2:-}" == "qwen36" ]]; then
            stop_server "Qwen3.6" "$QWEN36_PORT"
        else
            stop_server "Primary" "$PRIMARY_PORT"
            stop_server "Secondary" "$SECONDARY_PORT"
            stop_server "Ternary" "$TERNARY_PORT"
            stop_server "SuperGemma" "$SUPERGEMMA_PORT"
            stop_server "Gemma 4" "$GEMMA4_PORT"
            stop_server "Qwen3 0.6B" "$QWEN3_TINY_PORT"
             stop_server "REAP" "$REAP_PORT"
             stop_server "Qwopus" "$QWOPUS_PORT"
             stop_server "PARO" "$PARO_PORT"
             stop_server "Qwen PARO" "$QWEN_PARO_PORT"
             stop_server "Qwen3.6" "$QWEN36_PORT"
        fi
        ;;
    restart)
        "$0" stop "$2"
        sleep 2
        "$0" start "$2"
        ;;
    status)
        show_status
        ;;
    test)
        test_endpoint "Primary" "$PRIMARY_PORT"
        test_endpoint "Secondary" "$SECONDARY_PORT"
        test_endpoint "Ternary" "$TERNARY_PORT"
        test_endpoint "SuperGemma" "$SUPERGEMMA_PORT"
        test_endpoint "Gemma 4" "$GEMMA4_PORT"
        test_endpoint "Qwen3 0.6B" "$QWEN3_TINY_PORT"
        test_endpoint "REAP" "$REAP_PORT"
        test_endpoint "Qwopus" "$QWOPUS_PORT"
        test_endpoint "PARO" "$PARO_PORT"
        test_endpoint "Qwen PARO" "$QWEN_PARO_PORT"
        test_endpoint "Qwen3.6" "$QWEN36_PORT"
        ;;
    logs)
        local target="${2:-all}"
        if [[ "$target" == "primary" || "$target" == "all" ]]; then
            echo "=== Primary logs ==="
            tail -20 "$LOG_DIR/primary.log" 2>/dev/null || echo "(no logs)"
        fi
        if [[ "$target" == "secondary" || "$target" == "all" ]]; then
            echo ""
            echo "=== Secondary logs ==="
            tail -20 "$LOG_DIR/secondary.log" 2>/dev/null || echo "(no logs)"
        fi
        if [[ "$target" == "ternary" || "$target" == "all" ]]; then
            echo ""
            echo "=== Ternary logs ==="
            tail -20 "$LOG_DIR/ternary.log" 2>/dev/null || echo "(no logs)"
        fi
        if [[ "$target" == "supergemma" || "$target" == "all" ]]; then
            echo ""
            echo "=== SuperGemma logs ==="
            tail -20 "$LOG_DIR/supergemma.log" 2>/dev/null || echo "(no logs)"
        fi
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|test|logs} [primary|secondary|ternary|supergemma]"
        echo ""
        echo "Commands:"
        echo "  start [primary|secondary|ternary|supergemma]  Start server(s) (default: all)"
        echo "  stop  [primary|secondary|ternary|supergemma]  Stop server(s) (default: all)"
        echo "  restart [primary|secondary|ternary|supergemma] Restart server(s)"
        echo "  status                     Show server status"
        echo "  test                       Test all endpoints"
        echo "  logs  [primary|secondary|ternary|supergemma]  Show recent logs"
        echo ""
        echo "Environment:"
        echo "  TRIATTN_KV_BUDGET=N        KV cache budget for secondary (default: 512)"
        exit 1
        ;;
esac
