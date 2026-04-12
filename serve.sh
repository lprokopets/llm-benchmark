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
mkdir -p "$LOG_DIR"

PRIMARY_PORT=11435
SECONDARY_PORT=8091

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

    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "  Primary API:   http://localhost:$PRIMARY_PORT/v1/chat/completions"
    echo "  Secondary API: http://localhost:$SECONDARY_PORT/v1/chat/completions"
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
        else
            start_primary
            start_secondary
        fi
        show_status
        ;;
    stop)
        if [[ "${2:-}" == "primary" ]]; then
            stop_server "Primary" "$PRIMARY_PORT"
        elif [[ "${2:-}" == "secondary" ]]; then
            stop_server "Secondary" "$SECONDARY_PORT"
        else
            stop_server "Primary" "$PRIMARY_PORT"
            stop_server "Secondary" "$SECONDARY_PORT"
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
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|test|logs} [primary|secondary]"
        echo ""
        echo "Commands:"
        echo "  start [primary|secondary]  Start server(s) (default: both)"
        echo "  stop  [primary|secondary]  Stop server(s) (default: both)"
        echo "  restart [primary|secondary] Restart server(s)"
        echo "  status                     Show server status"
        echo "  test                       Test both endpoints"
        echo "  logs  [primary|secondary]  Show recent logs"
        echo ""
        echo "Environment:"
        echo "  TRIATTN_KV_BUDGET=N        KV cache budget for secondary (default: 512)"
        exit 1
        ;;
esac
