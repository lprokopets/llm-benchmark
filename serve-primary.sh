#!/usr/bin/env bash
# =============================================================================
# Primary LLM Server — Qwen3.5-35B-A3B via llama.cpp
# Port: 11435 | OpenAI-compatible API
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="$SCRIPT_DIR/models/Qwen3.5-35B-A3B-Q4_K_M.gguf"
PORT=11435
HOST="${LISTEN_HOST:-0.0.0.0}"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# Kill existing server on this port
kill_server() {
    local pid
    pid=$(lsof -tiTCP:$PORT -sTCP:LISTEN 2>/dev/null || true)
    if [[ -n "$pid" ]]; then
        log "Stopping existing server on :$PORT (PID $pid)"
        kill "$pid" 2>/dev/null || true
        sleep 2
    fi
}

wait_ready() {
    local max=120 elapsed=0
    while ! curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; do
        sleep 2; elapsed=$((elapsed+2))
        if (( elapsed >= max )); then
            log "TIMEOUT waiting for server (>${max}s)"
            return 1
        fi
    done
    log "Server ready on :$PORT (${elapsed}s)"
}

case "${1:-start}" in
    start)
        if [[ ! -f "$MODEL" ]]; then
            log "ERROR: Model not found at $MODEL"
            exit 1
        fi

        if lsof -tiTCP:$PORT -sTCP:LISTEN &>/dev/null; then
            log "Server already running on :$PORT"
            exit 0
        fi

        log "Starting Qwen3.5-35B-A3B on :$PORT"
        llama-server \
            -m "$MODEL" \
            -ngl 99 \
            -c 8192 \
            -fa on \
            --mlock \
            -ctk q8_0 -ctv q8_0 \
            -t 8 \
            --port "$PORT" \
            --host "$HOST" \
            >>"$LOG_DIR/primary.log" 2>&1 &

        wait_ready
        log "Primary server: http://localhost:$PORT/v1/chat/completions"
        ;;
    stop)
        kill_server
        ;;
    restart)
        kill_server
        sleep 2
        exec "$0" start
        ;;
    status)
        if lsof -tiTCP:$PORT -sTCP:LISTEN &>/dev/null; then
            pid=$(lsof -tiTCP:$PORT -sTCP:LISTEN 2>/dev/null)
            log "Primary server RUNNING on :$PORT (PID $pid)"
        else
            log "Primary server STOPPED"
        fi
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac
