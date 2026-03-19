#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
BACKEND_DIR="$ROOT_DIR/ai_architect_backend"

usage() {
  echo "Usage: ./run_ai_architect.sh [all|smoke|serve|eval|dryrun]"
  echo "  all   : run Mongo smoke test, then start API server"
  echo "  smoke : run Mongo smoke test only"
  echo "  serve : start API server only"
  echo "  eval  : run planner evaluation cases"
  echo "  dryrun: run planner -> query -> insight on one question"
}

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Error: Python interpreter not found at $PYTHON_BIN"
  exit 1
fi

if [[ ! -d "$BACKEND_DIR" ]]; then
  echo "Error: Backend directory not found at $BACKEND_DIR"
  exit 1
fi

MODE="${1:-all}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8090}"

run_smoke() {
  echo "Running MongoDB smoke test..."
  cd "$BACKEND_DIR"
  "$PYTHON_BIN" scripts/mongodb_smoke_test.py
}

run_server() {
  if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    local health
    health="$(curl -sS "http://${HOST}:${PORT}/health" 2>/dev/null || true)"
    if [[ "$health" == *'"status":"ok"'* && "$health" == *'"app":"materials-copilot-ai-architect"'* ]]; then
      echo "API already running on ${HOST}:${PORT}."
      echo "$health"
      return 0
    fi

    echo "Error: ${HOST}:${PORT} is already in use by another process."
    echo "Tip: either stop the old process or run with a different port, e.g. PORT=8100 ./run_ai_architect.sh serve"
    return 1
  fi

  echo "Starting FastAPI server on ${HOST}:${PORT}..."
  cd "$ROOT_DIR"
  exec "$PYTHON_BIN" -m uvicorn --app-dir "$BACKEND_DIR" app.main:app --host "$HOST" --port "$PORT" --reload
}

run_eval() {
  echo "Running planner evaluation harness..."
  cd "$BACKEND_DIR"
  # Evaluate deterministic planner logic by default to keep this command fast.
  PLANNER_MODE="${EVAL_PLANNER_MODE:-mock}" "$PYTHON_BIN" scripts/planner_eval.py --strict
}

run_dryrun() {
  local question
  question="${QUESTION:-Compare Machine A vs Machine B over the last 7 days and check anomalies}"
  echo "Running end-to-end dry run..."
  cd "$BACKEND_DIR"
  "$PYTHON_BIN" scripts/e2e_dry_run.py --question "$question"
}

case "$MODE" in
  all)
    run_smoke
    run_server
    ;;
  smoke)
    run_smoke
    ;;
  serve)
    run_server
    ;;
  eval)
    run_eval
    ;;
  dryrun)
    run_dryrun
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Unknown mode: $MODE"
    usage
    exit 1
    ;;
esac
