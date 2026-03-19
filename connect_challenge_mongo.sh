#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
SMOKE_SCRIPT="$ROOT_DIR/ai_architect_backend/scripts/mongodb_smoke_test.py"

IMAGE="${IMAGE:-ghcr.io/svenstamm/txp-mongo:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-txp-database}"
HOST_PORT="${HOST_PORT:-27018}"
HOST_ARCH="$(uname -m)"

if [[ -n "${DOCKER_PLATFORM:-}" ]]; then
  SELECTED_PLATFORM="$DOCKER_PLATFORM"
else
  case "$HOST_ARCH" in
    arm64|aarch64)
      SELECTED_PLATFORM="linux/amd64"
      ;;
    x86_64|amd64)
      SELECTED_PLATFORM="linux/amd64"
      ;;
    *)
      SELECTED_PLATFORM=""
      ;;
  esac
fi

docker_platform_args() {
  if [[ -n "$SELECTED_PLATFORM" ]]; then
    echo "--platform $SELECTED_PLATFORM"
  fi
}

usage() {
  echo "Usage: ./connect_challenge_mongo.sh [login|pull|start|status|smoke|wait|reset|reclaim]"
  echo "  login : authenticate Docker to ghcr.io using GHCR_USER and GHCR_TOKEN"
  echo "  pull  : pull challenge image"
  echo "  start : run container on HOST_PORT (default 27018)"
  echo "  status: show image/container status"
  echo "  smoke : run backend smoke test against HOST_PORT"
  echo "  wait  : monitor restore logs until Mongo is ready"
  echo "  reset : remove challenge container and its data volumes"
  echo "  reclaim: prune unused Docker build cache/images/containers"
  echo
  echo "Environment vars:"
  echo "  GHCR_USER=<your_github_username>"
  echo "  GHCR_TOKEN=<your_pat_with_read:packages>"
  echo "  HOST_PORT=27018"
  echo "  DOCKER_PLATFORM=linux/amd64"
  echo "  WAIT_TIMEOUT_SEC=7200"
  echo "  ALLOW_DESTRUCTIVE_RESET=YES  # required only for reset"
}

require_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "Error: docker command not found"
    exit 1
  fi
  if ! docker info >/dev/null 2>&1; then
    echo "Error: Docker daemon is not running"
    exit 1
  fi
}

do_login() {
  require_docker
  if [[ -z "${GHCR_USER:-}" || -z "${GHCR_TOKEN:-}" ]]; then
    echo "Error: set GHCR_USER and GHCR_TOKEN first"
    exit 1
  fi
  echo "$GHCR_TOKEN" | docker login ghcr.io -u "$GHCR_USER" --password-stdin
}

do_pull() {
  require_docker
  if [[ -n "$SELECTED_PLATFORM" ]]; then
    docker pull --platform "$SELECTED_PLATFORM" "$IMAGE"
  else
    docker pull "$IMAGE"
  fi
}

container_exists() {
  docker ps -a --format '{{.Names}}' | grep -Fx "$CONTAINER_NAME" >/dev/null 2>&1
}

container_running() {
  docker ps --format '{{.Names}}' | grep -Fx "$CONTAINER_NAME" >/dev/null 2>&1
}

do_start() {
  require_docker

  if container_exists; then
    if container_running; then
      echo "Container '$CONTAINER_NAME' is already running."
    else
      echo "Starting existing container '$CONTAINER_NAME'..."
      docker start "$CONTAINER_NAME" >/dev/null
      echo "Container started."
    fi
  else
    if lsof -nP -iTCP:"$HOST_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
      echo "Error: port $HOST_PORT is already in use."
      echo "Tip: run with a different port, e.g. HOST_PORT=27019 ./connect_challenge_mongo.sh start"
      exit 1
    fi
    echo "Creating and starting container '$CONTAINER_NAME' on localhost:$HOST_PORT..."
    if [[ -n "$SELECTED_PLATFORM" ]]; then
      docker run -d --platform "$SELECTED_PLATFORM" -p "${HOST_PORT}:27017" --name "$CONTAINER_NAME" "$IMAGE" >/dev/null
    else
      docker run -d -p "${HOST_PORT}:27017" --name "$CONTAINER_NAME" "$IMAGE" >/dev/null
    fi
    echo "Container created and started."
  fi

  # Keep Mongo available after host restart without re-running restore steps.
  docker update --restart unless-stopped "$CONTAINER_NAME" >/dev/null || true

  echo "Use this backend URI: mongodb://localhost:${HOST_PORT}"
}

do_status() {
  require_docker
  echo "Host arch: $HOST_ARCH"
  echo "Selected platform: ${SELECTED_PLATFORM:-native}"
  echo "Image check:"
  docker images --format '{{.Repository}}:{{.Tag}}' | grep -F "$IMAGE" || echo "  not present"
  echo "Container check:"
  docker ps -a --format '{{.ID}} {{.Image}} {{.Status}} {{.Ports}} {{.Names}}' | grep -F "$CONTAINER_NAME" || echo "  not present"
}

do_smoke() {
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Error: Python interpreter not found at $PYTHON_BIN"
    exit 1
  fi
  if [[ ! -f "$SMOKE_SCRIPT" ]]; then
    echo "Error: smoke script not found at $SMOKE_SCRIPT"
    exit 1
  fi
  "$PYTHON_BIN" "$SMOKE_SCRIPT" --uri "mongodb://localhost:${HOST_PORT}"
}

do_wait() {
  require_docker

  local timeout_sec
  timeout_sec="${WAIT_TIMEOUT_SEC:-7200}"
  local start_ts
  start_ts="$(date +%s)"

  if ! container_exists; then
    echo "Error: container '$CONTAINER_NAME' does not exist. Run start first."
    exit 1
  fi

  echo "Waiting for challenge MongoDB restore to finish..."
  echo "Timeout: ${timeout_sec}s"

  while true; do
    local now_ts elapsed
    now_ts="$(date +%s)"
    elapsed="$((now_ts - start_ts))"

    if (( elapsed > timeout_sec )); then
      echo "Timeout reached after ${elapsed}s."
      echo "Check logs: docker logs --tail 200 $CONTAINER_NAME"
      exit 1
    fi

    local recent_logs
    recent_logs="$(docker logs --tail 80 "$CONTAINER_NAME" 2>&1 || true)"

    if grep -Eq "WiredTiger library panic|WT_PANIC|Fatal assertion|Input/output error|No space left on device" <<<"$recent_logs"; then
      echo "Fatal MongoDB storage error detected in logs."
      echo "Likely cause: insufficient Docker disk space or corrupted partial restore state."
      echo "Next steps:"
      echo "  1) ./connect_challenge_mongo.sh reset"
      echo "  2) ./connect_challenge_mongo.sh reclaim"
      echo "  3) ./connect_challenge_mongo.sh start"
      echo "  4) ./connect_challenge_mongo.sh wait"
      exit 2
    fi

    if grep -q "finished restoring txp_clean.valuecolumns_migrated" <<<"$recent_logs"; then
      echo "Restore completion marker found in logs."
    fi

    local progress_line
    progress_line="$(grep -E "txp_clean\.valuecolumns_migrated" <<<"$recent_logs" | tail -n 1 || true)"
    if [[ -n "$progress_line" ]]; then
      echo "Progress: $progress_line"
    else
      echo "Progress: waiting for restore log output..."
    fi

    if do_smoke >/tmp/challenge_smoke_out.txt 2>/tmp/challenge_smoke_err.txt; then
      echo "MongoDB is ready."
      cat /tmp/challenge_smoke_out.txt
      rm -f /tmp/challenge_smoke_out.txt /tmp/challenge_smoke_err.txt
      return 0
    fi

    echo "Still initializing... retrying in 20s"
    sleep 20
  done
}

do_reset() {
  require_docker

  if [[ "${ALLOW_DESTRUCTIVE_RESET:-}" != "YES" ]]; then
    echo "Refusing destructive reset without explicit confirmation."
    echo "If you really want to delete Mongo container data, run:"
    echo "  ALLOW_DESTRUCTIVE_RESET=YES ./connect_challenge_mongo.sh reset"
    exit 1
  fi

  local volumes=()
  if container_exists; then
    local inspect_volumes
    inspect_volumes="$(docker inspect "$CONTAINER_NAME" --format '{{range .Mounts}}{{if eq .Type "volume"}}{{println .Name .Destination}}{{end}}{{end}}' 2>/dev/null || true)"
    while read -r vol_name vol_dest; do
      if [[ -n "${vol_name:-}" && ("$vol_dest" == "/data/db" || "$vol_dest" == "/data/configdb") ]]; then
        volumes+=("$vol_name")
      fi
    done <<<"$inspect_volumes"

    echo "Removing container '$CONTAINER_NAME'..."
    docker rm -f "$CONTAINER_NAME" >/dev/null || true
  else
    echo "Container '$CONTAINER_NAME' not present; nothing to remove."
  fi

  if (( ${#volumes[@]} > 0 )); then
    echo "Removing challenge data volumes..."
    for vol in "${volumes[@]}"; do
      docker volume rm "$vol" >/dev/null || true
    done
  else
    echo "No challenge data volumes found to remove."
  fi

  echo "Reset completed."
}

do_reclaim() {
  require_docker

  echo "Pruning stopped containers..."
  docker container prune -f >/dev/null || true
  echo "Pruning dangling images..."
  docker image prune -f >/dev/null || true
  echo "Pruning unused build cache..."
  docker builder prune -af >/dev/null || true
  echo "Reclaim completed."
  docker system df
}

MODE="${1:-status}"
case "$MODE" in
  login)
    do_login
    ;;
  pull)
    do_pull
    ;;
  start)
    do_start
    ;;
  status)
    do_status
    ;;
  smoke)
    do_smoke
    ;;
  wait)
    do_wait
    ;;
  reset)
    do_reset
    ;;
  reclaim)
    do_reclaim
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
