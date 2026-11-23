#!/usr/bin/env bash

# Usage:
#   source ./run.sh [IP] [PORT]
# or:
#   ./run.sh [IP] [PORT]

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename "${PROJECT_DIR}")"

IMAGE_NAME="${IMAGE_NAME:-${PROJECT_NAME}:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-${PROJECT_NAME}}"

IP="${1:-0.0.0.0}"
PORT="${2:-30000}"

if [[ -f "${PROJECT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${PROJECT_DIR}/.env"
  set +a
fi

load_example_defaults() {
  local example_file="${PROJECT_DIR}/.env.example"
  [[ -f "${example_file}" ]] || return 0

  while IFS= read -r line; do
    [[ "${line}" =~ ^#[A-Z0-9_]+= ]] || continue

    local kv="${line#\#}"
    local var_name="${kv%%=*}"
    local raw_value="${kv#*=}"

    if [[ -z "${!var_name+x}" || -z "${!var_name}" ]]; then
      local expanded
      expanded=$(eval "echo \"${raw_value}\"")
      export "${var_name}=${expanded}"
    fi
  done < "${example_file}"
}

load_example_defaults

: "${MODEL_PATH:?MODEL_PATH must be set (via .env or .env.example).}"
: "${SERVED_MODEL_NAME:?SERVED_MODEL_NAME must be set (via .env or .env.example).}"
: "${TP_SIZE:?TP_SIZE must be set (via .env or .env.example).}"
: "${EP_SIZE:?EP_SIZE must be set (via .env or .env.example).}"
: "${MEM_FRACTION_STATIC:?MEM_FRACTION_STATIC must be set (via .env or .env.example).}"
: "${HTTP_PORT:=${PORT}}"
: "${SHM_SIZE:?SHM_SIZE must be set (via .env or .env.example).}"
: "${GPU_DEVICE:?GPU_DEVICE must be set (via .env or .env.example).}"
: "${RESTART_POLICY:?RESTART_POLICY must be set (via .env or .env.example).}"
: "${HF_HOME:=${HOME}/.cache/huggingface}"
: "${HF_HUB_CACHE:=${HF_HOME}}"
: "${TRANSFORMERS_CACHE:=${HF_HOME}}"
if [[ -z "${API_KEY:-}" ]]; then
  echo "ERROR: API_KEY is required in .env" >&2
  return 1 2>/dev/null || exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN is not set; model downloads from the Hugging Face Hub may be rate-limited." >&2
fi

mkdir -p "${HF_HOME}"

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

DOCKER_TTY_ARGS=()
if [[ -t 0 && -t 1 ]]; then
  DOCKER_TTY_ARGS=(-it)
fi

DOCKER_RUN_CLEANUP_ARGS=(--rm)
if [[ "${RESTART_POLICY}" != "no" ]]; then
  DOCKER_RUN_CLEANUP_ARGS=(--restart "${RESTART_POLICY}")
fi

DOCKER_OPTIONAL_ENV_ARGS=()
if [[ -n "${CHUNKED_PREFILL_SIZE:-}" ]]; then
  DOCKER_OPTIONAL_ENV_ARGS+=(
    -e CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE}"
  )
fi

echo "Starting container '${CONTAINER_NAME}' with restart policy '${RESTART_POLICY}'"

exec docker run "${DOCKER_RUN_CLEANUP_ARGS[@]}" "${DOCKER_TTY_ARGS[@]}" \
  --name "${CONTAINER_NAME}" \
  --gpus "${GPU_DEVICE}" \
  --shm-size "${SHM_SIZE}" \
  -p "${IP}:${PORT}:${HTTP_PORT}" \
  -e MODEL_PATH="${MODEL_PATH}" \
  -e SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
  -e TP_SIZE="${TP_SIZE}" \
  -e EP_SIZE="${EP_SIZE}" \
  -e MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC}" \
  -e CONTEXT_LENGTH="${CONTEXT_LENGTH:-}" \
  -e PAGE_SIZE="${PAGE_SIZE:-128}" \
  -e ATTENTION_BACKEND="${ATTENTION_BACKEND:-fa4}" \
  -e BF16_GEMM_BACKEND="${BF16_GEMM_BACKEND:-}" \
  -e DTYPE="${DTYPE:-bfloat16}" \
  -e GRAMMAR_BACKEND="${GRAMMAR_BACKEND:-xgrammar}" \
  -e MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-}" \
  -e MAX_QUEUED_REQUESTS="${MAX_QUEUED_REQUESTS:-}" \
  -e KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}" \
  -e ENABLE_MULTIMODAL="${ENABLE_MULTIMODAL:-1}" \
  -e ENABLE_REASONING="${ENABLE_REASONING:-1}" \
  -e ENABLE_TOOLS="${ENABLE_TOOLS:-1}" \
  -e ENABLE_DSPARK="${ENABLE_DSPARK:-0}" \
  -e SPECULATIVE_NUM_DRAFT_TOKENS="${SPECULATIVE_NUM_DRAFT_TOKENS:-8}" \
  -e INKLING2_LOOP_NUM_PASSES="${INKLING2_LOOP_NUM_PASSES:-}" \
  -e INKLING2_LOOP_SHARE_CACHE="${INKLING2_LOOP_SHARE_CACHE:-0}" \
  -e ENABLE_SLEEP_ON_IDLE="${ENABLE_SLEEP_ON_IDLE:-0}" \
  -e ENABLE_MIXED_CHUNK="${ENABLE_MIXED_CHUNK:-0}" \
  -e TOOL_SERVER="${TOOL_SERVER:-}" \
  "${DOCKER_OPTIONAL_ENV_ARGS[@]}" \
  -e MAX_JOBS="${MAX_JOBS:-4}" \
  -e API_KEY="${API_KEY}" \
  -e ADMIN_API_KEY="${ADMIN_API_KEY:-}" \
  -e HTTP_PORT="${HTTP_PORT}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e HF_HOME="/app/models" \
  -e HF_HUB_CACHE="/app/models" \
  -e TRANSFORMERS_CACHE="/app/models" \
  -e FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}" \
  -v "${HF_HOME}:/app/models" \
  "${IMAGE_NAME}"