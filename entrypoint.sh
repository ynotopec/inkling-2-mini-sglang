#!/usr/bin/env bash
set -euo pipefail

# Assemble the SGLang launch command for Inkling-2-Mini on DGX Spark (GB10).
# Base recipe = SGLang cookbook, hw=dgx-spark quant=nvfp4 strategy=balanced:
#   TP1 / EP1, FA4 attention, BF16 GEMM, page 128, mem-fraction-static 0.45.
# DSpark (strategy=dspark) adds the bundled MTP draft head with 8 draft tokens.

ARGS=(
  sglang serve
  --model-path "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --tp "${TP_SIZE}"
  --ep "${EP_SIZE}"
  --attention-backend "${ATTENTION_BACKEND}"
  --dtype "${DTYPE}"
  --page-size "${PAGE_SIZE}"
  --grammar-backend "${GRAMMAR_BACKEND}"
  --host 0.0.0.0
  --port "${HTTP_PORT}"
)

[[ -n "${MEM_FRACTION_STATIC:-}" ]] && ARGS+=( --mem-fraction-static "${MEM_FRACTION_STATIC}" )
[[ -n "${CONTEXT_LENGTH:-}" ]] && ARGS+=( --context-length "${CONTEXT_LENGTH}" )
[[ -n "${MAX_RUNNING_REQUESTS:-}" ]] && ARGS+=( --max-running-requests "${MAX_RUNNING_REQUESTS}" )
[[ -n "${MAX_QUEUED_REQUESTS:-}" ]] && ARGS+=( --max-queued-requests "${MAX_QUEUED_REQUESTS}" )
[[ -n "${BF16_GEMM_BACKEND:-}" ]] && ARGS+=( --bf16-gemm-backend "${BF16_GEMM_BACKEND}" )
[[ -n "${CHUNKED_PREFILL_SIZE:-}" ]] && ARGS+=( --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" )
[[ -n "${KV_CACHE_DTYPE:-}" ]] && ARGS+=( --kv-cache-dtype "${KV_CACHE_DTYPE}" )

[[ "${ENABLE_MULTIMODAL:-1}" == "1" ]] && ARGS+=( --enable-multimodal )
[[ "${ENABLE_REASONING:-1}" == "1" ]] && ARGS+=( --reasoning-parser inkling )
[[ "${ENABLE_TOOLS:-1}" == "1" ]] && ARGS+=( --tool-call-parser inkling2 )
[[ -n "${TOOL_SERVER:-}" && "${ENABLE_TOOLS:-1}" == "1" ]] && ARGS+=( --tool-server "${TOOL_SERVER}" )

[[ "${ENABLE_MIXED_CHUNK:-0}" == "1" ]] && ARGS+=( --enable-mixed-chunk )
[[ "${ENABLE_SLEEP_ON_IDLE:-0}" == "1" ]] && ARGS+=( --sleep-on-idle )

# Loop-core controls: only valid on a variable-loop checkpoint
# (loop_config.is_varloop=true). The published Inkling-2-Mini checkpoint is
# fixed-loop (is_varloop=false), so passing these makes SGLang refuse at
# startup — they are opt-in for the varloop research checkpoint only.
[[ -n "${INKLING2_LOOP_NUM_PASSES:-}" ]] && ARGS+=( --inkling2-loop-num-passes "${INKLING2_LOOP_NUM_PASSES}" )
[[ "${INKLING2_LOOP_SHARE_CACHE:-0}" == "1" ]] && ARGS+=( --inkling2-loop-share-cache )

# DSpark speculative decoding on the bundled MTP draft head.
if [[ "${ENABLE_DSPARK:-0}" == "1" ]]; then
  ARGS+=(
    --speculative-algorithm DSPARK
    --speculative-num-draft-tokens "${SPECULATIVE_NUM_DRAFT_TOKENS:-8}"
  )
fi

if [[ -n "${API_KEY:-}" ]]; then
  ARGS+=( --api-key "${API_KEY}" )
fi
if [[ -n "${ADMIN_API_KEY:-}" ]]; then
  ARGS+=( --admin-api-key "${ADMIN_API_KEY}" )
fi

echo "Launching SGLang:"
printf ' %q' "${ARGS[@]}"
echo

health_url="http://127.0.0.1:${HTTP_PORT}/health"
monitor_interval="${HEALTHCHECK_MONITOR_INTERVAL:-15}"
max_failures="${HEALTHCHECK_MONITOR_MAX_FAILURES:-8}"
start_grace="${HEALTHCHECK_MONITOR_START_GRACE:-1800}"

"${ARGS[@]}" &
server_pid=$!

cleanup() {
  trap - INT TERM
  if kill -0 "${server_pid}" 2>/dev/null; then
    kill -TERM "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" || true
  fi
}
trap cleanup INT TERM

(
  start_ts=$(date +%s)
  fails=0
  while kill -0 "${server_pid}" 2>/dev/null; do
    now_ts=$(date +%s)
    if (( now_ts - start_ts < start_grace )); then
      sleep "${monitor_interval}"
      continue
    fi

    if curl -fsS "${health_url}" >/dev/null 2>&1; then
      fails=0
    else
      fails=$((fails + 1))
      echo "[watchdog] /health failed (${fails}/${max_failures}) on ${health_url}" >&2
      if (( fails >= max_failures )); then
        echo "[watchdog] Health check failed repeatedly; terminating sglang pid ${server_pid} to trigger container restart." >&2
        kill -TERM "${server_pid}" 2>/dev/null || true
        sleep 10
        kill -KILL "${server_pid}" 2>/dev/null || true
        break
      fi
    fi
    sleep "${monitor_interval}"
  done
) &
monitor_pid=$!

wait "${server_pid}"
exit_code=$?
kill "${monitor_pid}" 2>/dev/null || true
wait "${monitor_pid}" 2>/dev/null || true
exit "${exit_code}"