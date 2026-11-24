ARG BASE_IMAGE=lmsysorg/sglang:inkling-2-mini
FROM ${BASE_IMAGE}

# Inkling-2-Mini is supported from SGLang PR #51926, shipped in the
# `lmsysorg/sglang:inkling-2-mini` image (multi-arch amd64/arm64, CUDA 13).
# We keep the base image's prebuilt SGLang / sgl-kernel / FlashInfer stack
# untouched: overriding it is the #1 cause of CUDA ABI mismatches on GB10
# (sm_121). Only opt-in knobs are exposed, mirroring qwen36-sglang.

ARG UPGRADE_TRANSFORMERS=0
ARG UPGRADE_SGLANG=0
ARG SGLANG_WHL_INDEX=https://docs.sglang.ai/whl/cu130/

RUN if [ "${UPGRADE_TRANSFORMERS}" = "1" ]; then \
      pip install --no-cache-dir --upgrade "git+https://github.com/huggingface/transformers.git"; \
    else \
      echo "Skipping transformers upgrade to keep base-image CUDA stack coherent."; \
    fi

RUN if [ "${UPGRADE_SGLANG}" = "1" ]; then \
      pip install --no-cache-dir --upgrade \
        sglang sglang[all] sglang-kernel \
        --index-url "${SGLANG_WHL_INDEX}"; \
    else \
      echo "Skipping sglang upgrade; using base-image prebuilt binaries (inkling-2-mini)."; \
    fi

# Defaults target the DGX Spark (GB10, sm_121, 128 GB unified LPDDR5X).
# Verified cookbook recipe: TP1 / EP1, FA4 attention, BF16 GEMM, page 128,
# mem-fraction-static 0.45. Override freely via .env.
ENV MODEL_PATH="thinkingmachines/Inkling-2-Mini" \
    SERVED_MODEL_NAME="custom" \
    TP_SIZE="1" \
    EP_SIZE="1" \
    MEM_FRACTION_STATIC="0.45" \
    CONTEXT_LENGTH="" \
    PAGE_SIZE="128" \
    ATTENTION_BACKEND="fa4" \
    BF16_GEMM_BACKEND="" \
    DTYPE="bfloat16" \
    GRAMMAR_BACKEND="xgrammar" \
    MAX_RUNNING_REQUESTS="" \
    MAX_QUEUED_REQUESTS="" \
    HTTP_PORT="30000" \
    ENABLE_MULTIMODAL="1" \
    ENABLE_TOOLS="1" \
    ENABLE_REASONING="1" \
    ENABLE_DSPARK="1" \
    SPECULATIVE_NUM_DRAFT_TOKENS="8" \
    INKLING2_LOOP_NUM_PASSES="" \
    INKLING2_LOOP_SHARE_CACHE="0" \
    ENABLE_SLEEP_ON_IDLE="0" \
    ENABLE_MIXED_CHUNK="0" \
    CHUNKED_PREFILL_SIZE="" \
    KV_CACHE_DTYPE="" \
    TOOL_SERVER="" \
    MAX_JOBS="4" \
    HEALTHCHECK_MONITOR_START_GRACE="1800" \
    HEALTHCHECK_MONITOR_INTERVAL="15" \
    HEALTHCHECK_MONITOR_MAX_FAILURES="8" \
    HF_HOME="/app/models" \
    TRANSFORMERS_CACHE="/app/models" \
    HF_HUB_CACHE="/app/models" \
    XDG_CACHE_HOME="/app/models" \
    FLASHINFER_DISABLE_VERSION_CHECK="1" \
    PYTHONUNBUFFERED="1"

RUN mkdir -p /app/models /app/data
WORKDIR /app

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 30000

HEALTHCHECK --interval=30s --timeout=30s --start-period=1800s --retries=5 \
  CMD curl -fsS "http://127.0.0.1:${HTTP_PORT}/health" || exit 1

CMD ["/app/entrypoint.sh"]