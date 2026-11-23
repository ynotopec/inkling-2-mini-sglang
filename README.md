# inkling-2-mini-sglang

Serve `thinkingmachines/Inkling-2-Mini` with SGLang in Docker, tuned for a
**single DGX Spark (GB10)** — TP1 / EP1, FA4 attention, BF16 GEMM, page 128,
`--mem-fraction-static 0.45`.

Looped MoE, **26B total / 5.4B active** per token. 24 layers, 128 routed
experts (top-8, no shared), layers 6–17 run 3× per token (48 layer visits).
Experts are NVFP4 (layers 0–19) and MXFP8 (layers 20–23); weights ~18 GB, so
it fits one GPU. Text + image input. A 3-layer DSpark MTP draft head is bundled
in the checkpoint (`mtp.safetensors`) for speculative decoding.

## Features
- idempotent `install.sh`
- `source run.sh [IP] [PORT]`
- API token required
- DGX Spark (GB10) defaults; override to run on H200 / RTX PRO 6000 / GB300
- Balanced (plain decode) **or** DSpark (speculative) strategy, one env var
- multimodal, reasoning parser (`inkling`), tool-call parser (`inkling2`)

## Requirements
- Docker + NVIDIA Container Toolkit
- NVIDIA GPU with ≥ 32 GB VRAM / unified memory (DGX Spark GB10: 128 GB LPDDR5X)
- enough disk for the ~18 GB checkpoint + container image

## Install
```bash
./install.sh
```

If `.env` does not exist, it is created from `.env.example`. `install.sh` and
`run.sh` treat `.env.example` as the default source of truth: any unset variable
is loaded from the commented defaults, and `.env` only overrides what you change
(including `BASE_IMAGE`).

## Configure

Edit `.env` and set at least:

```bash
API_KEY=change-me
```

Common options:

* `HF_TOKEN` if the model download requires authentication
* `SERVED_MODEL_NAME=custom` is the default OpenAI-compatible model ID
* `MAX_RUNNING_REQUESTS=8` and `MAX_QUEUED_REQUESTS=12` bound concurrency and queue depth
* `ENABLE_DSPARK=0` selects plain decoding
* `SPECULATIVE_NUM_DRAFT_TOKENS=8` — the verified DSpark draft-tokens per step
* `CONTEXT_LENGTH=65536` to leave more room for the KV pool / concurrency
  (unset = model default 1,048,576)
* `MEM_FRACTION_STATIC=0.45` — the verified DGX Spark value
* `ENABLE_MULTIMODAL=0` for a text-only server (drops `--enable-multimodal`)
* `BASE_IMAGE=lmsysorg/sglang:inkling-2-mini` — the image carrying SGLang
  support for Inkling-2-Mini (SGLang PR #51926; not in a release yet)

## Run

```bash
source ./run.sh 0.0.0.0 30000
```

## Health

```bash
curl http://127.0.0.1:30000/health
```

## OpenAI-compatible API

```bash
curl http://127.0.0.1:30000/v1/models \
  -H "Authorization: Bearer ***"
```

### Chat completion

```bash
curl http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ***" \
  -d '{
    "model": "custom",
    "messages": [
      {"role": "user", "content": "Bonjour"}
    ]
  }'
```

### Image input (multimodal)

```bash
curl http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ***" \
  -d '{
    "model": "custom",
    "messages": [
      {"role": "user", "content": [
        {"type": "text", "text": "Describe this image."},
        {"type": "image_url", "image_url": {"url": "https://.../photo.jpg"}}
      ]}
    ]
  }'
```

## Two strategies

### Balanced (`ENABLE_DSPARK=0`)
Plain decode. Verified on DGX Spark at **93.56 %** on GSM8K (sgl-eval, 1319
questions, temperature 0).

### DSpark (default, `ENABLE_DSPARK=1`)
Speculative decoding on the bundled MTP draft head, 8 draft tokens per step.
Verified on DGX Spark at **93.93 %** on the same GSM8K run. The launcher applies these cookbook settings when DSpark is active. Each can be overridden in `.env`:

```
SGLANG_OPT_INKLING2_VERIFY_REL_BIAS_KERNEL=1
SGLANG_OPT_DSV41_DSPARK_MID_FOLD=1
SGLANG_OPT_DSV41_DSPARK_PREPARE_INPUTS_ONCE=1
SGLANG_OPT_DSV41_DSPARK_SKIP_PREFILLED_INPUTS=1
```

## Verified DGX Spark recipe (source: SGLang cookbook)

The cookbook Balanced command below is the baseline for both strategies:

```
MAX_JOBS=4
sglang serve \
  --model-path thinkingmachines/Inkling-2-Mini \
  --tp 1 --ep 1 \
  --attention-backend fa4 \
  --dtype bfloat16 \
  --page-size 128 \
  --mem-fraction-static 0.45 \
  --grammar-backend xgrammar \
  --enable-multimodal \
  --reasoning-parser inkling \
  --tool-call-parser inkling2 \
  --host 0.0.0.0 --port 30000
```

DSpark adds `--speculative-algorithm DSPARK --speculative-num-draft-tokens 8`.

## Suggested defaults

### DGX Spark (GB10, 128 GB) — this repo's target
* `TP_SIZE=1`, `EP_SIZE=1`
* `MEM_FRACTION_STATIC=0.45`
* `CONTEXT_LENGTH` unset (model default) — set `65536` to trade context for KV
* `ENABLE_DSPARK=1` (DSpark), 8 running requests and 12 queued by default

### H200 / GB300 (4 GPU)
* `TP_SIZE=4`, `EP_SIZE=4`
* `MEM_FRACTION_STATIC=0.85`
* `CONTEXT_LENGTH=131072`

### RTX PRO 6000 / RTX 5090
* `TP_SIZE=1`, `EP_SIZE=1`
* RTX 5090 (32 GB): `CONTEXT_LENGTH=65536` (Balanced) or `32768` (DSpark)

## Notes

* Keep `EP == TP`. The NVFP4 experts cannot be TP-sharded, because their
  activation scales must cover each expert's full input width.
* DSpark accepts a non-default loop pass count? No — it refuses it at startup.
  Leave `INKLING2_LOOP_NUM_PASSES` empty.
* `--inkling2-loop-num-passes` / `--inkling2-loop-share-cache` only work on the
  variable-loop research checkpoint (`loop_config.is_varloop=true`); the main
  `Inkling-2-Mini` checkpoint is fixed-loop and refuses them. Left empty here.
* Keep the default prefill CUDA graph backend. `--cuda-graph-backend-prefill
  breakable` crashes on the first replay.
* DGX Spark uses 128 GB **unified** LPDDR5X shared with the CPU (~225 GB/s
  usable). Keep `--mem-fraction-static` ≤ 0.80 and run ONE engine at a time —
  a second engine or an oversized prefill batch can freeze the box, and leaked
  unified memory survives `docker stop` (only a reboot clears it).
* If you hit OOM, reduce `MEM_FRACTION_STATIC`, `CONTEXT_LENGTH`, or
  `MAX_RUNNING_REQUESTS`.
* The container auto-restarts on hang/crash (`RESTART_POLICY=unless-stopped`).
  An in-container watchdog terminates SGLang if `/health` stays down after the
  startup grace, so Docker can recover. Tune with
  `HEALTHCHECK_MONITOR_START_GRACE` (default 1800s — first load downloads the
  ~18 GB checkpoint), `_INTERVAL`, `_MAX_FAILURES`.

## References

* Model: https://huggingface.co/thinkingmachines/Inkling-2-Mini
* SGLang cookbook: https://docs.sglang.ai/cookbook/autoregressive/ThinkingMachines/Inkling-2-Mini
* SGLang support PR: https://github.com/sgl-project/sglang/pull/51926
* Image: `lmsysorg/sglang:inkling-2-mini` (multi-arch amd64/arm64, CUDA 13)
* SGLang server args: https://docs.sglang.ai/advanced_features/server_arguments.html