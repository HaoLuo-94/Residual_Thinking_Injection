#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Batch inference + evaluation launcher (multi-dataset + run_evals-style engine management + pass@k)
# Edit the "Configurable parameters" section below, then run:
#   bash examples/run_batch.sh
# -----------------------------------------------------------------------------
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
set -euo pipefail

# ======================== Configurable parameters ========================

MODEL=""

# Data: one or more files. Multi-dataset runs write separate outputs by file stem.
#   Single dataset -> same behavior as legacy (respects OUTPUT_FILE / SUMMARY_FILE / DATASET_NAME)
#   Multi-dataset  -> each dataset writes RUN_DIR/<stem>.jsonl, plus RUN_DIR/summary_all.json
# When DATA is empty and DATASET_NAME is set, auto-discover test.jsonl etc. under DATA_ROOT/<DATASET_NAME>/
DATA=(
  ""
)
DATA_ROOT=""
# Single dataset only. Enables dataset-specific prompt building and evaluation in utils:
#   aime2025 | aime2325 | math500 | minerva | mbpp | humaneval | livecodebench | ifeval
DATASET_NAME="math500"

# Output directory (auto-appends timestamped subdirectory)
OUTPUT_DIR=""

# Prompt (used when no utils evaluator matches; evaluator build_prompt takes over when matched)
PROMPT_FIELD="${PROMPT_FIELD:-prompt}"   # empty -> auto-infer from problem/question etc.
PROMPT_FORMAT="chat"                     # generation | chat
# 1=enable Qwen3 thinking mode (chat template generates ... block)
ENABLE_THINKING=1
SYSTEM_PROMPT=""
PROMPT_TEMPLATE=""                       # e.g. 'Question: {problem}\nAnswer:'

# Batch run
BATCH_SIZE=128
MAX_TOKENS=8192                     # can be overridden per dataset in run_batch.py DATASET_MAX_TOKENS
SKIP=0
LIMIT=""                                 # empty=no limit

# Residual injection
ALPHA=0.5
RESIDUAL_TOP_K=16
RESIDUAL_ARCH="Qwen3ForResidualInjection"                         # empty=auto-select from config.json
USE_PLUGIN_MODE=0

# baseline: 1=run native model, no injection/patching/architecture override
# set BASELINE=0 for think-phase injection experiments
BASELINE=1

# Injection phase: all=entire generation; think=only <think>...</think> thinking phase
INJECT_PHASE="think"
INJECT_LAYER=0                           # -1=all layers; 0=first layer; k=layer k (0-indexed)
THINK_START_TOKEN="<think>"
THINK_END_TOKEN="</think>"
THINK_START_ID=""                        # required for think in plugin mode; otherwise resolved via tokenizer
THINK_END_ID=""

# vLLM
GPU_MEMORY_UTILIZATION=0.95

# Sampling (temperature=0 is greedy; TOP_K=-1 means unlimited in vLLM)
TEMPERATURE=0.6
TOP_P=0.95
TOP_K=20
PRESENCE_PENALTY=0.0
REPETITION_PENALTY=1.0

NUM_SAMPLES=1

PASSK_TEMPERATURE=0.6                     # only used when TEMPERATURE above stays at default 0
PASSK_TOP_P=0.95                          # only used when TOP_P above stays at default 1.0

# Random seed (empty=not fixed: true pass@k capability / no effect under greedy decoding)
SEED=46

# Evaluation: 1=skip (when data has no answer field)
NO_EVAL=0

# ---- Engine management (ported from run_evals; meaningful for multi-dataset runs only) ----
# 1=reuse the same engine for all datasets (avoids reloading weights; recommended for same-model multi-dataset, default)
# 0=rebuild a fresh engine per dataset (fully isolated/reproducible, but reloads weights for each dataset)
REUSE_ENGINE=0
# 1=enable CUDA deterministic algorithms (reproducible, slightly slower)
DETERMINISTIC=1
# 1=set CUDA_LAUNCH_BLOCKING=1 (improves determinism further but significantly slows inference)
CUDA_LAUNCH_BLOCKING=0

PYTHON="${PYTHON:-python}"

# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# When DATA is empty, auto-resolve data file under DATA_ROOT by DATASET_NAME
if [[ ${#DATA[@]} -eq 0 && -n "${DATASET_NAME}" ]]; then
  for candidate in test.jsonl dev.jsonl test_25.jsonl test-400.jsonl; do
    auto_path="${DATA_ROOT}/${DATASET_NAME}/${candidate}"
    if [[ -f "${auto_path}" ]]; then
      DATA=("${auto_path}")
      echo "Auto-resolved DATA from DATASET_NAME: ${auto_path}"
      break
    fi
  done
  if [[ ${#DATA[@]} -eq 0 ]]; then
    echo "ERROR: DATA is empty and no file found under ${DATA_ROOT}/${DATASET_NAME}/" >&2
    exit 1
  fi
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_DIR}/batch_${TIMESTAMP}"
mkdir -p "${RUN_DIR}"

# Multi-dataset: run_batch.py uses OUTPUT_FILE parent directory (RUN_DIR) to derive per-dataset filenames
OUTPUT_FILE="${RUN_DIR}/batch_output.jsonl"
SUMMARY_FILE="${RUN_DIR}/batch_output.summary.json"
LOG_FILE="${RUN_DIR}/run.log"

NUM_DATA=${#DATA[@]}

cd "${PROJECT_ROOT}"

CMD=(
  "${PYTHON}" "${SCRIPT_DIR}/run_batch.py"
  --model "${MODEL}"
  --data "${DATA[@]}"
  --data-root "${DATA_ROOT}"
  --output-file "${OUTPUT_FILE}"
  --batch-size "${BATCH_SIZE}"
  --max-tokens "${MAX_TOKENS}"
  --alpha "${ALPHA}"
  --residual-top-k "${RESIDUAL_TOP_K}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --top-k "${TOP_K}"
  --presence-penalty "${PRESENCE_PENALTY}"
  --num-samples "${NUM_SAMPLES}"
  --passk-temperature "${PASSK_TEMPERATURE}"
  --passk-top-p "${PASSK_TOP_P}"
  --prompt-format "${PROMPT_FORMAT}"
  --skip "${SKIP}"
  --inject-phase "${INJECT_PHASE}"
  --inject-layer "${INJECT_LAYER}"
)

# Pass --summary-file / --dataset-name for single dataset only (ignored in multi-dataset mode)
if [[ "${NUM_DATA}" -eq 1 ]]; then
  CMD+=(--summary-file "${SUMMARY_FILE}")
  if [[ -n "${DATASET_NAME}" ]]; then
    CMD+=(--dataset-name "${DATASET_NAME}")
  fi
fi

if [[ -n "${PROMPT_FIELD}" ]]; then
  CMD+=(--prompt-field "${PROMPT_FIELD}")
fi
if [[ -n "${SYSTEM_PROMPT}" ]]; then
  CMD+=(--system-prompt "${SYSTEM_PROMPT}")
fi
if [[ -n "${PROMPT_TEMPLATE}" ]]; then
  CMD+=(--prompt-template "${PROMPT_TEMPLATE}")
fi
if [[ -n "${LIMIT}" ]]; then
  CMD+=(--limit "${LIMIT}")
fi
if [[ -n "${SEED}" ]]; then
  CMD+=(--seed "${SEED}")
fi
if [[ -n "${RESIDUAL_ARCH}" ]]; then
  CMD+=(--residual-arch "${RESIDUAL_ARCH}")
fi
if [[ "${USE_PLUGIN_MODE}" == "1" ]]; then
  CMD+=(--plugin-mode)
fi
if [[ "${BASELINE}" == "1" ]]; then
  CMD+=(--baseline)
fi
if [[ "${ENABLE_THINKING}" == "1" ]]; then
  CMD+=(--enable-thinking)
fi
if [[ -n "${THINK_START_TOKEN}" ]]; then
  CMD+=(--think-start-token "${THINK_START_TOKEN}")
fi
if [[ -n "${THINK_END_TOKEN}" ]]; then
  CMD+=(--think-end-token "${THINK_END_TOKEN}")
fi
if [[ -n "${THINK_START_ID}" ]]; then
  CMD+=(--think-start-id "${THINK_START_ID}")
fi
if [[ -n "${THINK_END_ID}" ]]; then
  CMD+=(--think-end-id "${THINK_END_ID}")
fi
if [[ "${NO_EVAL}" == "1" ]]; then
  CMD+=(--no-eval)
fi

# Engine management flags
if [[ "${REUSE_ENGINE}" == "1" ]]; then
  CMD+=(--reuse-engine)
fi
if [[ "${DETERMINISTIC}" == "1" ]]; then
  CMD+=(--deterministic)
fi
if [[ "${CUDA_LAUNCH_BLOCKING}" == "1" ]]; then
  CMD+=(--cuda-launch-blocking)
fi

echo "Run dir: ${RUN_DIR}  (datasets: ${NUM_DATA}, pass@${NUM_SAMPLES})"
echo "Command: ${CMD[*]}" | tee "${RUN_DIR}/launch_args.txt"

exec "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"