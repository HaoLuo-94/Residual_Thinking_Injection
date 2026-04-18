#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ===== 基础配置：按需修改 =====
MODEL_PATH="/kpfs-llm-text/models/Qwen3-4B"
DATASET_NAME="gsm8k"
DATA_ROOT="/kpfs-llm-text/hao.luo/project/Residual/data"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
OUTPUT_DIR="/kpfs-llm-text/hao.luo/project/Residual/code/output"
PROMPT_FIELD="${PROMPT_FIELD:-prompt}"
PROMPT_FORMAT="chat"         # chat | generation
SYSTEM_PROMPT=""
OUTPUT_FIELD="/kpfs-llm-text/hao.luo/project/Residual/code/llm_output"

# ===== 生成参数：按需修改 =====
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_NEW_TOKENS="1024"
DTYPE="${DTYPE:-bfloat16}"                     # float16 | bfloat16 | float32
DEVICE_MAP="${DEVICE_MAP:-auto}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
DO_SAMPLE="${DO_SAMPLE:-false}"
RESUME="${RESUME:-false}"

# ===== residual 注入参数：按需修改 =====
ENABLE_RESIDUAL="true"
RESIDUAL_ALPHA="0.001"
RESIDUAL_LAYER_START="0"
RESIDUAL_LAYER_END="-1"
RESIDUAL_ENTROPY_THRESHOLD="0.01"
RESIDUAL_LOW_ENTROPY_PATIENCE="5"
RESIDUAL_THINK_END_TOKEN="</think>"
RESIDUAL_TOPK="32"

mkdir -p "$OUTPUT_DIR"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
RUN_DIR="${RUN_DIR:-$OUTPUT_DIR/batch_run_${TIMESTAMP}}"
mkdir -p "$RUN_DIR"

OUTPUT_FILE="${OUTPUT_FILE:-$RUN_DIR/batch_output.jsonl}"
SUMMARY_FILE="${SUMMARY_FILE:-$RUN_DIR/batch_output.summary.json}"
LOG_FILE="${LOG_FILE:-$RUN_DIR/run.log}"
ARGS_FILE="${ARGS_FILE:-$RUN_DIR/launch_args.txt}"

# 同时输出到终端和日志文件，方便回看整次跑批结果。
exec > >(tee -a "$LOG_FILE") 2>&1

if [[ "$MODEL_PATH" == "/path/to/your/model" ]]; then
  echo "请先设置 MODEL_PATH，再运行脚本。"
  echo "例如：MODEL_PATH=/your/model/path bash start_batch.sh"
  exit 1
fi

if [[ -z "$DATASET_NAME" ]]; then
  echo "请先设置 DATASET_NAME，再运行脚本。"
  exit 1
fi

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "数据目录不存在：$DATA_ROOT"
  exit 1
fi

CMD=(
  python  "$SCRIPT_DIR/run_batch.py"
  --model "$MODEL_PATH"
  --dataset-name "$DATASET_NAME"
  --data-root "$DATA_ROOT"
  --dataset-split "$DATASET_SPLIT"
  --output-file "$OUTPUT_FILE"
  --summary-file "$SUMMARY_FILE"
  --prompt-field "$PROMPT_FIELD"
  --prompt-format "$PROMPT_FORMAT"
  --output-field "$OUTPUT_FIELD"
  --batch-size "$BATCH_SIZE"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --dtype "$DTYPE"
  --device-map "$DEVICE_MAP"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --repetition-penalty "$REPETITION_PENALTY"
)

if [[ -n "$SYSTEM_PROMPT" ]]; then
  CMD+=(--system-prompt "$SYSTEM_PROMPT")
fi

if [[ "$DO_SAMPLE" == "true" ]]; then
  CMD+=(--do-sample)
fi

if [[ "$RESUME" == "true" ]]; then
  CMD+=(--resume)
fi

if [[ "$ENABLE_RESIDUAL" == "true" ]]; then
  CMD+=(
    --enable-residual
    --residual-alpha "$RESIDUAL_ALPHA"
    --residual-layer-start "$RESIDUAL_LAYER_START"
    --residual-layer-end "$RESIDUAL_LAYER_END"
    --residual-entropy-threshold "$RESIDUAL_ENTROPY_THRESHOLD"
    --residual-low-entropy-patience "$RESIDUAL_LOW_ENTROPY_PATIENCE"
    --residual-think-end-token "$RESIDUAL_THINK_END_TOKEN"
    --residual-topk "$RESIDUAL_TOPK"
  )
fi

if [[ $# -gt 0 ]]; then
  CMD+=("$@")
fi

{
  echo "TIMESTAMP=$TIMESTAMP"
  echo "RUN_DIR=$RUN_DIR"
  echo "MODEL_PATH=$MODEL_PATH"
  echo "DATASET_NAME=$DATASET_NAME"
  echo "DATA_ROOT=$DATA_ROOT"
  echo "DATASET_SPLIT=$DATASET_SPLIT"
  echo "OUTPUT_DIR=$OUTPUT_DIR"
  echo "OUTPUT_FILE=$OUTPUT_FILE"
  echo "SUMMARY_FILE=$SUMMARY_FILE"
  echo "LOG_FILE=$LOG_FILE"
  echo "PROMPT_FIELD=$PROMPT_FIELD"
  echo "PROMPT_FORMAT=$PROMPT_FORMAT"
  echo "SYSTEM_PROMPT=$SYSTEM_PROMPT"
  echo "OUTPUT_FIELD=$OUTPUT_FIELD"
  echo "BATCH_SIZE=$BATCH_SIZE"
  echo "MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
  echo "DTYPE=$DTYPE"
  echo "DEVICE_MAP=$DEVICE_MAP"
  echo "TEMPERATURE=$TEMPERATURE"
  echo "TOP_P=$TOP_P"
  echo "REPETITION_PENALTY=$REPETITION_PENALTY"
  echo "DO_SAMPLE=$DO_SAMPLE"
  echo "RESUME=$RESUME"
  echo "ENABLE_RESIDUAL=$ENABLE_RESIDUAL"
  echo "RESIDUAL_ALPHA=$RESIDUAL_ALPHA"
  echo "RESIDUAL_LAYER_START=$RESIDUAL_LAYER_START"
  echo "RESIDUAL_LAYER_END=$RESIDUAL_LAYER_END"
  echo "RESIDUAL_ENTROPY_THRESHOLD=$RESIDUAL_ENTROPY_THRESHOLD"
  echo "RESIDUAL_LOW_ENTROPY_PATIENCE=$RESIDUAL_LOW_ENTROPY_PATIENCE"
  echo "RESIDUAL_THINK_END_TOKEN=$RESIDUAL_THINK_END_TOKEN"
  echo "RESIDUAL_TOPK=$RESIDUAL_TOPK"
  echo
  echo "CMD="
  printf ' %q' "${CMD[@]}"
  echo
} > "$ARGS_FILE"

echo "启动批处理任务："
printf ' %q' "${CMD[@]}"
echo
echo "运行目录：$RUN_DIR"
echo "输出文件：$OUTPUT_FILE"
echo "汇总文件：$SUMMARY_FILE"
echo "日志文件：$LOG_FILE"
echo "参数文件：$ARGS_FILE"

"${CMD[@]}"
