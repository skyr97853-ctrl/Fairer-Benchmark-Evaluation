#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_DIR/LLaMA-Factory" || exit 1

DATASET=""
MODEL_PATH=""
ADAPTER_PATH=""
OUTPUT_DIR=""

mkdir -p "$OUTPUT_DIR"

llamafactory-cli train \
    --stage sft \
    --model_name_or_path "$MODEL_PATH" \
    --adapter_name_or_path "$ADAPTER_PATH" \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template default \
    --dataset_dir "$PROJECT_DIR/LLaMA-Factory/data" \
    --eval_dataset "$DATASET" \
    --cutoff_len 1024 \
    --per_device_eval_batch_size 4 \
    --predict_with_generate True \
    --max_new_tokens 512 \
    --output_dir "$OUTPUT_DIR" \
    --do_predict True \
    --overwrite_output_dir True \
    --report_to none \
    --trust_remote_code True
