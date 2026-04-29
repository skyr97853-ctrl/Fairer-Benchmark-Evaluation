#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR/LLaMA-Factory" || exit 1

DATASET=""
MODEL_PATH=""
ADAPTER_PATH=""
OUTPUT_DIR=""

llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path "$MODEL_PATH" \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template default \
    --flash_attn auto \
    --dataset_dir LLaMA-Factory/data \
    --dataset "$DATASET" \
    --cutoff_len 2048 \
    --learning_rate 5e-05 \
    --num_train_epochs 3.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 10 \
    --warmup_steps 0 \
    --packing False \
    --enable_thinking True \
    --report_to none \
    --output_dir "$OUTPUT_DIR" \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --adapter_name_or_path "$ADAPTER_PATH" \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --use_rslora True \
    --pissa_init True \
    --pissa_convert True \
    --lora_target all \
    --val_size 0.01 \
    --eval_strategy steps \
    --eval_steps 10 \
    --per_device_eval_batch_size 2
