#!/bin/bash

# 指定使用的显卡序号（0代表第一块显卡）
export CUDA_VISIBLE_DEVICES=1


# --- 核心配置 ---
DATA_NAME="aqua" # 目标数据集简称
KIND="normal"
BASE_MODEL="LLama-2-7B"
MODEL_PATH="/data/fanzeyu/.cache/modelscope/hub/models/modelscope/Llama-2-7b-ms"
BASE_ADAPTER_DIR="/data/fanzeyu/saves/self/$BASE_MODEL/$DATA_NAME/$BASE_MODEL-$KIND-$DATA_NAME-3-25-high-rank-lr_5e-5-lora_rank_128-pure-more"
BASE_OUTPUT_ROOT="/data/fanzeyu/evaluation/self/$BASE_MODEL/$DATA_NAME/$KIND-high-rank-lr_5e-5-lora_rank_128-pure-more"

# --- Checkpoint 范围 ---
START=10
STOP=250
STEP=10

CURRENT_DATE=$(date +%m-%d)

for i in $(seq $START $STEP $STOP)
do
    CKPT_NAME="checkpoint-$i"
    CURRENT_ADAPTER="$BASE_ADAPTER_DIR/$CKPT_NAME"
    
    # 检查适配器路径是否存在，不存在则跳过
    if [ ! -d "$CURRENT_ADAPTER" ]; then
        echo "警告: $CURRENT_ADAPTER 不存在，跳过本次循环。"
        continue
    fi

    # 自动拼接输出目录名：模型-数据集-步数-日期
    CURRENT_OUTPUT="${BASE_OUTPUT_ROOT}/$BASE_MODEL-${KIND}-${DATA_NAME}-${CKPT_NAME}-${CURRENT_DATE}-high-rank-lr_5e-5-lora_rank_128-pure-more"

    echo "----------------------------------------------------------------"
    echo "开始测评: $CKPT_NAME | 数据集: data_normal_$DATA_NAME"
    echo "输出目录: $CURRENT_OUTPUT"
    
    mkdir -p "$CURRENT_OUTPUT"

    llamafactory-cli train \
        --stage sft \
        --model_name_or_path "$MODEL_PATH" \
        --adapter_name_or_path "$CURRENT_ADAPTER" \
        --preprocessing_num_workers 16 \
        --finetuning_type lora \
        --template default \
        --dataset_dir data \
        --eval_dataset "data_normal_${DATA_NAME}" \
        --cutoff_len 1024 \
        --per_device_eval_batch_size 4 \
        --predict_with_generate True \
        --max_new_tokens 512 \
        --output_dir "$CURRENT_OUTPUT" \
        --do_predict True \
        --overwrite_output_dir True \
        --report_to none \
        --trust_remote_code True

    echo "Checkpoint $i 测评完成。"
    # 稍微休息 2 秒，确保显存回收
    sleep 2
done

echo "所有批次执行完毕！"
#source eval1_next1.sh