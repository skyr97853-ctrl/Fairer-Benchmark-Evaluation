#!/bin/bash

# 1. 指定使用的显卡 ID
GPU_IDS="0" 

# 2. 定义数据集列表
#DATASETS=("data_normal_aqua" "data_normal_logiqa" "data_normal_mmlu" "data_normal_openbookqa" "data_normal_commonsense_qa" "data_normal_arc_c" "data_normal_winogrande" "data_normal_hellaswag")
DATASETS=( "data_normal_logiqa" "data_normal_mmlu" "data_normal_openbookqa" "data_normal_commonsense_qa" "data_normal_arc_c" "data_normal_winogrande" "data_normal_hellaswag")

# 3. 定义对应的训练轮数 (根据您的新指令设为 2.0)
#EPOCHS=(15.0 4.0 0.5 2.0 0.5 2.0 5.0 0.5)
EPOCHS=( 5.0 0.5 2.0 0.5 2.0 5.0 0.5)


# 4. 实验元数据配置 (修改这里即可)
SOURCE_TASK="aqua"  # 来源 Adapter 的任务名
# 自动获取日期，例如今天会生成 "3-12"
CURRENT_DATE=$(date +%-m-%-d) 


# 5. 路径配置
MODEL_PATH="/data/fanzeyu/.cache/modelscope/hub/models/modelscope/Llama-2-7b-ms"
# 固定的 Adapter 路径
FIXED_ADAPTER="/data/fanzeyu/saves/self/LLama-2-7B/${SOURCE_TASK}/LLama-2-7B-reverse-${SOURCE_TASK}-3-18"
# 输出父目录
OUTPUT_BASE_DIR="/data/fanzeyu/saves/compare/LLama-2-7B/${SOURCE_TASK}"


# ================= 循环执行区 =================

for i in "${!DATASETS[@]}"
do
    DS_NAME=${DATASETS[$i]}
    CURRENT_EPOCH=${EPOCHS[$i]}
    
    # 鲁棒性检查
    [ -z "$CURRENT_EPOCH" ] && CURRENT_EPOCH=3.0

    # 提取当前训练数据集简称
    SHORT_NAME=${DS_NAME#data_normal_}

    # 动态构建输出目录：包含 [来源任务]-[当前任务]-[日期]
    CURRENT_OUTPUT_DIR="$OUTPUT_BASE_DIR/LLama-2-7B-compare-${SOURCE_TASK}-${SHORT_NAME}-${CURRENT_DATE}"

    echo "========================================================="
    echo "任务进度: $((i+1))/${#DATASETS[@]}"
    echo "实验日期: $CURRENT_DATE"
    echo "来源任务: $SOURCE_TASK"
    echo "加载 Adapter: $FIXED_ADAPTER"
    echo "当前训练: $DS_NAME ($CURRENT_EPOCH Epochs)"
    echo "输出路径: $CURRENT_OUTPUT_DIR"
    echo "========================================================="

    CUDA_VISIBLE_DEVICES=$GPU_IDS  llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path $MODEL_PATH \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template default \
    --flash_attn auto \
    --dataset_dir data \
    --dataset $DS_NAME \
    --cutoff_len 2048 \
    --learning_rate 5e-05 \
    --num_train_epochs $CURRENT_EPOCH \
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
    --output_dir $CURRENT_OUTPUT_DIR \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --adapter_name_or_path $FIXED_ADAPTER \
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

    if [ $? -ne 0 ]; then
        echo "任务 $DS_NAME 失败，程序退出。"
        #exit 1
    fi
done

echo "所有实验已于 $CURRENT_DATE 完成！"