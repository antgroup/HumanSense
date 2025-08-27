export WANDB_PROJECT=stage2
export WANDB_NAME=stage2

export WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
export WANDB_MODE=offline

export NCCL_DEBUG=WARN
export TORCHDTENSOR_FALLBACK=1
export CUDA_LAUNCH_BLOCKING=1

source activate omni

# for H20
pip install nvidia-cublas-cu12 -U
export LD_LIBRARY_PATH=/opt/conda/envs/omni/lib/python3.10/site-packages/nvidia/cublas/lib/:$LD_LIBRARY_PATH
ldconfig


cd Open-R1-Video


experiment_dir="experiments"
output_dir="${experiment_dir}/${WANDB_PROJECT}/"
export SAVE_DATASET_JSON="${output_dir}/dataset.json"
dataset_json_file="data/merged_audio.json"


mkdir -p ${experiment_dir}/$WANDB_PROJECT/$WANDB_NAME
mkdir -p ${output_dir}


checkpoint_dir="xxxx/checkpoint-333"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="15352" \
    src/open_r1_video/grpo_hezhi_omini.py \
    --deepspeed scripts/zero3_offload_omini.json \
    --output_dir ${experiment_dir}/$WANDB_PROJECT/$WANDB_NAME \
    --model_name_or_path ${checkpoint_dir} \
    --dataset_name xxx \
    --jsonl_path ${dataset_json_file} \
    --max_prompt_length 16384 \
    --learning_rate 1e-6 \
    --beta 0.1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to tensorboard \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $WANDB_NAME \
    --save_steps 10 \
    --save_only_model false \
    --save_total_limit 10 \
    --num_generations 8  \
    # --resume_from_checkpoint 