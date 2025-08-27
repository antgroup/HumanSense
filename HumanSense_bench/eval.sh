cd HumanSense_bench/src

# Qwen2-VL-7B-Instruct, Qwen-25-VL-7B-Instruct, Qwen25-Omni-7B, rivideo-omni7B, InternVL3-8B-Instruct, LLaVA-NeXT-Video, LLaVA-OneVision, MiniCPM-V-2_6, LongVA, VideoLLaMA3, IXC25omniLive, gpt4o, ola
source activate omni # Switch to the corresponding model environment.
EVAL_MODEL="Qwen25-Omni-7B"

Devices=0


# Resume code from interruption
RESUME=false
# Whether to use sound: true for the omni model, false for all other models.
USE_AUDIO=true
# Mainly applicable to models trained using GRPO
USE_THINK=false
# Whether to include ASR text
USE_ASR=false


SUFFIX=""
if [ "$USE_AUDIO" = true ]; then
    SUFFIX="${SUFFIX}_audio"
fi
if [ "$USE_THINK" = true ]; then
    SUFFIX="${SUFFIX}_think"
fi
if [ "$USE_ASR" = true ]; then
    SUFFIX="${SUFFIX}_asr"
fi

DATA_FILE="data/HumanSense_VQA.json"
OUTPUT_FILE="results/HumanSense_VQA_${EVAL_MODEL}${SUFFIX}.json"


CMD="CUDA_VISIBLE_DEVICES=$Devices python eval.py --model_name $EVAL_MODEL --data_file $DATA_FILE --output_file $OUTPUT_FILE"
if [ "$USE_AUDIO" = true ]; then
    CMD="$CMD --audio"
fi
if [ "$USE_THINK" = true ]; then
    CMD="$CMD --think"
fi
if [ "$USE_THINK_omini" = true ]; then
    CMD="$CMD --think_omini"
fi
if [ "$USE_ASR" = true ]; then
    CMD="$CMD --asr"
fi
if [ "$RESUME" = true ]; then
    CMD="$CMD --resume"
fi

echo "Running command: $CMD"
eval $CMD