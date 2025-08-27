cd HumanSense_bench/src


EVAL_MODEL="Qwen25-Omni-7B"


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


if [ "$USE_THINK" = true ]; then
    SCRIPT_NAME="score_think.py"
else
    SCRIPT_NAME="score.py"
fi

OUTPUT_FILE="results/HumanSense_VQA_${EVAL_MODEL}${SUFFIX}.json"
OUTPUT_FILE_audio="results/HumanSense_AQA_${EVAL_MODEL}${SUFFIX}.json"




echo "Running VQA task for model: $EVAL_MODEL with SUFFIX: $SUFFIX"
python $SCRIPT_NAME \
    --model $EVAL_MODEL\
    --src $OUTPUT_FILE \
    --save_dir "results/scores_HumanSense_VQA_${EVAL_MODEL}${SUFFIX}.json"


echo "Running AQA task for model: $EVAL_MODEL with SUFFIX: $SUFFIX"
python $SCRIPT_NAME \
    --model $EVAL_MODEL \
    --src $OUTPUT_FILE_audio \
    --save_dir "results/scores_HumanSense_AQA_${EVAL_MODEL}${SUFFIX}.json"