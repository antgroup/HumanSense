# Self-test Model Modifications

1. Modify relevant parameters in HumanSense_bench/eval.sh and HumanSense_bench/eval_audio.sh

```python
source activate xxxx # Switch to the corresponding model environment.
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
```

2. Modify codes in HumanSense_bench/src/eval.py
```python
model_classes = {
    "Qwen2-VL-7B-Instruct": ("model.Qwen2VL", "Qwen2VL"),
    "Qwen-25-VL-7B-Instruct": ("model.Qwen25VL", "Qwen25VL"),
    "Qwen25-Omni-7B": ("model.Qwen25omini", "Qwen25omini"),
    "Qwen25-Omni-3B": ("model.Qwen25omini3B", "Qwen25omini"), 
    "rivideo-omni7B": ("model.rivideo_omni", "rivideo_omni"),
    "rivideo-7B": ("model.rivideo", "rivideo"),
    "InternVL3-8B-Instruct": ("model.InternVL3", "InternVL"),
    "LLaVA-Next-Video-7B": ("model.LLaVANeXTVideo", "LLaVANextVideo7"),
    "Qwen2-Audio-7B-Instruct": ("model.Qwen2audio", "Qwen2audio7B"),
    "LLaVA-OneVision": ("model.LLaVAOneVision", "LLaVAOneVision"),
    "MiniCPM-V-2_6": ("model.MiniCPMV", "MiniCPMV"),
    "LongVA": ("model.LongVA", "LongVA"),
    "VideoLLaMA3": ("model.VideoLLaMA3", "VideoLLaMA3"),
    "IXC25omniLive": ("model.IXComini", "IXC25omniLive"),
    "gpt4o": ("model.gpt4o", "gpt4o"),
    "ola": ("model.ola_omni", "ola_omni"),
    }
```

3. Create your own file for your model, following the pattern of other files in HumanSense-main/HumanSense_bench/src/model (This code template is modified from [Streamingbench](https://github.com/THUNLP-MT/StreamingBench))

All models must subclass the `model.modelclass` class.

The class enforces a common interface via which we can extract responses from a model:

```python
class Model:
    def __init__(self):
        """
        Initialize the model
        """
        pass
    
    def Run(self, question_info):
        """
        Given the question information (file_path, input prompt and full_frame_number), run the model and return the response
        """

        return ""

    def name():
        """
        Return the name of the model
        """
        return ""
```
- `__init__(self)`
  - This is where you should load your model weights, tokenizer, and any other necessary components. For example:
    ```python
    def __init__(self):
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct", 
            torch_dtype="auto", 
            device_map="auto", 
            attn_implementation="flash_attention_2"
        )

        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    ```

- `Run(self, question_info)`
    ```python
    file = question_info["file"]
    inp = question_info["inp"]
    frame_num = question_info["full_frame_number"]
    ```
  - This is where you should implement the logic for running your model on the given input. The function should return the response from the model as a string.

- `name(self)`
  - This is where you should return the name of your model. This will be used to identify your model when running the benchmark.
  