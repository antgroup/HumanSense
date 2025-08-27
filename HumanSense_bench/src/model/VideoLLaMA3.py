import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoModel, AutoImageProcessor

from model.modelclass import Model
class VideoLLaMA3(Model):
    def __init__(self):
        VideoLLaMA3_Init()

    def Run(self, question_info):
        return VideoLLaMA3_Run(question_info)
    
    def name(self):
        return "VideoLLaMA3"

def VideoLLaMA3_Init():
    global model, processor
    model_name = "DAMO-NLP-SG/VideoLLaMA3-7B"
    model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)


def VideoLLaMA3_Run(question_info):
    # Video Inference
    file = question_info["file"]
    inp = question_info["inp"]

    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": {"video_path": f"file://{file}", "fps": 1, "max_frames": 128}},
                {"type": "text", "text": inp},
            ]
        },
    ]

    inputs = processor(conversation=conversation, return_tensors="pt")
    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    output_ids = model.generate(**inputs, max_new_tokens=128)
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    print('*'*30)
    print(response)

    return response