from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

import torch

model, processor = None, None

from model.modelclass import Model
class Qwen25omini(Model):
    def __init__(self):
        Qwen25omini_Init()

    def Run(self, question_info):
        return Qwen25omini_Run(question_info)
    
    def name(self):
        return "Qwen25-Omni-7B"

def Qwen25omini_Init():
    # Load the OneVision model
    global model, processor
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B",
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
    )

    processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

def Qwen25omini_Run(question_info):
    file = question_info["file"]
    inp = question_info["inp"]
    # set use audio in video
    USE_AUDIO_IN_VIDEO = question_info["if_audio"]


    
    if question_info["audio_eval"]:
        conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}        
                ],
        },
        {
            "role": "user",
            "content": [
                {
                "type": "audio", 
                "audio": f"file://{file}",
                },
                {"type": "text", "text": inp},

            ],
        },
        ]
    else:
        # 看到的帧数不超过100帧
        frame_num = question_info["full_frame_number"]
        if frame_num > 300 and frame_num < 600:
            fps = 0.5
        elif frame_num >= 600:
            fps = 0.2
        else:
            fps = 1.0

        if (frame_num/int(question_info["fps"])) * fps > 24:
            fps = float(24 / (frame_num/int(question_info["fps"])))

        conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}        
                ],
        },
        {
            "role": "user",
            "content": [
                {
                "type": "video", 
                "video": f"file://{file}",
                        "max_pixels": 360 * 420,
                        "fps": fps,
                },
                {"type": "text", "text": inp},

            ],
        },
        ]

    # Preparation for inference
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = inputs.to(model.device).to(model.dtype)

    # Inference: Generation of the output text and audio
    text_ids, audio = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, text_ids)
    ]
    text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)



    response = text[0]
    print('*'*30)
    print(inp)
    print('*'*30)
    print(response)

    return response