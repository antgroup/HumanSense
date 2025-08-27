from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import re
import os

model, processor = None, None

from model.modelclass import Model
class Qwen25VL(Model):
    def __init__(self):
        Qwen25VL_Init()

    def Run(self, question_info):
        return Qwen25VL_Run(question_info)
    
    def name(self):
        return "Qwen-25-VL-7B-Instruct"

def Qwen25VL_Init():
    # 加载 Qwen2.5-VL-7B-Instruct 模型
    global model, processor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", 
        torch_dtype="auto", 
        device_map="auto", 
        attn_implementation="flash_attention_2"
    )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

def Qwen25VL_Run(question_info):
    file = question_info["file"]
    inp = question_info["inp"]
    frame_num = question_info["full_frame_number"]

    print('*'*30, file)
    print(inp)
    print('*'*30)
    

    if frame_num > 300 and frame_num < 600:
        fps = 0.5
    elif frame_num >= 600:
        fps = 0.2
    else:
        fps = 1.0


    messages = [
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
            ]
        }
    ]

    # 准备推理数据
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # 推理
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    response = output_text[0]
    print(response)
    return response
