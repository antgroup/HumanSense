from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

import torch

model, processor = None, None

from model.modelclass import Model
class Qwen2audio7B(Model):
    def __init__(self):
        Qwen2audio7B_Init()

    def Run(self, question_info):
        return Qwen2audio7B_Run(question_info)
    
    def name(self):
        return "Qwen2-Audio-7B-Instruct"

def Qwen2audio7B_Init():
    # Load the OneVision model
    global model, processor
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="auto").cuda()

def Qwen2audio7B_Run(question_info):
    file = question_info["file"]
    inp = question_info["inp"]
    # set use audio in video
    USE_AUDIO_IN_VIDEO = question_info["if_audio"]


    
    if question_info["audio_eval"]:
        conversation = [
        {
            "role": "user",
            "content": [
                {
                "type": "audio", 
                "audio_url": f"file://{file}",
                },
                {"type": "text", "text": inp},

            ],
        },
        ]
   
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios = []
    for message in conversation:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "audio":
                    audios.append(
                        librosa.load(
                            BytesIO(urlopen(ele['audio_url']).read()), 
                            sr=processor.feature_extractor.sampling_rate)[0]
                    )

    inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
    inputs = {key: value.to("cuda") for key, value in inputs.items()}

    generate_ids = model.generate(**inputs, max_length=16384)
    generate_ids = generate_ids[:, inputs["input_ids"].size(1):]

    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    print('*'*30)
    print(response)
    return response


