from qwen_vl_utils import process_vision_info
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info


model_path = "xxxx/checkpoint-xx"


model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
    )
model.disable_talker()
processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

conversation = [
    {
        "role": "user",
        "content": [
                    {
                        "type": "video",
                        "video": "file:///xxx.mp4",
                        "max_pixels": 151200
                    },
                    {
                        "type": "text",
                        "text": "\n"
                        }
                ],
    }
]

USE_AUDIO_IN_VIDEO=True

text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True,padding_side="left",add_special_tokens=False, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = inputs.to(model.device).to(model.dtype)

# Inference: Generation of the output text and audio
text_ids = model.generate(**inputs,return_audio=False, use_audio_in_video=USE_AUDIO_IN_VIDEO)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, text_ids)
]
text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)



response = text[0]
print('*'*30)
print(response)
