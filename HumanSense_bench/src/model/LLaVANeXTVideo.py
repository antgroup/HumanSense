import av
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration

from model.modelclass import Model

class LLaVANextVideo7(Model):
    def __init__(self):
        LLaVANextVideo7_Init()

    def Run(self, question_info):
        return LLaVANextVideo7_Run(question_info)
    
    def name(self):
        return "LLaVA-Next-Video-7B"


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])



def LLaVANextVideo7_Init():
    model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"

    global model, processor

    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
    ).cuda()

    processor = LlavaNextVideoProcessor.from_pretrained(model_id)


def LLaVANextVideo7_Run(question_info):

    video_path = question_info["file"]
    inp = question_info["inp"]
    # define a chat history and use `apply_chat_template` to get correctly formatted prompt
    
    
    # Each value in "content" has to be a list of dicts with types ("text", "image", "video") 
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": inp},
                {"type": "video"},
                ],
        },
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    
    container = av.open(video_path)

    # sample uniformly 8 frames from the video, can sample more for longer videos
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / 8).astype(int)
    clip = read_video_pyav(container, indices)
    inputs_video = processor(text=prompt, videos=clip, padding=True, return_tensors="pt").to(model.device)

    output = model.generate(**inputs_video, max_new_tokens=100, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs_video.input_ids, output)
    ]
    response = processor.decode(generated_ids_trimmed[0], skip_special_tokens=True)


    print("*"*30)
    print(response)
    return  response
