# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from open_r1_video.trainer_omini import Qwen2VLGRPOHezhiTrainer_omini
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from my_data_utils.sat_aug_qa_prompt_omini import make_dataset
from torch.serialization import add_safe_globals
from deepspeed.runtime.fp16.loss_scaler import LossScaler
from deepspeed.runtime.zero.config import ZeroStageEnum

add_safe_globals([LossScaler, ZeroStageEnum])

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        # default_factory=lambda: ["accuracy"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    jsonl_path: Optional[str] = field(
        default=None,
        metadata={"help": "json file path"},
    )


# Multiple-choice question (select one or more answer choices)
def Multiple_choice_score_answer(correct_answer, model_answer):
    for answer in model_answer:
        if answer not in correct_answer: 
            # print('多选题'*10)
            # print(correct_answer)
            # print(model_answer)
            return 0
    answer_exist = [answer in model_answer for answer in correct_answer]  
    return 1 if all(answer_exist) else 0.5      



def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""

    contents = [completion[0]["content"] for completion in completions]
    # print('contents', contents) # print online completion
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    # print('*'*10)
    # print("content",len(contents))
    # print("sol",len(solution))
    # print("content", contents[0])
    # print("sol", solution[0]["solution"])

    print('*'*10)
    print(contents[:2], solution[:2])
    
    for content, sol_items in zip(contents, solution):
        sol = sol_items["solution"]
        solution_value = sol_items["solution_value"]
        question_pattern = sol_items["querstion_pattern"]

        # print('*'*10)
        # print("content",content)
        # print("sol",sol)
        # print('*'*10)

        reward = 0.0
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        if reward == 0.0:
            try:
                ground_truth = sol

                # Extract answer from content if it has think/answer tags
                content_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
                # student_answer = content_match.group(1).strip() if content_match else content.strip()
                if content_match:
                    model_answer = content_match.group(1).strip()

                    if question_pattern == "Question and answer (Arabic numerals)":
                        model_answer = model_answer[0]
                        correct_answer = ground_truth[0]
                        if correct_answer == model_answer:
                            reward = 1.0
                    elif question_pattern == "Multiple-choice question (select one or more answer choices)":
                        model_answer = model_answer.replace('.', '').replace(',', '').replace('，', '').replace(' ', '').lower()
                        correct_answer = ground_truth.replace('.', '').replace(',', '').replace('，', '').replace(' ', '').lower()
                        score = Multiple_choice_score_answer(correct_answer, model_answer)
                        
                        reward = score
                    
                    elif question_pattern == "Multiple-choice question (select one answer choice)":
                        
                        model_answer = model_answer[0]
                        correct_answer = ground_truth
                        if model_answer[0].lower() == ground_truth.lower():
                           
                            reward = 1
                       
                else:
                    print('格式错误'*10, content)
                    model_answer = content
                    if question_pattern == "Question and answer (Arabic numerals)":
                        
                        model_answer = model_answer[0]
                        correct_answer = ground_truth[0]
                        if correct_answer == model_answer:
                          
                            reward = 1.0
                    
                    elif question_pattern == "Multiple-choice question (select one or more answer choices)":
                        
                        model_answer = model_answer.replace('.', '').replace(',', '').replace('，', '').replace(' ', '').lower()
                        correct_answer = ground_truth.replace('.', '').replace(',', '').replace('，', '').replace(' ', '').lower()
                        score = Multiple_choice_score_answer(correct_answer, model_answer)
                       
                        reward = score
                    
                    elif question_pattern == "Multiple-choice question (select one answer choice)":
                        
                        model_answer = model_answer[0]
                        correct_answer = ground_truth
                        if model_answer[0].lower() == ground_truth.lower():
                            reward = 1
                       

            except Exception:
                pass  # Keep reward as 0.0 if both methods fail



        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]
    return [1 if match else 0.0 for match in matches]




reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

from datasets import Dataset, DatasetDict
import json

def create_dataset_from_jsonl_simple(jsonl_path):
    base_dataset = Dataset.from_json(jsonl_path)
    return DatasetDict({
        "train": base_dataset
    })


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    

    base_model_prompt = False
    if model_args.model_name_or_path.split("/")[-1] == "Qwen2-VL-2B" or "Base" in model_args.model_name_or_path:
        base_model_prompt = True

   

    dataset = make_dataset(script_args.jsonl_path, base_model_prompt=base_model_prompt)
    dataset = {'train': dataset}


    

    # import pdb; pdb.set_trace()

    trainer_cls = Qwen2VLGRPOHezhiTrainer_omini

  
    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )


    # Train and push the model to the Hub
    # trainer.train()
    add_safe_globals([LossScaler, ZeroStageEnum])
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)


    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    print(training_args.resume_from_checkpoint)
    training_args.max_completion_length = 2048
    print(training_args.max_completion_length)
    main(script_args, training_args, model_args)
