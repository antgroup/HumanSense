from PIL import Image
import os
import random
import numpy as np
import copy
import json
from qwen_vl_utils import process_vision_info
from .PROMPT_TEMPLATE_easy import PROMPT_TEMPLATE_INDEFINITE, PROMPT_TEMPLATE_SINGLE4, PROMPT_TEMPLATE_SINGLE2, PROMPT_TEMPLATE_QA, PROMPT_TEMPLATE_SINGLE3, PROMPT_TEMPLATE_SINGLE5


def _make_QA_sat_aug(example, pmt_tag=""):
    
    """
    {
        'a_type': 'NA', 
        'question': 'How many Chairs are visible in the scene?', 
        'answer_choices': ['3', '0', '6', '4', '5'], 
        'correct_answer': '3',
    }
    {
        'a_type': 'JA', 
        'question': 'Consider the 3D positions of the objects in the scene and not just the 2D positions in the image. Is the centerpoint of  black colour chair (marked A) at a higher height than brown top white leg dining table (marked B)?', 
        'answer_choices': ['yes', 'no'], 
        'correct_answer': 'yes',
    }
    {
        'a_type': 'MCA', 
        'question': 'Considering the relative positions, where is black colour chair  shape square (highlighted by a blue box) with respect to brown rectangular extendable dining table (highlighted by a green box)?', 
        'answer_choices': ['left', 'right'], 
        'correct_answer': 'left',
    }

    """

    def _choose_NA_pmt():
        # random
        pmts = [
            "\nPlease answer the question using a single word or phrase.",
            "\nPlease answer the question using a single word or phrase.",
            "\nPlease answer the question using a single word or phrase.",
            "\nAnswer with a single number.",
            "\nAnswer with a single number.",
            "\nDo not response anything other than a single number!",
        ]
        pmt = random.choice(pmts)
        return pmt
    
    def _choose_MCA_pmt():
        # random
        pmt1_pmt2_list = [
            ["\nSelect from the following choices.", ""],
            ["\nChoose between the following options.", ""],
            ["\nAnswer with the option's letter from the given choices directly.", ""],
            ["", "\nPlease select the correct answer from the options above."],
            ["", "\nAnswer with the option's letter from the given choices directly."],
            ["", ""],
        ]
        pmt1, pmt2 = random.choice(pmt1_pmt2_list)
        return pmt1, pmt2


    def _make_MCA_options_answer(answer_choices, correct_answer, supp_options_flag=False):

        supp_list = [
            "I don't know."
            "I have no idea."
            "The question is unclear.",
            "Cannot be determined from the context.",
            "Not enough information to determine.",
            
            "All options are correct.",
            "At least two options are correct.",

            "Depends on interpretation.",
            "Varies by situation.",
            "There are exceptions.",
            "This requires further research.",

            "The premise is incorrect.",
            "Multiple answers are possible.",
            "The question contains a logical fallacy.",
            "This cannot be answered objectively.",
        ]

        option_temp_list = [
            "{opt_letter}. {opt_text}",
            "({opt_letter}) {opt_text}",
        ]

        options_prefix_list = [
            "\nOptions:\n",
            "\noptions:\n",
            "\n",
            " ",
        ]

        option_letters = "ABCDEFGHI"

        # 补充
        if supp_options_flag:
            target_opt_num = random.choice([3, 3, 4, 4, 5])
            supp_choices = np.random.choice(supp_list, target_opt_num - len(answer_choices), replace=False).tolist()
            all_answer_choices = answer_choices + supp_choices
        else:
            all_answer_choices = answer_choices
        
        # 打乱
        shuffled_choices = copy.deepcopy(all_answer_choices)
        random.shuffle(shuffled_choices)

        # 选项
        options_dict = {}
        opt_temp = random.choice(option_temp_list)
        shuffled_choices_with_letter = []
        for i, opt_text in enumerate(shuffled_choices):
            opt_letter = option_letters[i]
            opt = opt_temp.format(opt_letter=opt_letter, opt_text=opt_text)
            shuffled_choices_with_letter.append(opt)
            options_dict[opt_letter] = opt_text

        options_text = '\n'.join(shuffled_choices_with_letter)
        options_prefix = random.choice(options_prefix_list)

        options = options_prefix + options_text

        answer_ind = shuffled_choices.index(correct_answer)
        answer = option_letters[answer_ind]

        info_for_reward = {
            "a_type": "MCA",
            "options_dict": options_dict,
        }
        return options, answer, info_for_reward
        

    a_type = example["questions"]["question_pattern"]
    question = example["questions"]["question"]
    correct_answer = example["questions"]["answer"]

    video_token = "<video>"
    if not pmt_tag:
        pmt_tag = "\nOutput the thinking process in <think> </think> and final answer in <answer> </answer> tags."

    # NA_THRE = 0.7
    NA_THRE = 1
    SUPP_OPTIONS_TERE = 0.8
    if a_type == "Question and answer (Arabic numerals)":
        answer_choices = [1,2,3,4,5]
        if random.random() < NA_THRE:
            # use NA
            answer_text = correct_answer

            info_for_reward = {
                "a_type": a_type,
            }
        else:
            # use MCA
            options, answer_text, info_for_reward = _make_MCA_options_answer(answer_choices, correct_answer, supp_options_flag=False)
            pmt1, pmt2 = _choose_MCA_pmt()

        # question_text = f'{video_token} {question}{pmt1}{options}{pmt2}{pmt_tag}'
        question_text = PROMPT_TEMPLATE_QA.format(question)
        # question_text += "\n\nAnswer:"

        return question_text, answer_text, info_for_reward, 0, "Question and answer (Arabic numerals)"

    elif a_type == "Judgment question (Yes or No)":
        # use NA
        answer_text = correct_answer

        info_for_reward = {
            "a_type": a_type,
        }
        question_text = PROMPT_TEMPLATE_YES_NO.format(question)
        # question_text += "\n\nAnswer:"

        return question_text, answer_text, info_for_reward, 0, "Judgment question (Yes or No)"
    elif a_type == "Multiple-choice question (select one or more answer choices)":
        options = example["questions"]["options"]
        question_text = PROMPT_TEMPLATE_INDEFINITE.format(question, *options)
        # question_text += "\n\nAnswer:"
        info_for_reward = {
            "a_type": a_type,
        }
        answer_text = correct_answer

        return question_text, answer_text, info_for_reward, 0, "Multiple-choice question (select one or more answer choices)"
    elif a_type == "Multiple-choice question (select one answer choice)":
        options = example["questions"]["options"]
        if len(options) == 2:
            question_text = PROMPT_TEMPLATE_SINGLE2.format(question, *options)
            # question_text += "\n\nAnswer:"
        elif len(options) == 3:
            question_text = PROMPT_TEMPLATE_SINGLE3.format(question, *options)
            # question_text += "\n\nAnswer:"
        elif len(options) == 4:
            # print("%"*30,options)
            question_text = PROMPT_TEMPLATE_SINGLE4.format(question, *options)
            # question_text += "\n\nAnswer:"
        elif len(options) == 5:
            # print("%"*30,options)
            question_text = PROMPT_TEMPLATE_SINGLE5.format(question, *options)
            # question_text += "\n\nAnswer:"

        if example["questions"]["answer"][0].lower() == 'a':
            value = options[0]
        elif example["questions"]["answer"][0].lower() == 'b':
            value = options[1]
        elif example["questions"]["answer"][0].lower() == 'c':
            value = options[2]
        elif example["questions"]["answer"][0].lower() == 'd':
            value = options[3]
        elif example["questions"]["answer"][0].lower() == 'e':
            value = options[4]
        
            
        answer_text = correct_answer
        info_for_reward = {
            "a_type": a_type,
        }

        return question_text, answer_text, info_for_reward, value, "Multiple-choice question (select one answer choice)"
    else: 
        return 0, 0, 0, 0


def make_conversation_sat_aug(example, base_model_prompt=False, dataroot="", pmt_tag=""):
    if base_model_prompt:
        raise NotImplementedError("only support instruct model now!")

    # img_p = os.path.join(dataroot, example["images"][0])
    # image = Image.open(img_p)
    question_text, answer_text, info_for_reward, value, querstion_pattern = _make_QA_sat_aug(example, pmt_tag=pmt_tag)

    if question_text == 0: return 0


    if_audio = False
    data_class="video_wo_audio"
    if example["questions"]["task_type"] in ["Rapport_Recognition", "Emo_Strategy", "Talkingperson_Recognition","Familiarity_Recognition","Relation_Recognition", "Fraud_Recognition","Psychological_Chat"]:
        if_audio = True
        data_class="video_w_audio"
    
    # file = example["video_path"]
    file = os.path.join(os.getcwd(), example["video_path"])


    if example["questions"]["task_type"] in ["Fraud_Recognition","Psychological_Chat"]:
        data_class = "audio"
        data_example = {
            'if_audio':if_audio,
            'data_class':data_class,
            "prompt": [
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
                            {"type": "text", "text": question_text},
                        ],
                    },
            ],
            "solution": {
                # "solution": f"<answer>{answer_text}</answer>",
                "solution": str(answer_text[0]),
                "solution_value": value,
                "querstion_pattern": querstion_pattern,
            },
            "task_type": example["questions"]["task_type"],
            "info_for_reward": info_for_reward,
        }
    else:
        data_example = {
            'if_audio':if_audio,
            'data_class':data_class,
            "prompt": [
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
                            },
                            {"type": "text", "text": question_text},
                        ],
                    },
            ],
            "solution": {
                # "solution": f"<answer>{answer_text}</answer>",
                "solution": str(answer_text),
                "solution_value": value,
                "querstion_pattern": querstion_pattern,
            },
            "task_type": example["questions"]["task_type"],
            "info_for_reward": info_for_reward,
        }

    return data_example


def make_dataset(dataset_json_file, base_model_prompt=False, pmt_tag=""):
    dataroot = os.path.dirname(dataset_json_file)

    with open(dataset_json_file, 'r', encoding='utf-8') as f:
        sat_data = json.load(f)
    
    dataset = []
    for i, sample in enumerate(sat_data):
        # print(i)
        data = make_conversation_sat_aug(sample, base_model_prompt=base_model_prompt, dataroot=dataroot, pmt_tag=pmt_tag)

        # if data["task_type"] not in ["Rapport_Recognition", "Emo_Strategy", "Talkingperson_Recognition","Familiarity_Recognition","Relation_Recognition", "Fraud_Recognition","Psychological Chat Room"]: continue
        # if data["task_type"] not in ["Fraud_Recognition","Psychological Chat Room"]: continue

        if data == 0: continue
        dataset.append(data)
    random.seed(42) 
    random.shuffle(dataset)

    save_dataset_json(dataset)
    return dataset


def save_dataset_json(dataset):
    save_dataset_json = os.getenv("SAVE_DATASET_JSON")
    if save_dataset_json:
        save_dataset = [{k:v for k, v in example.items() if k != 'image'} for example in dataset]
        with open(save_dataset_json, 'w', encoding="utf-8") as f:
            json.dump(save_dataset, f, ensure_ascii=False, indent=4)

            
