from tqdm import tqdm
import os
import json
from utils.data_execution import get_model_response
from utils.video_execution import split_video
import re


import transformers
print("transformers.__path__"*10, transformers.__path__)


class HSPIBench():
    def __init__(self, data):
        HSPIBenchInit(data)

    def eval(self, data, model, args):
        HSPIBenchEval(data, model, args)

def HSPIBenchInit(data):
    pass

def HSPIBenchEval(data, MODEL, args):
    output_path = args.output_file
    tmp_video_dir = args.tmp_video_dir

    if args.think:
        from benchmark.PROMPT_TEMPLATE_think import PROMPT_TEMPLATE_INDEFINITE, PROMPT_TEMPLATE_SINGLE4, PROMPT_TEMPLATE_SINGLE2, PROMPT_TEMPLATE_QA, PROMPT_TEMPLATE_SINGLE3, PROMPT_TEMPLATE_SINGLE5
    else:
        from benchmark.PROMPT_TEMPLATE import PROMPT_TEMPLATE_INDEFINITE, PROMPT_TEMPLATE_SINGLE4, PROMPT_TEMPLATE_SINGLE2, PROMPT_TEMPLATE_QA, PROMPT_TEMPLATE_SINGLE3, PROMPT_TEMPLATE_SINGLE5


    for subset in tqdm(data):
    # for subset in tqdm(reversed(data)):

        question = subset["questions"]
        ques = question["question"]
        
        if MODEL.name() in subset["questions"] and subset["questions"][MODEL.name()]:
            continue

        video_path = subset["video_path"]

        # if subset["video_or_clip"] == "clip":
        #     start_time, end_time = map(float, subset["time"].strip("[]").split(" - "))

        #     file = split_video(video_path, int(start_time), int(end_time), tmp_video_dir)
        # else:
        current_working_directory = os.getcwd()
        output_file = os.path.join(current_working_directory, video_path)
        file = output_file
    

        if not args.audio_eval:
            full_frame_number= question["full_frame_number"]
        question_pattern = question["question_pattern"]


        # if question["task_type"] not in ["Rapport_Recognition", "Emo_Strategy", "Familiarity_Recognition","Relation_Recognition","Psychological_Chat","Fraud_Recognition"]: continue
 
        if args.asr:
            if question["task_type"] not in ["Rapport_Recognition", "Emo_Strategy", "Talkingperson_Recognition","Familiarity_Recognition","Relation_Recognition"]: continue
        if args.think_omini:
            if question["task_type"] not in ["Rapport_Recognition", "Emo_Strategy", "Talkingperson_Recognition","Familiarity_Recognition","Relation_Recognition","Psychological_Chat","Fraud_Recognition"]: continue


        if question_pattern == "Question and answer (Arabic numerals)":
            inp = PROMPT_TEMPLATE_QA.format(ques)
            # continue
        elif question_pattern == "Multiple-choice question (select one answer choice)":
            options = question["options"]
            if len(options) == 2:
                inp = PROMPT_TEMPLATE_SINGLE2.format(ques, *options)
                # continue
            elif len(options) == 3:
                inp = PROMPT_TEMPLATE_SINGLE3.format(ques, *options)
                # continue
            elif len(options) == 4:
                inp = PROMPT_TEMPLATE_SINGLE4.format(ques, *options)
                # continue
            elif len(options) == 5:
                inp = PROMPT_TEMPLATE_SINGLE5.format(ques, *options)
                # continue
            
        elif question_pattern == "Multiple-choice question (select one or more answer choices)":
            options = question["options"]
            inp = PROMPT_TEMPLATE_INDEFINITE.format(ques, *options)
            # continue
        
        if_audio = False
        if args.audio:
            if question["task_type"] in ["Rapport_Recognition", "Emo_Strategy", "Talkingperson_Recognition","Familiarity_Recognition","Relation_Recognition","Fraud_Recognition","Psychological_Chat"]:
                if_audio = True
        
       

        if args.asr:
            if question["task_type"] in ["Rapport_Recognition", "Emo_Strategy", "Talkingperson_Recognition","Familiarity_Recognition","Relation_Recognition","Fraud_Recognition","Psychological_Chat"]:
                asr = "The following provides the ASR (Automatic Speech Recognition) text from the video:" + question["asr_en"]
                inp = asr + inp
        
        if args.think_omini:
            
            syst = '''System Prompt:

                    When analyzing the video, focus on identifying:

                    Key Characteristics: Recognize notable features, actions, emotions, or behaviors of people.

                    Chat Content: Extract relevant dialogue or spoken words.

                    Events: Identify significant activities, interactions, or key moments.

                    For the following tasks, base your reasoning on the above elements to draw conclusions.''' 
            inp = syst + inp

        if args.audio_eval:
            if_audio = True
            question_info = {
                            "file":file,
                            "inp":inp,
                            "if_audio": if_audio,
                            "audio_eval": args.audio_eval
                            }
        else:
            question_info = {
                            "file":file,
                            "inp":inp,
                            "full_frame_number":full_frame_number,
                            "if_audio": if_audio,
                            "fps": question["fps"],
                            "audio_eval": args.audio_eval
                            }

        response = get_model_response(MODEL, question_info)
        question[MODEL.name()] = response

        with open(output_path, "w") as f:
            json.dump(data, f, indent=4)
