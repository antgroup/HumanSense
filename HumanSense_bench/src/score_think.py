import json
import csv
from collections import defaultdict
import argparse
import os 
import re

def one_or_more_answeracore(model_answer, correct_answer):
    # model_answer = remove_think_tags(model_answer)
    for answer in model_answer:
        if answer not in correct_answer: return 0
    answer_exist = [answer in model_answer for answer in correct_answer]  
    return 1 if all(answer_exist) else 0.5      



def One_choice_score_answer(correct_letter, model_answer, solution_value):
    correct_letter = correct_letter[0].strip().lower()
   
    parts_ = solution_value.split('.', 1)    
    solution_value = parts_[1].strip().lower()

    model_answer = model_answer.strip().lower()


    if model_answer in ['a', 'b', 'c', 'd']:
        return 1 if correct_letter == model_answer else 0
    elif model_answer.startswith(('a.', 'b.', 'c.', 'd.')):
        parts = model_answer.split('.', 1)   
        le = parts[0]
        value = parts[1]
        if le.strip() == correct_letter or value.strip().replace('.', '').replace(',', '').replace('，', '') == solution_value.replace('.', '').replace(',', '').replace('，', ''):
          
            return 1
        else: 
          
            return 0
    else:

        return 1 if model_answer.strip().replace('.', '').replace(',', '').replace('，', '') == solution_value.replace('.', '').replace(',', '').replace('，', '') else 0



def count(args):
    src = args.src
    model = args.model

    # Load the JSON file
    with open(src, 'r') as file:
        data = json.load(file)

    # Initialize counters
    stats = defaultdict(lambda: defaultdict(int))

    total = 0

    for ques in data:
        question = ques["questions"]
        task_type = question["task_type"]
        question_pattern = question["question_pattern"]


        if model not in question or not question.get(model, None):
            continue

        ground_truth = question["answer"]
        content_match = re.search(r"<answer>(.*?)</answer>", question.get(model, None), re.DOTALL)
        
     
        if content_match:
            model_answer = content_match.group(1).strip()
     
            if question_pattern in ["Multiple-choice question (select one answer choice)"]:
                options = question["options"]
                if question["answer"][0].lower() == 'a':
                    value = options[0]
                elif question["answer"][0].lower() == 'b':
                    value = options[1]
                elif question["answer"][0].lower() == 'c':
                    value = options[2]
                elif question["answer"][0].lower() == 'd':
                    value = options[3]
                elif question["answer"][0].lower() == 'e':
                    value = options[4]
                model_answer = model_answer
                correct_answer = ground_truth
    
                total += 1
                stats[task_type]["total"] += 1
                if One_choice_score_answer(correct_answer, model_answer, value):
                    stats[task_type]["correct"] += 1
                # break
            elif question_pattern in ["Question and answer (Arabic numerals)"]:
                correct_answer = ground_truth
                model_answer = model_answer[0]
                total += 1
                stats[task_type]["total"] += 1
              
                if int(correct_answer) == int(model_answer):
                    stats[task_type]["correct"] += 1
            elif question_pattern in ["Judgment question (Yes or No)"]:
                correct_answer = question["answer"]
                model_answer = question.get(model, None).replace('.', '').replace(' ', '')
                if model_answer:
                    total += 1
                    stats[task_type]["total"] += 1
                    if correct_answer.lower() == model_answer.lower():
                        stats[task_type]["correct"] += 1
            elif question_pattern in ["Multiple-choice question (select one or more answer choices)"]:
                correct_answer = ground_truth.replace('.', '').replace(',', '').replace('，', '').replace(' ', '').lower()
                model_answer = model_answer.replace('.', '').replace(',', '').replace('，', '').replace(' ', '').lower()
                total += 1
                stats[task_type]["total"] += 1
                stats[task_type]["correct"] += one_or_more_answeracore(model_answer, correct_answer)
        else:
            if question_pattern in ["Multiple-choice question (select one or more answer choices)"]:
                correct_answer = ground_truth.replace('.', '').replace(',', '').replace('，', '').replace(' ', '').lower()
                model_answer = question.get(model, None).replace('.', '').replace(',', '').replace('，', '').replace(' ', '').replace('\n', '').lower()
                total += 1
                stats[task_type]["total"] += 1
                stats[task_type]["correct"] += one_or_more_answeracore(model_answer, correct_answer)
            elif question_pattern in ["Question and answer (Arabic numerals)"]:
                correct_answer = ground_truth
                model_answer = question.get(model, None)[0]
                total += 1
                stats[task_type]["total"] += 1
                if int(correct_answer) == int(model_answer):
                    stats[task_type]["correct"] += 1
            elif question_pattern in ["Multiple-choice question (select one answer choice)"]:
                
                options = question["options"]
                if question["answer"][0].lower() == 'a':
                    value = options[0]
                elif question["answer"][0].lower() == 'b':
                    value = options[1]
                elif question["answer"][0].lower() == 'c':
                    value = options[2]
                elif question["answer"][0].lower() == 'd':
                    value = options[3]
                elif question["answer"][0].lower() == 'e':
                    value = options[4]
                model_answer = question.get(model, None)[0]
                correct_answer = ground_truth
                total += 1
                stats[task_type]["total"] += 1
                if One_choice_score_answer(correct_answer, model_answer, value):
                    stats[task_type]["correct"] += 1
       

    total_correct = 0
    total_total = 0
    
    for task_type, counts in stats.items():
        counts["accuracy"] = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
        total_correct += counts["correct"]
        total_total += counts["total"]

    overall_accuracy = total_correct / total_total if total_total > 0 else 0
    stats["all"]["total"] = total_total
    stats["all"]["correct"] = total_correct
    stats["all"]["accuracy"] = overall_accuracy


    # Save results as a JSON file
    current_working_directory = os.getcwd()
    output_file = os.path.join(current_working_directory, f'{args.save_dir}')

    # Save results as a JSON file
    with open(f'{args.save_dir}{model}_stats.json', 'w') as json_file:
        json.dump(stats, json_file, indent=4)

    print(f"{total} items have been statisticed")
    print(f"results path: {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen2-VL', type=str, help='model name')
    parser.add_argument('--src', default='', type=str, help='Path to the data file')
    parser.add_argument('--save_dir', default='', type=str, help='Path to the data file')
    args = parser.parse_args()

    count(args)

if __name__ == "__main__":
    main()