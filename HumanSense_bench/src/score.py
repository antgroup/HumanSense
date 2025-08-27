import json
import csv
from collections import defaultdict
import argparse
import os 

def one_or_more_answeracore(model_answer, correct_answer):
    for answer in model_answer:
        if answer not in correct_answer: return 0
    answer_exist = [answer in model_answer for answer in correct_answer]  
    return 1 if all(answer_exist) else 0.5      


def count(args):
    src = args.src
    model = args.model

    # Load the JSON file
    with open(src, 'r') as file:
        data = json.load(file)

    # Initialize counters
    stats = defaultdict(lambda: defaultdict(int))

    # Process each entry in the JSON data
    total = 0

    for ques in data:
        question = ques["questions"]
        task_type = question["task_type"]
        if task_type == 'Abnormal_Recognition': continue
        question_pattern = question["question_pattern"]

        if model not in question or not question.get(model, None):
            continue

        if question_pattern in ["Multiple-choice question (select one answer choice)"]:
            correct_answer = question["answer"][0]
            model_answer = question.get(model, None)[0]
            if model_answer:
                total += 1
                stats[task_type]["total"] += 1
                if correct_answer.lower() == model_answer.lower():
                    stats[task_type]["correct"] += 1
        elif question_pattern in ["Question and answer (Arabic numerals)"]:
            correct_answer = question["answer"]
            model_answer = question.get(model, None)[0]
            if model_answer:
                total += 1
                stats[task_type]["total"] += 1
                try:
                    # print(correct_answer, model_answer, int(correct_answer) == int(model_answer))
                    if int(correct_answer) == int(model_answer):
                        stats[task_type]["correct"] += 1
                except:
                    pass
        elif question_pattern in ["Judgment question (Yes or No)"]:
            correct_answer = question["answer"]
            model_answer = question.get(model, None).replace('.', '').replace(' ', '')
            if model_answer:
                total += 1
                stats[task_type]["total"] += 1
                if correct_answer.lower() == model_answer.lower():
                    stats[task_type]["correct"] += 1
        elif question_pattern in ["Multiple-choice question (select one or more answer choices)"]:
            correct_answer = question["answer"].lower()
            model_answer = question.get(model, None).replace('.', '').replace(',', '').lower()
            if model_answer:
                total += 1
                stats[task_type]["total"] += 1
                stats[task_type]["correct"] += one_or_more_answeracore(model_answer, correct_answer)


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

    with open(output_file, 'w') as json_file:
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