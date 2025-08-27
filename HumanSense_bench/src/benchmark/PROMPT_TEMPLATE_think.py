
PROMPT_TEMPLATE_INDEFINITE = '''
You are an advanced multimodal analysis AI assistant. You have been provided with video content and a question.

Question: {}, Which of the following are correct? (Select all)

Options:
{}
{}
{}
{}

Output the thinking process in <think> </think> and final answer in <answer> </answer> tags, i.e., <think> reasoning process here </think><answer> answer correct letters only here </answer>.
'''

PROMPT_TEMPLATE_SINGLE5 = '''
You are an advanced multimodal analysis AI assistant. You have been provided with video content and a question.

Question: {}, which one of the following is correct?

Options:
{}
{}
{}
{}
{}

Output the thinking process in <think> </think> and final answer in <answer> </answer> tags, i.e., <think> reasoning process here </think><answer> answer one correct letter only here </answer>.
'''

PROMPT_TEMPLATE_SINGLE4 = '''
You are an advanced multimodal analysis AI assistant. You have been provided with video content and a question.

Question: {}, which one of the following is correct?

Options:
{}
{}
{}
{}

Output the thinking process in <think> </think> and final answer in <answer> </answer> tags, i.e., <think> reasoning process here </think><answer> answer one correct letter only here </answer>.
'''

PROMPT_TEMPLATE_SINGLE3 = '''
You are an advanced multimodal analysis AI assistant. You have been provided with video content and a question.

Question: {}, which one of the following is correct?

Options:
{}
{}
{}

Output the thinking process in <think> </think> and final answer in <answer> </answer> tags, i.e., <think> reasoning process here </think><answer> answer one correct letter only here </answer>.
'''

PROMPT_TEMPLATE_SINGLE2 = '''
You are an advanced multimodal analysis AI assistant. You have been provided with video content and a question.

Question: {}, which one of the following is correct?

Options:
{}
{}

Output the thinking process in <think> </think> and final answer in <answer> </answer> tags, i.e., <think> reasoning process here </think><answer> answer one correct letter only here </answer>.
'''



PROMPT_TEMPLATE_YES_NO = '''
You are a multimodal analysis AI assistant.

Your task:
- Analyze the video carefully
- Choose one correct answer from "Yes" or "No"

Question: {}

Output strictly in the following format:
Output the thinking process in <think> </think> and final answer in <answer> </answer> tags, i.e., <think> reasoning process here </think><answer> answer here </answer>.
'''

PROMPT_TEMPLATE_QA = '''
You are an advanced multimodal analysis AI assistant. You have been provided with video content and a question.

Question: {}

Output the thinking process in <think> </think> and final answer in <answer> </answer> tags, i.e., <think> reasoning process here </think><answer> answer a single Arabic numeral between 0 and 10 here </answer>.
'''



