import json
import os
import sys
import re
from openai import OpenAI

class ChatCompletion:
    def __init__(self, id, choices, created, model, object, service_tier, system_fingerprint, usage):
        self.choices = choices

class Choice:
    def __init__(self, finish_reason, index, logprobs, message):
        self.finish_reason = finish_reason
        self.index = index
        self.logprobs = logprobs
        self.message = message

class ChatCompletionMessage:
    def __init__(self, content, refusal, role, function_call, tool_calls):
        self.content = content

class CompletionUsage:
    def __init__(self, completion_tokens=541, prompt_tokens=72, total_tokens=613):
        self.completion_tokens = completion_tokens
        self.prompt_tokens = prompt_tokens
        self.total_tokens = total_tokens


def save_response(response_text, dataset, filename):
    assert dataset in ("pandasEval", "numpyEval", "humanEval", "classEval"), ValueError
    ans_save_path = f"results/{dataset}"
    with open(ans_save_path + "/" + filename, "w") as f:
        f.write(response_text)

dir_path = 'log/numpyEval/'

filelist = os.listdir(dir_path)

for filename in filelist:
    with open(dir_path + filename, 'r') as f:
        json_file = json.load(f)
        save_response(eval(json_file['response body']).choices[0].message.content, 'numpyEval', json_file['test_case_name'])