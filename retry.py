"""
retry mechanism

运行代码，获得正确，或者出错信息，然后将报错信息和之前的代码+注释，进行retry，做个小超参数试验.

**错误分析（**单个case的分析）：

统计出错情况

分析不同模型在不同任务上的表现。

瓶颈的问题
错误分析
insight：

标准的迭代的效用，瓶颈
标准迭代的错误，以及为什么不同的模型在不同的任务上会犯错
"""
import contextlib
import faulthandler

import gzip
import io
import json
import argparse
import json
import multiprocessing
import os
import platform
import re
import numpy as np
import signal
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict

import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftModel
from tqdm import tqdm
from tqdm import tqdm

from run_code import save_json
from tools import open_json


def read_jsonl(filepath):
    # 读取jsonl文件
    samples = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            samples.append(json.loads(line))
    return samples


def failed_examples_filter(examples):
    results = []
    for example in examples:
        if example['passed'] == False:
            results.append(example)
    return results

def judge_fail_type(response):
    if '\n' in response:
        response = response.replace('\n', '')
    if "test" in response or "Test" in response or "assert" in response or response == "failed: " or response == 'failed:':
        return "test_points"
    elif "timed out" in response:
        return "time_out"
    else:
        return "others"

def read_data():
    # Failed samples
    failed_examples = failed_examples_filter(samples)

    results_set = set()

    time_out = 0

    test_fail = 0

    others = 0

    retry_list = []
    # Read Dataset
    dataset = []
    file_path = "data/humaneval_comments_version/human-eval-v2-20210705.jsonl"
    f = open(file_path, 'r', encoding='utf-8')
    # f_lines = eval(f.read())
    f_lines = f.readlines()
    for line in f_lines:
        line = json.loads(line)
        dataset.append({
            "task_id": line['task_id'],
            "oral_prompt": line['prompt'],
            "prompt": prompt_p.format(line['prompt']),
            "test_code": line['test'],
            "entry_point": line['entry_point']
        })

    for example in failed_examples:
        for item in dataset:
            if item['task_id'] == example['task_id']:
                test_code = item['test_code']
                prompt = item['prompt']
                entry_point = item['entry_point']
                oral_prompt = item['oral_prompt']
        print(example)
        results_set.add(example['result'])
        if "test" in example['result'] or "Test" in example['result'] or "assert" in example['result'] or example[
            'result'] == "failed: ":
            test_fail += 1
            retry_list.append({
                "task_id": example['task_id'],
                "completion": example['completion'],
                "error_type": "test_points",
                "prompt_p": prompt,
                "test_code": test_code,
                "entry_point": entry_point,
                "oral_prompt": oral_prompt
            })
        elif "timed out" in example['result']:
            time_out += 1
            retry_list.append({
                "task_id": example['task_id'],
                "completion": example['completion'],
                "error_type": "time_out",
                "prompt_p": prompt,
                "test_code": test_code,
                "entry_point": entry_point,
                "oral_prompt": oral_prompt
            })
        else:
            retry_list.append({
                "task_id": example['task_id'],
                "completion": example['completion'],
                "error_type": "others",
                "prompt_p": prompt,
                "test_code": test_code,
                "entry_point": entry_point,
                "oral_prompt": oral_prompt
            })
            others += 1

    save_json(f"retry_errormsg_{model_name}.json", retry_list)


def load_model(model_name):
    if model_name == 'deepseek':
        tokenizer = AutoTokenizer.from_pretrained("deepseek/", local_files_only=True, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("deepseek/", local_files_only=True, trust_remote_code=True,
                                                     torch_dtype=torch.bfloat16).cuda()
    elif model_name == 'deepseek_finetune':
        model_path = 'deepseek/'
        lora_path = './output/deepseek_coder_v2/checkpoint-114'

        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16,
                                                     trust_remote_code=True).eval()

        # 加载lora权重
        model = PeftModel.from_pretrained(model, model_id=lora_path)
    elif model_name == 'mistral':
        # TODO
        model = None
        tokenizer = None
    else:
        model = None
        tokenizer = None
    return model, tokenizer


def large_model_api(model_name, prompt, model, tokenizer):
    if model_name == 'deepseek' or model_name == 'deepseek_finetune':
        messages = [
            {'role': 'user', 'content': prompt}
        ]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(
            model.device)
        # tokenizer.eos_token_id is the id of <｜end▁of▁sentence｜>  token
        outputs = model.generate(inputs, max_new_tokens=4096, do_sample=True, top_k=50, top_p=0.95,
                                 num_return_sequences=1,
                                 eos_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    elif model_name == 'gpt4o-mini':
        from openai import OpenAI

        client = OpenAI(api_key="sk-kKcQtH3Dh1DStsLW8e0fDe76F1B24dA2BaA1Ba6874D8E0Bf",
                        base_url="https://vip.apiyi.com/v1")

        try:

            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                stream=False,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            result = completion.choices[0].message.content
        except:
            print(completion.choices[0].message)
            return "[ERROR]"

        return result
    elif model_name == 'gpt3_5':
        from openai import OpenAI

        client = OpenAI(api_key="sk-kKcQtH3Dh1DStsLW8e0fDe76F1B24dA2BaA1Ba6874D8E0Bf",
                        base_url="https://vip.apiyi.com/v1")

        try:

            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                stream=False,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            result = completion.choices[0].message.content
        except:
            print(completion.choices[0].message)
            return "[ERROR]"

        return result
        # return None
    elif model_name == 'gpt_4':
        modelname = 'gpt-4-turbo'
        api_key = "sk-fyw5AXxFDRzRHzJ344240f868d964f04Ba90BfBe75A08a73"
        api_base = "https://api.fast-tunnel.one/v1"
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=api_base)

        # save the whole response body
        completion = client.chat.completions.create(
            model=f"{modelname}",
            messages=[
                {"role": "user", "content": f"{prompt}"}
            ]
        )
        return completion.choices[0].message.content
    elif model_name == 'mistral':
        from transformers import pipeline
        from huggingface_hub import login
        login(token='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')

        messages = [
            {"role": "user", "content": prompt},
        ]
        pipe = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2", max_new_tokens=5096)
        return pipe(messages)[0]['generated_text'][1]['content']
    elif model_name == 'deepseek_v2':
        from openai import OpenAI
        client = OpenAI(api_key='sk-5780d51e99ed49f7a0c0696089b7dc62', base_url="https://api.deepseek.com")

        response = client.chat.completions.create(
            model="deepseek-coder",
            messages=[
                {"role": "user", "content": prompt},
            ],
            stream=False
        )

        return response.choices[0].message.content
    # TODO:需要包含日志保存
    # 输入模型种类、prompt，返回模型的输出

    pass

def extract_python_code(text):
    """
    从给定的文本中提取所有位于 ```python 和 ``` 之间的代码块。

    参数:
        text (str): 要搜索的文本内容。

    返回:
        list: 包含所有提取出的Python代码块的列表。
    """
    # 定义正则表达式模式
    pattern = re.compile(r"```python\s*([\s\S]*?)```", re.IGNORECASE)
    # 使用findall方法提取所有匹配的代码块
    code_blocks = pattern.findall(text)

    return code_blocks

def write_jsonl(filename: str, data: dict, append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = 'ab'
    else:
        mode = 'wb'
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode='wb') as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode('utf-8'))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode('utf-8'))

if __name__ == '__main__':
    # Set model
    model_name = 'gpt3_5'
    assert model_name in ['deepseek_v2', 'gpt3_5', 'gpt4o-mini', 'mistral', 'gpt_4', "deepseek_7b", "deepseek_finetune"]

    if model_name == 'deepseek_v2':
        jsonl_path = "human-eval-master/human_eval/sample_deepseek_v2_results.jsonl"
    elif model_name == 'gpt3_5':
        jsonl_path = "human-eval-master/human_eval/sample_gpt3_5.jsonl_results.jsonl"
    elif model_name == 'gpt4o-mini':
        jsonl_path = "human-eval-master/human_eval/sample_deepseek_finetune_results.jsonl"
    elif model_name == 'mistral':
        jsonl_path = "human-eval-master/human_eval/sample_deepseek_finetune_results.jsonl"
    elif model_name == 'gpt_4':
        jsonl_path = "human-eval-master/human_eval/sample_gpt4.jsonl_results.jsonl"
    elif model_name == 'deepseek_7b':
        jsonl_path = "human-eval-master/human_eval/samples_deepseek_results.jsonl"
    elif model_name == 'deepseek_finetune':
        jsonl_path = "human-eval-master/human_eval/sample_deepseek_finetune_results.jsonl"

    samples = read_jsonl(jsonl_path)

    # Set prompt

    prompt_p = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

        Note: You only need to provide Python code, don't generate unnecessary content.


        ### Instruction:
        Create a Python script for this problem:
        {}

        ### Response:"""

    read_data()

    prompt_l = """{}
    
    The above is the code you generated last time, which has the following issues. I hope you can fix this problem as required.
    
    Note AGAIN: You only need to provide Python code, do NOT generate unnecessary content.
    
    ###Existing problems:
    {}
    
    ### Your repaired code:
    """

    # 设置三种出错类型的描述
    # test_points: 有未通过的测试样例
    # time_out: 超时
    # others: 语法错误，如有未引入的库等
    error_prompt = {
        "test_points": "The code did not pass some of the test points, which may be due to incomplete functional implementation, logical problems, etc",
        "time_out": "Code runtime timeout, there may be issues with dead loops, recursion, etc",
        "others": "There is a syntax error in the code, which may be due to undeclared libraries or other syntax issues"
    }

    retry_list = open_json(f"retry_errormsg_{model_name}.json")
    """
    {
        "task_id",
        "completion",
        "error_type",
        "prompt_p",
        "test_code",
    }
    """

    # TODO: 加载大模型(如果需要
    model, tokenizer = load_model(model_name)

    retry_log = []

    for _index, retry_item in enumerate(tqdm(retry_list)):
        MAX_RETRY_TIME = 10
        retry_single_log = {
            "task_id": retry_item['task_id'],
            "error": retry_item['error_type'],
            "retry_time": 0,
            "error_type": []
        }
        for i in range(MAX_RETRY_TIME):
            # TODO: 先设置prompt
            single_prompt = retry_item['prompt_p'] + prompt_l.format("\n```python\n" + retry_item['completion'] + '\n```\n',
                                                                     error_prompt[retry_item['error_type']])
            # TODO: 大模型重新生成
            response = large_model_api(model_name, single_prompt, model, tokenizer)

            if response == '[ERROR]':
                i -= 1
                continue
            # print(response)
            # print(response)
            # TODO: 测试修改后的代码
            try:
                write_jsonl("human-eval-master/human_eval/deepseek_v2_samples.jsonl", [{"task_id":retry_item['task_id'], "completion": extract_python_code(response)[0]}])
            except:
                i -= 1
                continue
            my_cmd = "cd human-eval-master && cd human_eval && python evaluate_functional_correctness.py deepseek_v2_samples.jsonl"
            out = os.popen(my_cmd)
            out_lines = out.readlines()
            print(out_lines)
            pass_or_not = eval(out_lines[-1])['pass@1']
            retry_single_log['retry_time'] += 1
            # 检查result，如果是1就不继续了，否则继续
            # sys.exit()
            if int(pass_or_not) == 1:
                retry_single_log['error_type'].append("Success")
                break
            else:
                # 检验错误类型
                error_type = judge_fail_type(out_lines[-2])
                print(error_type)
                retry_item['error_type'] = error_type
                pass
                # 调整prompt
                retry_item['completion'] = extract_python_code(response)[0]
                retry_single_log['error_type'].append(error_type)
        retry_log.append(retry_single_log)
        save_json(f"{model_name}_retry_log.json", retry_log)

