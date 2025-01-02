"""
构建humaneval solution
"""
import gzip
import json
import os
import re

from sympy.physics.units import percent

from tools import open_json

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

def handle_solution_code(code):
    pattern = r'^```(?:python)?\n([\s\S]*?)\n```$'
    match = re.match(pattern, code.strip(), re.MULTILINE)
    if match:
        return match.group(1)
    else:
        # 如果不匹配，返回原始文本
        return code

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

if __name__ == '__main__':
    file_path = "deepseek_v2_humaneval_code.json"
    f = open_json(file_path)
    samples = []
    # I = """<s> [INST] Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n        Note: You only need to provide Python code, don't generate unnecessary content.\n\n\n        ### Instruction:\n        Create a Python script for this problem:\n        \n\ndef same_chars(s0: str, s1: str):\n    \"\"\"\n    Check if two words have the same characters.\n    >>> same_chars('eabcdzzzz', 'dddzzzzzzzddeddabc')\n    True\n    >>> same_chars('abcd', 'dddddddabc')\n    True\n    >>> same_chars('dddddddabc', 'abcd')\n    True\n    >>> same_chars('eabcd', 'dddddddabc')\n    False\n    >>> same_chars('abcd', 'dddddddabce')\n    False\n    >>> same_chars('eabcdzzzz', 'dddzzzzzzzddddabc')\n    False\n    \"\"\"\n\n\n        ### Response: [/INST] """
    for item in f:
        samples.append({
            "task_id": item["task_id"],
            "completion": handle_solution_code(item["solution_code"])
        })
        # text = item["solution_code"]
        # text = text.split('[/INST]')[1]
        # text = text.strip()
        # if extract_python_code(text) == []:
        #     text = text
        # else:
        #     text = extract_python_code(text)[0]
        # samples.append({
        #     "task_id": item["task_id"],
        #     "completion": text
        # })
    print(samples)
    write_jsonl("sample_deepseek_v2.jsonl", samples)
    pass