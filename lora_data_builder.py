"""
示例json格式
[
    {
        "instruction": "小姐，别的秀女都在求中选，唯有咱们小姐想被撂牌子，菩萨一定记得真真儿的——",
        "input": "",
        "output": "嘘——都说许愿说破是不灵的。"
    }
    ....
]
"""
import copy
import os
import sys

from tools import open_json, save_json

template_json_sample = {
    "instruction": "",
    "input": "",
    "output": ""
}

json_file = []

# 定义数据集路径
# prompt
# prompt_path = open("prompt/test_prompts/finetune_prompt_comment.txt")
# prompt_prefix = prompt_path.read()
# prompt_prefix = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
#
#         Note: You only need to provide Python code, don't generate unnecessary content.
#
#
#         ### Instruction:
#         Create a Python script for this problem:
#         {}
#
#         ### Response:"""
prompt_prefix = """You are an expert in Python annotation writing. Here is a simple Python code, I need you to add comments in the<>area.
For example:<description for whole class>:This is a class for parsing command line arguments to a dictionary.
<description for whole function>：Filter the incoming request based on certain rules and conditions.
<Requirements for the code>:Provide a brief explanation of the code following this tag, taking into account the context.

Here are an example:

User:
```python
import numpy as np

a = np.array([1, 1, 1, 1, 1, 2, 2, 2, 3, 4, 3, 4, 3, 4, 3, 4, 5, 5, 5])
<Requirements for the code>
result = np.where(a[1:] != a[:-1])[0]
```

What you should respond:
```python
import numpy as np

a = np.array([1, 1, 1, 1, 1, 2, 2, 2, 3, 4, 3, 4, 3, 4, 3, 4, 5, 5, 5])
# Find each index where the value changes in an efficient way using numpy by comparing each element with its neighbor and utilizing np.where(condition).
result = np.where(a[1:] != a[:-1])[0]
```

Next, please complete the annotation writing task:

"""

def get_output(comment_path, code_with_tokens_list, filename):
    f = open_json(comment_path)
    tokens = [
        "<description for whole class>",
        "<description for whole function>",
        "<description for all parameters>",
        "<description for return statement>",
        "<some test cases for the function>",
        "<Requirements for the code>"
    ]
    for item in f:
        if item['filename'] == filename:
            comments = item['comments']
    count = 0
    new_code = ""
    print(filename)
    for line in code_with_tokens_list:
        for token in tokens:
            if token in line:
                if token not in ("<description for whole class>", "<description for whole function>", '<Requirements for the code>'):
                    line = line.replace(token, '')
                else:
                    try:
                        line = line.replace(token, comments[count])
                    except IndexError as e:
                        print(e, '==================================')
                        return None
                    count += 1
                continue
        new_code += line
    return new_code




# 构建任务
# classeval
file_path = "data/classeval_comments_version/code_without_comments"
file_names = os.listdir(file_path)
for file_name in file_names:
    if not file_name.endswith(".txt"):
        continue
    f = open(os.path.join(file_path, file_name), "r")
    new_sample_data = copy.deepcopy(template_json_sample)
    code_with_tokens_list = f.readlines()
    f.close()
    f = open(os.path.join(file_path, file_name), "r")
    code_with_tokens = f.read()
    f.close()
    new_sample_data["instruction"] = prompt_prefix + '\n' + code_with_tokens
    # 把注释融合进代码里
    # 获得对齐后的注释
    aligned_path = "aligned_classEval.json"
    output = get_output(aligned_path, code_with_tokens_list, file_name)
    new_sample_data['output'] = output
    json_file.append(new_sample_data)

# humaneval
# file_path = "data/humaneval_comments_version/humaneval_without_comments"
# file_names = os.listdir(file_path)
# for file_name in file_names:
#     if not file_name.endswith(".txt"):
#         continue
#     f = open(os.path.join(file_path, file_name), "r")
#     new_sample_data = copy.deepcopy(template_json_sample)
#     code_with_tokens_list = f.readlines()
#     code_with_tokens = f.read()
#     new_sample_data["instruction"] = prompt_prefix + '\n' + code_with_tokens
#     # 把注释融合进代码里
#     # 获得对齐后的注释
#     aligned_path = "aligned_humanEval.json"
#     output = get_output(aligned_path, code_with_tokens_list, file_name)
#     if output is None:
#         continue
#     new_sample_data['output'] = output
#     json_file.append(new_sample_data)

def get_output_pandas_numpy(comment_path, code_with_tokens_list, filename):
    f = open(comment_path + filename, 'r')
    tokens = [
        "<description for whole class>",
        "<description for whole function>",
        "<description for all parameters>",
        "<description for return statement>",
        "<some test cases for the function>",
        "<Requirements for the code>"
    ]
    comments = f.readlines()
    count = 0
    new_code = ""
    print(filename)
    for line in code_with_tokens_list:
        for token in tokens:
            if token in line:
                if token not in (
                "<description for whole class>", "<description for whole function>", '<Requirements for the code>'):
                    line = line.replace(token, '')
                else:
                    try:
                        line = line.replace(token, comments[count])
                    except IndexError as e:
                        print(e, '==================================')
                        return None
                    count += 1
                continue
        new_code += line
    return new_code

# pandasEval
file_path = "data/pandaseval_comments_version/pandaseval_without_comments"
file_names = os.listdir(file_path)
for file_name in file_names:
    if not file_name.endswith(".txt"):
        continue
    f = open(os.path.join(file_path, file_name), "r")
    new_sample_data = copy.deepcopy(template_json_sample)
    code_with_tokens_list = f.readlines()
    f.close()
    f = open(os.path.join(file_path, file_name), "r")
    code_with_tokens = f.read()
    f.close()
    new_sample_data["instruction"] = prompt_prefix + '\n' + code_with_tokens
    # 把注释融合进代码里
    # 获得对齐后的注释
    aligned_path = "prompt/aligned/pandasEval/"
    output = get_output_pandas_numpy(aligned_path, code_with_tokens_list, file_name)
    if output is None:
        continue
    new_sample_data['output'] = output
    json_file.append(new_sample_data)

# numpyEval
file_path = "data/numpyEval_comments_version/numpyEval_without_comments"
file_names = os.listdir(file_path)
for file_name in file_names:
    if not file_name.endswith(".txt"):
        continue
    f = open(os.path.join(file_path, file_name), "r")
    new_sample_data = copy.deepcopy(template_json_sample)
    code_with_tokens_list = f.readlines()
    f.close()
    f = open(os.path.join(file_path, file_name), "r")
    code_with_tokens = f.read()
    f.close()
    new_sample_data["instruction"] = prompt_prefix + '\n' + code_with_tokens
    # 把注释融合进代码里
    # 获得对齐后的注释
    aligned_path = "prompt/aligned/numpyEval/"
    output = get_output_pandas_numpy(aligned_path, code_with_tokens_list, file_name)
    if output is None:
        continue
    new_sample_data['output'] = output
    json_file.append(new_sample_data)

save_json("data/lora_data_without_humaneval.json", json_file)