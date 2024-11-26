"""
对齐classEval以及HumanEval
"""

import json

import argparse


# 打开json格式文件
def open_json(filename):
    """
        :param filename: 你要打开的json文件名
        :return: None
    """
    f = open(filename, encoding='utf-8')
    objects = json.load(f)
    f.close()
    return objects


# 保存json格式文件
def save_json(filename, objects):
    """
        :param filename: 你要保存的文件名
        :param objects: 你要保存的内容
        :return: None

        Warning：会覆盖原有内容，谨慎！
    """
    f = open(filename, 'w')
    json.dump(objects, f)
    f.close()


prompt_template = """You are an expert in Python code annotation.
Next, I will provide you with some code comments in different styles, some of which may be extremely colloquial. I hope you can help me turn them into code comments with a unified style of 'Requirements'.
Example 1:
User input: # How to drop rows of Pandas DataFrame whose value in a certain column is NaN
Your output:
# Drop rows of Pandas DataFrame whose value in a certain column is NaN
Example 2:
User input:
# list_of_lists format: [header, [row1], [row2], ...]
# header format: [column1, column2, ...]
# row format: [value1, value2, ...]
# How to convert list to dataframe?
# Return the dataframe
Your output:
# Given the list_of_lists，  header， row ， convert list to dataframe and return it。
Example 3:
User input:
# I need to remain the rows where line_num is not equal to 0.  What's the most efficient way to do it?
# it should be as simple as:
Your output:
# Remain the rows where line_num is not equal to 0 by the most efficient way.
Next, I will provide you with some annotations. You need to follow my requirements and examples to unify the annotation style into a non colloquial and formal style.

Here is the comment which you need handle：
"""

def align_classeval():
    # classeval注释的路径
    workspace = f"classeval_comments.json"
    f = open_json(workspace)
    results = []
    for item in f:
        comments = item["comments"]
        comment_message = {
            "filename": item["filename"],
            "comments_prompt": []
        }
        comments_prompt = []
        for comment in comments:
            comments_prompt.append(prompt_template + '\n' + comment)
        comment_message["comments_prompt"] = comments_prompt
        results.append(comment_message)
    save_json("classeval_align_prompt.json", results)

def align_humaneval():
    workspace = f"humaneval_comments.json"
    f = open_json(workspace)
    results = []
    for comment in f:
        comment_message = {
            "filename": comment["filename"],
            "comments_prompt": prompt_template + '\n' + comment['comments'][0]
        }
        results.append(comment_message)
    save_json("humaneval_align_prompt.json", results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--d", type=str, dest='dataset', required=True)
    args = parser.parse_args()
    dataset_name = args.dataset
    assert dataset_name in ("classEval", "humanEval")
    if dataset_name == "classEval":
        align_classeval()
    else:
        align_humaneval()



