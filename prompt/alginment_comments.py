import os

import sys

dataset = "numpyeval"

workspace = f"../datasets/{dataset}_comments_version/{dataset}_without_comments_json/"
filelist = os.listdir(workspace)

import json



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

for filename in filelist:
    if not filename.endswith(".json"):
        continue
    comment = ""
    with open(workspace + filename, 'r') as f:
        data = json.load(f)
        comment = data['comments']
    with open("../prompt/to_align/numpyEval/" + filename.split('.')[0] + '.txt', 'w') as f:
        f.write(prompt_template + comment)
    # with open("../prompt/aligned/numpyEval/" + filename.split('.')[0] + '.txt', 'w') as f:
    #     f.write('')


