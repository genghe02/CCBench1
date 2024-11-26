# 算钱
import os
import sys

from tools import open_json

filepaths = [
    "log/classEval/",
    "log/humanEval/",
    "log/numpy_alignment_log/",
    "log/numpyEval/",
    "log/pandas_alignment_log/",
    "log/pandasEval/",
]

_sum = 0
t = 1
in_token = 0
out_token = 0
total_token = 0
for filepath in filepaths:
    files = os.listdir(filepath)
    t += 1
    for file in files:
        if file.endswith(".json"):
            f = open_json(filepath + file)
            try:
                _sum += f["response body"]["consume"]
                in_token += f["response body"]["usage"]["prompt_tokens"]
                out_token += f["response body"]["usage"]["completion_tokens"]
                total_token += f["response body"]["usage"]["total_tokens"]
            except TypeError:
                pass
            except:
                print(filepath + file)
print(f"prompt_tokens: {in_token}")
print(f"completion_tokens: {out_token}")
print(f"total_tokens: {total_token}")
print(f"cost: {_sum}")