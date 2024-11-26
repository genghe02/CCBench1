"""
分析classEval以及numpyEval的长度以及可读性
"""

import json

from tools.json_open_close import open_json, save_json

def str_delete(target_str: str):
    while '#' in target_str or '\n' in target_str:
        target_str = target_str.replace('\n', '')
        target_str = target_str.replace('#', '')
    return target_str.strip()

def classeval_analysis():
    before_filepath = "classeval_comments.json"
    after_filepath = "aligned_classEval.json"
    before_length = 0
    after_length = 0
    before_file = open_json(before_filepath)
    after_file = open_json(after_filepath)
    before_results = []
    after_results = []
    for i in range(len(before_file)):
        before = before_file[i]
        after = after_file[i]
        _before = 0
        _after = 0
        for comment in before["comments"]:
            # 去掉开头以及结尾的'#‘和空格
            before_length += len(str_delete(comment))
            _before += len(str_delete(comment))
        for comment in after["comments"]:
            after_length += len(str_delete(comment))
            _after += len(str_delete(comment))
        before_results.append(_before)
        after_results.append(_after)
    # 保存长度文件
    print(before_length, after_length)
    save_json("classEval_original_length.json", before_results)
    save_json("classEval_aligned_length.json", after_results)
    print(before_results)

def humaneval_analysis():
    before_filepath = "humaneval_comments.json"
    after_filepath = "aligned_humanEval.json"
    before_length = 0
    after_length = 0
    before_file = open_json(before_filepath)
    after_file = open_json(after_filepath)
    before_results = []
    after_results = []
    for i in range(len(before_file)):
        before = before_file[i]
        after = after_file[i]
        _before = 0
        _after = 0
        for comment in before["comments"]:
            before_length += len(str_delete(comment))
            _before += len(str_delete(comment))
        for comment in after["comments"]:
            after_length += len(str_delete(comment))
            _after += len(str_delete(comment))
        before_results.append(_before)
        after_results.append(_after)
    print(before_length, after_length)
    save_json("humanEval_original_length.json", before_results)
    save_json("humanEval_aligned_length.json", after_results)

if __name__ == '__main__':
    # 53461 53554
    classeval_analysis()
    # 36231 30904
    humaneval_analysis()