# readAbility
import os
import textstat
from tools import open_json, save_json

def str_delete(target_str: str):
    while '#' in target_str or '\n' in target_str:
        target_str = target_str.replace('\n', '')
        target_str = target_str.replace('#', '')
    return target_str.strip()

def get_aligned_data(data):
    # aligned
    read_ability = {
        "flesch_reading_ease": [],
        "flesch_kincaid_grade": [],
        "smog_index": [],
        "coleman_liau_index": [],
        "automated_readability_index": [],
        "dale_chall_readability_score": [],
        "difficult_words": [],
        "linsear_write_formula": [],
        "gunning_fog": []
    }
    for item in data:
        for key in read_ability.keys():
            func = eval(f"textstat.{key}")
            read_ability[key].append(func(item))
    for key, value in read_ability.items():
        read_ability[key] = sum(value) / len(value)
    return read_ability

# 整理numpyeval
def numpyeval_handle():
    numpy_overall_readability = {
        "original": {},
        "aligned": {}
    }
    # 对齐之后
    file_list = os.listdir("./results/readAbality")
    for filename in file_list:
        file = open_json("./results/readAbality/" + filename)
        zb_name = filename.split('.')[0]
        for item in file:
            if item['model'] == 'Dataset' and item['dataset'] == 'numpyEval':
                numpy_overall_readability["aligned"][zb_name] = item[zb_name]
    # 对齐之前
    file_list = os.listdir("./data/numpyeval_comments_version/numpyeval_without_comments_json")
    comments = []
    for filename in file_list:
        file = open_json("./data/numpyeval_comments_version/numpyeval_without_comments_json/" + filename)
        comment = str_delete(file['comments'])
        comments.append(comment)
    numpy_overall_readability['original'] = get_aligned_data(comments)
    # 保存
    save_json("overall_numpyeval_readability.json", numpy_overall_readability)

# 整理pandasEval
def pandaseval_handle():
    pandas_overall_readability = {
        "original": {},
        "aligned": {}
    }
    # 对齐之后
    file_list = os.listdir("./results/readAbality")
    for filename in file_list:
        file = open_json("./results/readAbality/" + filename)
        zb_name = filename.split('.')[0]
        for item in file:
            if item['model'] == 'Dataset' and item['dataset'] == 'pandasEval':
                pandas_overall_readability["aligned"][zb_name] = item[zb_name]
    # 对齐之前
    file_list = os.listdir("./data/pandaseval_comments_version/pandaseval_without_comments_json")
    comments = []
    for filename in file_list:
        file = open_json("./data/pandaseval_comments_version/pandaseval_without_comments_json/" + filename)
        comment = file['comments']
        comments.append(str_delete(comment))
    pandas_overall_readability['original'] = get_aligned_data(comments)
    # 保存
    save_json("overall_pandaseval_readability.json", pandas_overall_readability)

# classEval
def classeval_handle():
    class_overall_readability = {
        "original": {},
        "aligned": {}
    }
    # 对齐之前
    file_dir = "./classeval_comments.json"
    file = open_json(file_dir)
    comments = []
    for item in file:
        for comment in item['comments']:
            comments.append(str_delete(comment))
    class_overall_readability['original'] = get_aligned_data(comments)
    # 对齐之后
    file_dir = "./aligned_classEval.json"
    comments = []
    file = open_json(file_dir)
    for item in file:
        for comment in item['comments']:
            comments.append(str_delete(comment))
    class_overall_readability['aligned'] = get_aligned_data(comments)
    save_json("overall_classEval_readability.json", class_overall_readability)

def humaneval_handle():
    human_overall_readability = {
        "original": {},
        "aligned": {}
    }
    # 对齐之前
    file_dir = "./humaneval_comments.json"
    file = open_json(file_dir)
    comments = []
    for item in file:
        for comment in item['comments']:
            comments.append(str_delete(comment))
    human_overall_readability['original'] = get_aligned_data(comments)
    # 对齐之后
    file_dir = "./aligned_humanEval.json"
    comments = []
    file = open_json(file_dir)
    for item in file:
        for comment in item['comments']:
            comments.append(str_delete(comment))
    human_overall_readability['aligned'] = get_aligned_data(comments)
    save_json("overall_humanEval_readability.json", human_overall_readability)

if __name__ == '__main__':
    numpyeval_handle()
    pandaseval_handle()
    classeval_handle()
    humaneval_handle()