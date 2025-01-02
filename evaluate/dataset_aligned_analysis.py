import os

import numpy as np
from scipy.stats import gaussian_kde
from textstat import textstat

import json


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

def length_analysis_before_and_after(dataset_name):
    """
    计算 (对齐前 / 对齐后) / 对齐前
    :param dataset_name:
    :return:
    """
    before_length = open_json(f"{dataset_name}_original_length.json")
    after_length = open_json(f"{dataset_name}_aligned_length.json")
    ans = []
    for i in range(len(before_length)):
        ans.append(after_length[i] / before_length[i])
    return ans

def get_aligned_data():
    datasets = ("pandasEval", "numpyEval", "humanEval", "classEval")
    # aligned
    for dataset in datasets:
        length_all = []
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
        if dataset in ("pandasEval", "numpyEval"):
            original_dataset_dir_path = f"prompt/aligned/{dataset}/"
            files = os.listdir(original_dataset_dir_path)
            for filename in files:
                if not filename.endswith(".txt"):
                    continue
                f = open(original_dataset_dir_path + filename, 'r').read()
                length_all.append(len(f))
                for key in read_ability.keys():
                    func = eval(f"textstat.{key}")
                    read_ability[key].append(func(f))
            save_json(f"{dataset}_aligned_length.json", length_all)
            save_json(f"{dataset}_aligned_read_ability.json", read_ability)
            print(read_ability)
            for key in read_ability.keys():
                read_ability[key] = sum(read_ability[key]) / len(read_ability[key])
            for key, value in read_ability.items():
                print(key, value)

def get_original_data():
    datasets = ("pandasEval", "numpyEval", "humanEval", "classEval")
    # original
    for dataset in datasets:
        length_all = []
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
        if dataset in ("pandasEval", "numpyEval"):
            original_dataset_dir_path = f"datasets/{dataset.lower()}_comments_version/{dataset.lower()}_without_comments_json/"
            files = os.listdir(original_dataset_dir_path)
            for filename in files:
                if not filename.endswith(".json"):
                    continue
                f = open_json(original_dataset_dir_path + filename)
                length_all.append(len(f['comments']))
                for key in read_ability.keys():
                    func = eval(f"textstat.{key}")
                    read_ability[key].append(func(f['comments']))
            save_json(f"{dataset}_original_length.json", length_all)
            save_json(f"{dataset}_read_ability.json", read_ability)
            for key in read_ability.keys():
                read_ability[key] = sum(read_ability[key]) / len(read_ability[key])
            for key, value in read_ability.items():
                print(key, value)


def differential_entropy(data, bw_method='scott'):
    """
    计算连续数据的差异熵
    :param data: 输入数据（浮点数）
    :param bw_method: 核密度估计的带宽（默认'scott'）
    :return: 差异熵
    """
    # 核密度估计
    kde = gaussian_kde(data, bw_method=bw_method)

    # 核密度估计函数
    def kde_log_prob(x):
        p_x = kde.evaluate(x)
        return p_x * np.log(p_x + 1e-5)

    # 创建一个范围（用于数值积分）
    x_grid = np.linspace(min(data), max(data), 1000)

    # 差异熵的数值积分
    entropy = -np.trapz(kde_log_prob(x_grid), x_grid)
    return entropy


def calculate_variance_and_std(data):
    """
    计算数据的方差和标准差
    :param data: 输入数据（浮点数或整数）
    :return: 方差和标准差
    """
    variance = np.var(data)  # 方差
    std_deviation = np.std(data)  # 标准差
    return variance, std_deviation

def cal_mess(dataset_name):
    before_length = open_json(f"{dataset_name}_original_length.json")
    before_length = np.array(before_length)
    print(f"Before: {differential_entropy(before_length)} {calculate_variance_and_std(before_length)}")
    if dataset_name in ("pandasEval", "numpyEval", "humanEval", "classEval"):
        after_length = open_json(f"{dataset_name}_aligned_length.json")
        after_length = np.array(after_length)
        print(f"After: {differential_entropy(after_length)} {calculate_variance_and_std(after_length)}")

def char_length(dataset_name):
    before_length = open_json(f"{dataset_name}_original_length.json")
    after_length = open_json(f"{dataset_name}_aligned_length.json")
    ans1 = sum([item for item in before_length])
    ans2 = sum([item for item in after_length])
    print(ans1, ans2)
    print(ans1 - ans2)
    print(abs(ans1 - ans2) / len(before_length))

def read_ability(dataset_name):
    pass

if __name__ == '__main__':
    # dataset_name = "humanEval"

    dataset_name = "codeSearchNet_test"

    # 代码长度分析
    ### OUTPUT
    ### 对齐前长度 对齐后长度
    ### 长度差
    ### 平均每个case减少了多少长度的字符
    # char_length(dataset_name)

    print("-------------------")

    # 代码长度方差分析
    ### OUTPUT
    ### Before: 熵 (方差、标准差)
    ### Before: 熵 (方差、标准差)
    cal_mess(dataset_name)






