"""
计算classEval以及humaneval的可读性
"""
from textstat import textstat

from class_human_eval_length import str_delete
from tools import open_json, save_json


def get_original_data(dataset):
    # 获取数据
    if dataset == "classEval":
        # read classeval file
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
                before_length += cal_read_ability(str_delete(comment))
                _before += cal_read_ability(str_delete(comment))
            for comment in after["comments"]:
                after_length += cal_read_ability(str_delete(comment))
                _after += cal_read_ability(str_delete(comment))
            _before /= len(before["comments"])
            _after /= len(after["comments"])
            before_results.append(_before)
            after_results.append(_after)
        save_json("classeval_readability_before.json", before_results)
        save_json("classeval_readability_aligned.json", after_results)
        pass
    else:
        # read humaneval file
        before_filepath = "humaneval_comments.json"
        after_filepath = "aligned_humaneval.json"
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
                before_length += cal_read_ability(str_delete(comment))
                _before += cal_read_ability(str_delete(comment))
            _before /= len(before["comments"])
            for comment in after["comments"]:
                after_length += cal_read_ability(str_delete(comment))
                _after += cal_read_ability(str_delete(comment))
            _after /= len(after["comments"])
            before_results.append(_before)
            after_results.append(_after)
        save_json("humaneval_readability_before.json", before_results)
        save_json("humaneval_readability_aligned.json", after_results)
        pass
        pass

def cal_read_ability(text):
    # 计算可读性
    flesch_score = textstat.flesch_reading_ease(text)
    return flesch_score

if __name__ == '__main__':
    get_original_data("classEval")