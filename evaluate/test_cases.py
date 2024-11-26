import os

import json


# 打开json格式文件
def open_json(filename):
    """
        :param filename: 你要打开的json文件名
        :return: None
    """
    f = open(filename)
    objects = json.load(f)
    f.close()
    return objects

# humaneval
for data in ("gpt_4", "deepseek_v2", "gpt_3_5", "mistral"):
    oral_dataset_path = "results/testcases/humaneval_branch.json"
    llm_path = f"results/testcases/cover/{data}_humaneval.json"
    oral = open_json(oral_dataset_path)
    llm = open_json(llm_path)
    _sum = 0
    _cha = 0
    for i in range(len(oral)):
        assert oral[i]['filename'] == llm[i]['filename']

        if oral[i]['branch'] == [] or llm[i]['cover'] == []:
            pass
        else:
            _sum += oral[i]['branch'][0]
            if oral[i]['branch'][0] - llm[i]['cover'][0] < 0:
                pass
            else:
                _cha += oral[i]['branch'][0] - llm[i]['cover'][0]
            pass
    print(data, (_sum - _cha) / _sum)

# classeval
# for datasets in ("gpt_4", "deepseek_v2", "gpt_3_5", "mistral"):
#     oral_dataset_path = "results/testcases/classeval_branch.json"
#     llm_path = f"results/testcases/cover/{datasets}_classeval.json"
#     oral = open_json(oral_dataset_path)
#     llm = open_json(llm_path)
#     _sum = 0
#     _cha = 0
#     for i in range(len(oral)):
#         assert oral[i]['filename'] == llm[i]['filename']
#         if oral[i]['branch'] == [] or llm[i]['cover'] == []:
#             pass
#         else:
#             length = len(llm[i]['cover'])
#             for j in range(length):
#                 _sum += oral[i]['branch'][0]
#                 assert len(llm[i]['cover']) == length
#                 if not len(oral[i]['branch']) == length:
#                     # print(oral[i]['filename'])
#                     continue

#                 if oral[i]['branch'][j] - llm[i]['cover'][j] < 0:
#                     pass
#                 else:
#                     _cha += oral[i]['branch'][j] - llm[i]['cover'][j]
#                 pass
#     print(datasets, (_sum - _cha) / _sum)

