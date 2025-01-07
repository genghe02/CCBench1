import pandas as pd
import json

from tqdm import tqdm

from prompt.alignment_comments_class_human import prompt_template


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

from datasets import load_dataset

ds = load_dataset("code_search_net", "python", trust_remote_code=True)

print(len(ds))

print(len(ds['train'].data))

print(type(ds['train'].data))

print(ds['train'].data.to_pydict())

# df = pd.read_csv('annotationStore.csv', encoding='utf-8', header=[0])
#
# python_data_list = []
# python_data_list_3 = []
# for i in range(df.shape[0]):
#     # print(df.iloc[i][0])
#     if df.iloc[i][0] == 'Python':
#         python_data_list.append(df.iloc[i][2])
#         if df.iloc[i][3] == 3:
#             python_data_list_3.append(df.iloc[i][2])
#
# print(len(python_data_list_3))
#
# print(len(set(python_data_list_3)))
#
# print(len(python_data_list))
#
# print(len(set(python_data_list)))
#
# subset_codesearchnet = []
#
# count = 0
#
# for filename in ['python_test_0.jsonl', "python_train_0.jsonl", "python_train_1.jsonl",
#                  "python_train_2.jsonl", "python_train_3.jsonl",
#                  "python_train_4.jsonl", "python_train_5.jsonl",
#                  "python_train_6.jsonl", "python_train_7.jsonl",
#                  "python_train_8.jsonl", "python_train_9.jsonl",
#                  "python_train_10.jsonl", "python_train_11.jsonl",
#                  "python_train_12.jsonl", "python_train_13.jsonl",
#                  "python_valid_0.jsonl" ]:
#
#     with open(filename, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#     test_urls = []
#     for item in lines:
#         test_urls.append(eval(item)['url'])
#         # print(eval(item)['url'])
#
#     in_num = 0
#     not_in_num = 0
#
#     for i, item in enumerate(tqdm(test_urls)):
#         if item not in python_data_list_3:
#             not_in_num += 1
#         else:
#             count += 1
#             js1 = eval(lines[i])
#             code1 = ' '.join(js1['code_tokens']).replace('\n', ' ')
#             code1 = ' '.join(code1.strip().split())
#             subset_codesearchnet.append({
#                 "docstring": js1['docstring'],
#                 "code": code1
#             })
#             in_num += 1
#     print(in_num, not_in_num)
#     save_json("subset_codesearchnet.json", subset_codesearchnet)
# print(count)