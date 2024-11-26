import os

from tqdm import tqdm
from datasets import load_dataset
import textstat
from tools import save_json, open_json

def str_delete(target_str: str):
    while '#' in target_str or '\n' in target_str:
        target_str = target_str.replace('\n', '')
        target_str = target_str.replace('#', '')
    return target_str.strip()
#
ds = load_dataset("claudios/code_search_net", "python")
#
# # print(ds['train'][0])
#
# # train, test, validation
# print(ds)

# # print(dict(ds['train'][0])['func_documentation_string'])
#
# # print(dict(ds['train'][0]).keys())
#
# train_comments = []
# test_comments = []
# validation_comments = []
#
# for _index in tqdm(range(len(ds['train']))):
#     train_comments.append(len(str_delete(ds['train'][_index]['func_documentation_string'])))
#
# save_json("codeSearchNet_train_original_length.json", train_comments)
#
# for _index in tqdm(range(len(ds['test']))):
#     test_comments.append(len(str_delete(ds['test'][_index]['func_documentation_string'])))
#
# save_json("codeSearchNet_test_original_length.json", test_comments)
#
# for _index in tqdm(range(len(ds['validation']))):
#     validation_comments.append(len(str_delete(ds['validation'][_index]['func_documentation_string'])))
#
# save_json("codeSearchNet_validation_original_length.json", validation_comments)

# train_comments_length = open_json("codeSearchNet_train_original_length.json")
# print(sum(train_comments_length) / len(train_comments_length))
#
# test_comments_length = open_json("codeSearchNet_test_original_length.json")
# print(sum(test_comments_length))
#
# validation_comments_length = open_json("codeSearchNet_validation_original_length.json")
# print(sum(validation_comments_length))

def get_aligned_data():
    datasets = ("train", "validation", "test")
    # aligned
    for dataset in datasets:
        f = ds[dataset]
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
        for i in tqdm(range(len(f))):
            comment = f[i]
            length_all = []
            length_all.append(len(comment['func_documentation_string']))
            for key in read_ability.keys():
                func = eval(f"textstat.{key}")
                read_ability[key].append(func(comment['func_documentation_string']))
        save_json(f"detailed_{dataset}_aligned_read_ability.json", read_ability)
        for key in read_ability.keys():
            read_ability[key] = sum(read_ability[key]) / len(read_ability[key])
        save_json(f"{dataset}_aligned_read_ability.json", read_ability)

get_aligned_data()