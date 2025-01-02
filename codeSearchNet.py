import copy
import os

from tqdm import tqdm
from datasets import load_dataset
import textstat

from lora_data_builder import new_sample_data
from tools import save_json, open_json

def str_delete(target_str: str):
    while '#' in target_str or '\n' in target_str:
        target_str = target_str.replace('\n', '')
        target_str = target_str.replace('#', '')
    return target_str.strip()

# def code_handle(code_tokens)
template_json_sample = {
    "instruction": "",
    "input": "",
    "output": ""
}

prompts = []
#
for i in range(14):
    ds = open(f"data/codesearchnet/python_train_{i}.jsonl", 'r')

    # js = eval(ds.readlines()[0])
    #
    # # print(eval(lines)['code_tokens'])
    # code = ' '.join(js['code_tokens']).replace('\n', ' ')
    # code = ' '.join(code.strip().split())
    # print(code)

    lines = ds.readlines()

    ds1 = open("data/codesearchnet/python_valid_0.jsonl", 'r')

    lines1 = ds1.readlines()

    js1 = eval(lines[0])
    #
    # print(eval(lines)['code_tokens'])
    code1 = ' '.join(js1['code_tokens']).replace('\n', ' ')
    code1 = ' '.join(code1.strip().split())
    nl = ' '.join(js1['docstring_tokens']).replace('\n', '')
    nl = ' '.join(nl.strip().split())
    print(lines)



    for item in lines:
        new_sample_data = copy.deepcopy(template_json_sample)
        js = eval(item)
        code = ' '.join(js['code_tokens']).replace('\n', ' ')
        code = ' '.join(code.strip().split())

        nl = ' '.join(js['docstring_tokens']).replace('\n', '')
        nl = ' '.join(nl.strip().split())
        prompt = f"""Here is a Python code, and I hope you can generate a brief code summary based on the code.
        Note: You only need to generate a concise code summary, without saying any extra words or providing any extra content. 

        Here is the task you should finish:

        Code Snippet: 
        {code}


        Summary:"""
        new_sample_data['instruction'] = prompt
        new_sample_data['output'] = nl
        prompts.append(new_sample_data)

save_json('data/codesearchnet/finetune_prompts.json', prompts)

#
# print(ds['train'][0])
# print(len(ds['train']))
#
# print(len(ds['test']))
#
# print(len(ds['validation']))
#
# # train, test, validation
# print(len(ds['train']) + len(ds['test']) + len(ds['validation']))
#
# print(dict(ds['validation'][1])['func_documentation_string'])
#
# print(dict(ds['validation'][1])['whole_func_string'])
#
# print(dict(ds['validation'][1]))
#
# print(dict(ds['validation'][1])['func_code_string'])
# #
# print(dict(ds['train'][0]).keys())
# print(dict(ds['validation'][0]).keys())
# print(dict(ds['test'][0]).keys())
#
# print(dict(ds['test'][0]))


# print(dict(ds['validation'][0]['code_tokens']))
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

# def get_aligned_data():
#     datasets = ("train", "validation", "test")
#     # aligned
#     for dataset in datasets:
#         f = ds[dataset]
#         read_ability = {
#             "flesch_reading_ease": [],
#             "flesch_kincaid_grade": [],
#             "smog_index": [],
#             "coleman_liau_index": [],
#             "automated_readability_index": [],
#             "dale_chall_readability_score": [],
#             "difficult_words": [],
#             "linsear_write_formula": [],
#             "gunning_fog": []
#         }
#         for i in tqdm(range(len(f))):
#             comment = f[i]
#             length_all = []
#             length_all.append(len(comment['func_documentation_string']))
#             for key in read_ability.keys():
#                 func = eval(f"textstat.{key}")
#                 read_ability[key].append(func(comment['func_documentation_string']))
#         save_json(f"detailed_{dataset}_aligned_read_ability.json", read_ability)
#         for key in read_ability.keys():
#             read_ability[key] = sum(read_ability[key]) / len(read_ability[key])
#         save_json(f"{dataset}_aligned_read_ability.json", read_ability)
#
# get_aligned_data()