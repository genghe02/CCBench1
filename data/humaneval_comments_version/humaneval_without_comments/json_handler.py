import json

import os

_l = os.listdir("../")
print(_l)
ans = []
for item in _l:
    print(item)
    if len(item.split('.')) == 2 and item.split('.')[1] == 'txt':
        ans.append(item)
for item in ans:
    new_file = open(f"{item}", "w")
    new_file.close()
# # 由于文件中有多行，直接读取会出现错误，因此一行一行读取
# file = open("HumanEval.jsonl", 'r', encoding='utf-8')
# papers = []
# for line in file.readlines():
#     dic = json.loads(line)
#     print(dic)
#     new_file_name = dic['task_id'].split('/')[0] + '_' + dic['task_id'].split('/')[1]
#     new_file = open(f"{new_file_name}.txt", "w")
#     new_file.write(dic['prompt'])
#     new_file.write(dic['canonical_solution'])
#     new_file.close()




