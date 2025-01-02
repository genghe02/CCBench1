import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# from evaluate import CompressionRadio


def str_delete(target_str: str):
    while '#' in target_str or '\n' in target_str:
        target_str = target_str.replace('\n', '')
        target_str = target_str.replace('#', '')
    return target_str.strip()

def get_data():
    # for pandasEval and numpyEval
    data1 = []
    for model in ('deepseek_v2', 'gpt_3_5', 'mistral', 'gpt_4'):
        for dataset in ("pandasEval", 'numpyEval'):
            temp1 = []
            dataset_path = f'prompt/aligned/{dataset}'
            # path of results
            result_path = f'results/{model}/{dataset}'
            dataset_filenames = os.listdir(dataset_path)
            result_filenames = os.listdir(result_path)
            result = []
            # pandasEval
            for i in tqdm(range(len(dataset_filenames))):
                dataset_filename = dataset_filenames[i]
                if not dataset_filename.endswith('.txt'):
                    continue
                dataset_file = open(dataset_path + '/' + dataset_filename, 'r').read()
                result_filename = dataset_filename.split('.')[0] + '.txt'
                result_file = open(result_path + '/' + result_filename, 'r')
                comment = ""
                for line in result_file:
                    if line.strip().startswith('#'):
                        comment += str_delete(line)
                temp1.append(len(str_delete(comment))/ len(str_delete(dataset_file)))
            data1.append(sum(temp1) / len(temp1))
            print(len(temp1), dataset, model, sum(temp1) / len(temp1))
    return [[data1[0], data1[2], data1[4], data1[6]], [data1[1], data1[3], data1[5], data1[7]]]

values1, values2 = get_data()

# 数据
categories = ['Deepseek-v2', 'GPT 3.5', 'Mistral', 'GPT 4']  # 每个柱的标签
# values1 = [10, 15, 7, 12]  # 第一组柱的高度
# values2 = [8, 10, 5, 9]    # 第二组柱的高度
colors = ['r', 'b']  # 不同的颜色

# 设置柱子宽度
bar_width = 0.35  # 每组柱子的宽度
x = np.arange(len(categories))  # 每个类别的X位置

# 绘制柱状图，使用偏移量来创建簇
plt.bar(x - bar_width/2, values1, color=colors[0], width=bar_width, label='PandasEval')
plt.bar(x + bar_width/2, values2, color=colors[1], width=bar_width, label='NumpyEval')


# 绘制柱状图

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# 添加标题和标签
plt.title('CompressionRadio for different LLMs')
plt.xlabel('Models')
plt.ylabel('CompressionRadio')
plt.legend(loc='lower right')
x = np.arange(len(categories))  # 每个柱的X位置
plt.xticks(x, categories)
# 显示图表
plt.show()
