import json

import numpy as np
import matplotlib.pyplot as plt


def open_json(filename):
    """
        :param filename: 你要打开的json文件名
        :return: None
    """
    f = open(filename)
    objects = json.load(f)
    f.close()
    return objects

def draw():
    # 数据
    n_groups = 4  # 每个簇中的柱子数量
    bar_width = 0.35  # 每个柱子的宽度
    opacity = 0.8

    # 生成一些随机数据
    data1 = [23, 35, 30, 35]
    data2 = [25, 32, 34, 20]

    data3 = [22, 30, 33, 25]
    data4 = [24, 35, 27, 30]

    index = np.arange(n_groups)  # 簇的位置

    # 创建两个子图
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # 绘制第一个子图，包含两个簇
    axs[0].bar(index, data1, bar_width, alpha=opacity, color='b', label='Group 1')
    axs[0].bar(index + bar_width, data2, bar_width, alpha=opacity, color='r', label='Group 2')

    # 设置第一个子图的标签和标题
    axs[0].set_xlabel('Category')
    axs[0].set_ylabel('Values')
    axs[0].set_title('Subplot 1: Grouped Bar Chart')
    axs[0].set_xticks(index + bar_width / 2)
    axs[0].set_xticklabels(['A', 'B', 'C', 'D'])
    axs[0].legend()

    # 绘制第二个子图，包含两个簇
    axs[1].bar(index, data3, bar_width, alpha=opacity, color='g', label='Group 3')
    axs[1].bar(index + bar_width, data4, bar_width, alpha=opacity, color='y', label='Group 4')

    # 设置第二个子图的标签和标题
    axs[1].set_xlabel('Category')
    axs[1].set_ylabel('Values')
    axs[1].set_title('Subplot 2: Grouped Bar Chart')
    axs[1].set_xticks(index + bar_width / 2)
    axs[1].set_xticklabels(['A', 'B', 'C', 'D'])
    axs[1].legend()

    # 自动调整布局
    plt.tight_layout()

    # 显示图形
    plt.show()


if __name__ == '__main__':
    # 一个子图是一个数据集
    # 一个柱是一个指标
    # 一个簇是一个模型
    p_pandas = []
    r_pandas = []
    f1_pandas = []
    for datasets in ("pandasEval", 'numpyEval'):
        for model in ('deepseek_v2', 'gpt_3_5', 'mistral', 'gpt_4'):
            filepath = f"results/evaluation/{model}/{datasets}/bert.json"
            average_p = []
            average_r = []
            average_f1 = []
            f = open_json(filepath)
            for item in f:
                average_p.append(item['score']['P'])
                average_r.append(item['score']['R'])
                average_f1.append(item['score']['F1'])
            p_pandas.append(sum(average_p) / len(average_p))
            r_pandas.append(sum(average_r) / len(average_r))
            f1_pandas.append(sum(average_f1) / len(average_f1))
    print(p_pandas)
    print(r_pandas)
    print(f1_pandas)
# deepseek-pandasEval:0.7007726708261093, 0.6812494645024291, 0.6878048538571537
# gpt_3_5-pandasEval:0.6624738801233839,0.7008417596911439,0.6780269019084402
# mistral-pandasEval: 0.5927339606355913,0.6706141077055789,0.6260603967279491
# gpt_4-pandasEval:0.6072292566889583,0.6892678047170734,0.6430349040149462
# deepseek-numpyEval: 0.6502024491803836,0.6497787556289571,0.6462650586545995
# gpt_3_5-numpyEval: 0.6392627007300311,0.6844953033003477,0.6570842962453861
# mistral-numpyEval: 0.5798301667270094,0.6614419044834552,0.614225287543665
# gpt_4-numpyEval:0.5988919023830112,0.6783610334490785, 0.6328017658526355
