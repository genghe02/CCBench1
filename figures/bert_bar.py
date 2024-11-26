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


def getdata():
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


# 数据
n_groups = 4  # 每个簇中的柱子数量
bar_width = 0.2  # 每个柱子的宽度
opacity = 1

# 生成一些随机数据
# bert score
data1 = [0.7007726708261093, 0.6624738801233839, 0.5927339606355913, 0.6072292566889583]
data2 = [0.6812494645024291, 0.7008417596911439, 0.6706141077055789, 0.6892678047170734]
data3 = [0.6878048538571537, 0.6780269019084402, 0.6260603967279491, 0.6430349040149462]

data4 = [0.6502024491803836, 0.6392627007300311, 0.5798301667270094, 0.5988919023830112]
data5 = [0.6497787556289571, 0.6844953033003477, 0.6614419044834552, 0.6783610334490785]
data6 = [0.6462650586545995, 0.6570842962453861, 0.614225287543665, 0.6328017658526355]

# rouge
# data1 = [0.30293148514851476, 0.23920475247524753, 0.11176584158415845, 0.12679554455445544]
# data2 = [0.23909445544554458, 0.3157696039603962, 0.30854079207920787, 0.38789534653465335]
# data3 = [0.22737564356435638, 0.2331091089108911, 0.12997930693069304, 0.15753782178217826]
#
# data4 = [0.20345653465346533, 0.18758158415841586, 0.09162564356435648, 0.112110198019802]
# data5 = [0.22590178217821794, 0.2673766336633662, 0.2845875247524753, 0.2841366336633663]
# data6 = [0.17374069306930692, 0.17995970297029698, 0.09981980198019799, 0.13104752475247522]

# data3 = [22, 30, 33, 25]
# data4 = [24, 35, 27, 30]

index = np.arange(n_groups)  # 簇的位置

# 创建两个子图
fig, axs = plt.subplots(1, 2, figsize=(12, 10))

# 绘制第一个子图，包含两个簇
axs[0].bar(index, data1, bar_width, alpha=opacity, color='r', label='P')
axs[0].bar(index + bar_width, data2, bar_width, alpha=opacity, color='g', label='R')
axs[0].bar(index + 2 * bar_width, data3, bar_width, alpha=opacity, color='b', label='F1')

# 绘制第二个子图，包含两个簇
axs[1].bar(index, data4, bar_width, alpha=opacity, color='r', label='P')
axs[1].bar(index + bar_width, data5, bar_width, alpha=opacity, color='g', label='R')
axs[1].bar(index + 2 * bar_width, data6, bar_width, alpha=opacity, color='b', label='F1')

# 设置第一个子图的标签和标题
axs[0].set_xlabel('Model')
axs[0].set_ylabel('Score')
axs[0].set_title('PandasEval')
axs[0].set_xticklabels(['deepseek_v2', 'gpt_3_5', 'mistral', 'gpt_4'])
axs[0].set_xticks(index + bar_width)
axs[0].legend(loc='lower right')

# 设置第二个子图的标签和标题
axs[1].set_xlabel('Model')
axs[1].set_ylabel('Score')
axs[1].set_title('NumpyEval')
axs[1].set_xticklabels(['deepseek_v2', 'gpt_3_5', 'mistral', 'gpt_4'])
axs[1].set_xticks(index + bar_width)
axs[1].legend(loc='lower right')

# 自动调整布局
plt.tight_layout()

# 显示图形
plt.show()
