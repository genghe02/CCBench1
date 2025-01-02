import json
from random import shuffle

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

# 获取pandasEval的注释长度
comment_length = open_json('pandasEval_original_length.json')
aligned_comment_length = open_json('pandasEval_aligned_length.json')

from sklearn.preprocessing import MinMaxScaler
import numpy as np

f_data = []
f_aligned_data = []

for length in comment_length:
    f_data.append([length])

for aligned_length in aligned_comment_length:
    f_aligned_data.append([aligned_length])

data = np.array(f_data)
data_aligned = np.array(f_aligned_data)

# 创建 MinMaxScaler 对象，指定数据范围为 [0, 1]
scaler1 = MinMaxScaler(feature_range=(0, 1))

print(data)

# 使用 MinMaxScaler 进行归一化处理
# data_normalized1 = scaler1.fit_transform(data)
data_normalized1 = data

shape_x = 30
shape_y = 30

_index_roller = [i for i in range(len(data_normalized1))]
shuffle(_index_roller)
_index_roller = _index_roller[:shape_x]

print(data_normalized1)



# 生成随机数据
data = np.zeros([shape_x, shape_y])

for i in range(shape_x):
    for j in range(i, shape_y):
        temp = abs(data_normalized1[_index_roller[i]] - data_normalized1[_index_roller[j]]) / data_normalized1[_index_roller[i]]
        data[i][j] = temp
        data[j][i] = temp



# 创建figure
fig, ax = plt.subplots(figsize=(16, 12))

# 使用pcolormesh绘制热力图，并设置每个小方格的边界线
cax = ax.pcolormesh(data, edgecolors='black', linewidth=2, cmap='coolwarm')

# 添加颜色条
cbar = fig.colorbar(cax, ax=ax, shrink=0.8)

# 在每个格子中标注数值
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        text = ax.text(j + 0.5, i + 0.5, f'{data[i, j]:.2f}',
                       ha='center', va='center', color='black', fontsize=10)

# 设置标题和坐标轴标签
ax.set_title('Fancy Heatmap with Borders', fontsize=20)
ax.set_xlabel('X-axis Label', fontsize=15)
ax.set_ylabel('Y-axis Label', fontsize=15)

# 调整坐标轴刻度
ax.tick_params(axis='x', labelsize=12, rotation=45)
ax.tick_params(axis='y', labelsize=12)

# 显示图像
plt.tight_layout()
plt.savefig('dataset_heatmap.png')
