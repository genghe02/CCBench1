import argparse

import matplotlib.pyplot as plt
import json

def open_json(filename):
    """
        :param filename: 你要打开的json文件名
        :return: None
    """
    f = open(filename)
    objects = json.load(f)
    f.close()
    return objects

def draw(y1, y2, y3, y4, y5, y6):
    import numpy as np
    import matplotlib.pyplot as plt

    start_0 = .0
    x = []
    while start_0 <= 1:
        x.append(start_0)
        start_0 += .01

    # 创建画布和多个子图，2 行 3 列
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # 绘制第一行的三张图
    axs[0, 0].plot(x, y1, label='P', color='b', linewidth=2.5)
    axs[0, 0].set_title('PandasEval-P')

    axs[0, 1].plot(x, y2, label='R', color='r', linewidth=2.5)
    axs[0, 1].set_title('PandasEval-R')

    axs[0, 2].plot(x, y3, label='F1', color='g', linewidth=2.5)
    axs[0, 2].set_title('PandasEval-F1')

    # 绘制第二行的三张图
    axs[1, 0].plot(x, y4, label='P', color='m', linewidth=2.5)
    axs[1, 0].set_title('NumpyEval-P')

    axs[1, 1].plot(x, y5, label='R', color='c', linewidth=2.5)
    axs[1, 1].set_title('NumpyEval-R')

    axs[1, 2].plot(x, y6, label='F1', color='y', linewidth=2.5)
    axs[1, 2].set_title('NumpyEval-F1')

    # 为每个子图添加图例、标签和网格
    for ax in axs.flat:
        ax.legend(loc='upper right', fontsize=10)
        ax.set_xlabel('Score', fontsize=12)
        ax.set_ylabel('Percent', fontsize=12)
        ax.grid(True, linestyle=':', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=10)

    # 在每一行上方添加标题
    fig.text(0.5, 0.92, 'PandasEval', ha='center', fontsize=16, fontweight='bold')
    fig.text(0.5, 0.48, 'NumpyEval', ha='center', fontsize=16, fontweight='bold')

    # 调整子图之间的间距，增加行间距
    plt.subplots_adjust(hspace=0.5)  # hspace 值越大，行距越大

    # 自动调整子图之间的间距和外部边界
    # plt.tight_layout(rect=[0, 0, 1, 0.9])

    plt.show()


def bert_score_handle(score_base_dir_paths, datasets):
    start_0 = .0
    x = []
    y = []
    while start_0 <= 1:
        x.append(start_0)
        start_0 += .01
    for score_base_dir_path in score_base_dir_paths:
        score_file = open_json(score_base_dir_path)
        start_0 = 0
        y1, y2, y3 = [], [], []
        while start_0 <= 1:
            num_P = 0
            num_R = 0
            num_F1 = 0
            for item in score_file:
                item = item['score']
                if item['P'] < start_0:
                    num_P += 1
                if item['R'] < start_0:
                    num_R += 1
                if item['F1'] < start_0:
                    num_F1 += 1
            y1.append(num_P / len(score_file))
            y2.append(num_R / len(score_file))
            y3.append(num_F1 / len(score_file))
            start_0 += .01
        y.append(y1)
        y.append(y2)
        y.append(y3)
    draw(*y)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--m", type=str, dest='model_name', required=True)
    parser.add_argument("--i", type=str, dest="indicator", required=True)
    args = parser.parse_args()
    model_name, indicator = args.model_name, args.indicator
    assert indicator in ('bert', 'rouge'), ValueError
    # assert dataset in ('pandasEval', 'numpyEval', 'humanEval', 'classEval'), ValueError
    assert model_name in ('deepseek_v2', 'gpt_3_5', 'gpt4omini', 'lamma', 'mistral', 'gpt_4'), ValueError
    score_base_dir_paths = []
    datasets = ('pandasEval', 'numpyEval')
    for dataset in datasets:
        score_base_dir_paths.append(f"results/evaluation/{model_name}/{dataset}/{indicator}.json")
    scores_message = bert_score_handle(score_base_dir_paths, datasets)
