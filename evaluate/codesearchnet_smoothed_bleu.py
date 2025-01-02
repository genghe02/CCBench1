# from tqdm import tqdm
#
# from evaluate.dataset_aligned_analysis import open_json
# from smoothed_bleu import get_score
# import nltk
# # 确保nltk的资源已经被下载
# nltk.download('punkt')
# nltk.download('punkt_tab')
# ds = open("../data/codesearchnet/python_test_0.jsonl", 'r', encoding='utf-8')
# results = open_json("comment_generated.json")
# lines = ds.readlines()
# scores = []
# for i, line in enumerate(tqdm(lines)):
#     # try:
#         js = eval(lines[i])
#         nl = ' '.join(js['docstring_tokens']).replace('\n', '')
#         nl = ' '.join(nl.strip().split())
#         # print(nl, results[i])
#         score = get_score(results[i], nl)
#         scores.append(score)
#     # except:
#     #     print('1')
#     #     continue
# print(sum(scores) / len(scores))
# for item in scores:
#     print(item)


### TEST
import sacrebleu

from tools import open_json

def str_delete(target_str: str):
    while '#' in target_str or '\n' in target_str:
        target_str = target_str.replace('\n', '')
        target_str = target_str.replace('#', '')
    return target_str.strip()


def compute_smoothed_bleu4(reference_file, hypothesis_file):
    """
    使用 sacrebleu 计算 Smoothed BLEU-4 分数（范围 ~0-100）。
    参数:
        reference_file (str): 参考文本文件路径，每行一条参考
        hypothesis_file (str): 预测文本文件路径，每行一条预测
    返回:
        float: BLEU 分数（0~100），越高越好
    """
    # 读取参考文本
    references = open_json(reference_file)
    # with open(reference_file, 'r', encoding='utf-8') as f:
    #     references = [line.strip() for line in f if line.strip()]
    #
    # # references = references[:1]
    #
    # new_references = []
    #
    # for item in references:
    #     js = eval(item)
    #     # nl = ' '.join(js['docstring_tokens']).replace('\n', '')
    #     # nl = ' '.join(nl.strip().split())
    #     nl = js['docstring']
    #     new_references.append(nl)
    #
    # references = new_references

    # new_references = new_references[:1]

    # 读取预测文本
    # with open(hypothesis_file, 'r', encoding='utf-8') as f:
    #     hypotheses = [line.strip() for line in f if line.strip()]

    f = open_json(hypothesis_file)
    hypotheses = [item for item in f]

    # hypotheses = hypotheses[:1]

    # print(new_references)

    # print(hypotheses)

    # 确保两者长度一致
    assert len(references) == len(hypotheses), (
        f"参考({len(references)}行)与预测({len(hypotheses)}行)数量不匹配"
    )

    # sacrebleu 的 corpus_bleu 第二个参数可以传“二维列表”，以支持多条参考；
    # 这里我们只用一条参考，就包一层[]即可：
    _a = []
    for i, item in enumerate(hypotheses):
        print("Generated:", hypotheses[i])
        print("Dataset:",references[i])
        # bleu = sacrebleu.corpus_bleu([hypotheses[i]], [[references[i]]])
        bleu = sacrebleu.corpus_bleu([references[i]], [[hypotheses[i]]])
        print(bleu.score)
        print('=' * 20)

        if bleu.score < 15.0:
            continue

        _a.append(bleu.score)
        # print(bleu.score)
    # sacrebleu 返回一个对象，其中 bleu.score 就是以 0~100 为区间的数值
    # print(sum(_a) / len(_a))
    # print(_a)
    return sum(_a) / len(_a)

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 初始化平滑函数
smooth_fn = SmoothingFunction().method1  # 选择一种平滑方法，例如method1


def smoothed_bleu4(str1, str2):
    """
    计算两个字符串的平滑后的BLEU-4分数
    :param str1: 参考字符串
    :param str2: 生成字符串
    :return: BLEU-4分数
    """
    # 分词
    reference = [str1.split()]  # 参考文本，必须是列表形式，包含一个子列表
    candidate = str2.split()  # 生成文本，直接作为列表

    # 计算BLEU-4分数
    bleu_score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_fn)

    return bleu_score * 100

if __name__ == "__main__":
    from smoothed_bleu import get_score
    # 假设你在同一目录下放有 refs.txt 和 hyps.txt
    refs_file = "../data/codexglue/codexglue_fixed_test.json"
    hyps_file = "../data/codexglue/deepseek_finetune.json"
    f1 = open_json(refs_file)
    f2 = open_json(hyps_file)
    _sum = []
    for i in range(len(f1)):
        # print(smoothed_bleu4(f1[i], f2[i]))
        # score = get_score(str_delete(f1[i]), str_delete(f2[i]))
        score = get_score(str_delete(f2[i]), str_delete(f1[i]))
        if score < 15:
            print("Dataset:", f1[i])
            print("Predicted:", f2[i])
            print("Score:", score)
            print('=' * 20)
        if score < 10:
            continue
        _sum.append(score)
    print(sum(_sum) / len(_sum))
    # bleu_score = compute_smoothed_bleu4(refs_file, hyps_file)
    # print(f"Smoothed BLEU-4 score: {bleu_score:.2f}")
    # hyps_file = "comment_generated.json"
    # bleu_score = compute_smoothed_bleu4(refs_file, hyps_file)
    # print(f"Smoothed BLEU-4 score: {bleu_score:.2f}")
    # hyps_file = "comment_generated_oral.json"
    # bleu_score = compute_smoothed_bleu4(refs_file, hyps_file)
    # print(f"Smoothed BLEU-4 score: {bleu_score:.2f}")

