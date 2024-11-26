import math
from collections import Counter


def calculate_entropy(text):
    # 统计词频
    word_counts = Counter(text.split())
    total_words = sum(word_counts.values())

    # 计算熵
    entropy = -sum((count / total_words) * math.log2(count / total_words) for count in word_counts.values())
    return entropy


def information_gain(text_a, text_b):
    entropy_a = calculate_entropy(text_a)
    entropy_b = calculate_entropy(text_b)

    # 信息增益可以表示为文本B的熵相对于文本A的减少量
    return entropy_a - entropy_b


# 示例
text_a = "the quick brown fox jumps over the lazy dog"
text_b = "the quick brown fox jumps over the lazy dog"

ig = information_gain(text_a, text_b)
print(f"Information Gain of text B relative to text A: {ig}")
