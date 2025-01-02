import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 假设有两个文本数据集
texts1 = [
    "This is a sample sentence.",
    "Another example of text.",
    "Text analysis is interesting.",
]

texts2 = [
    "This is a different sentence.",
    "Yet another text example.",
    "Analyzing text data can be fun.",
]

# 合并两个数据集
all_texts = texts1 + texts2

# 文本向量化
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(all_texts)

# 降维
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X.toarray())

# 创建数据框以便可视化
df = pd.DataFrame(X_reduced, columns=['PC1', 'PC2'])
df['Dataset'] = ['Dataset1'] * len(texts1) + ['Dataset2'] * len(texts2)

# 绘图
plt.figure(figsize=(10, 6))
plt.scatter(df[df['Dataset'] == 'Dataset1']['PC1'], df[df['Dataset'] == 'Dataset1']['PC2'], color='red', label='Dataset 1')
plt.scatter(df[df['Dataset'] == 'Dataset2']['PC1'], df[df['Dataset'] == 'Dataset2']['PC2'], color='blue', label='Dataset 2')
plt.title('Text Dataset Distribution')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid()
plt.show()
