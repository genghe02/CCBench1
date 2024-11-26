# 实验结果&结论

<!--1.数据对应的问题-->
<!--2.数据-->
<!--3.问题对应的目的-->

## 1数据集分析
### 1.1数据集的可读性：
**1.1.1对齐前的可读性**

**数据对应的问题**：对齐前的数据集是否可读性较差

**问题对应的目的**：证明原始数据集的缺陷：包含可读性较差、注释有误等（可以举具体的case放附录里）

路径：.{数据集名称}_read_abality.json

**注意**：这里的可读性是精确到每个case的精确值。如果要看均值，可以移步2.2章节

json文件解释：
```python
[
    # 指标名
    "flesch_reading_ease": [
        # 每一个case的可读性指标得分
        74.69,
        61.33,
        69.79,
        66.67,
        43.39,
        ...]
]
```
**1.1.2对齐后的可读性**

**数据对应的问题**：对齐后的数据集是否可读性有上升？

**问题对应的目的**：证明我们数据集制作过程中对齐这一步是必要的。

路径：.{数据集名称}_aligned_read_abality.json

### 1.2数据集的长度分析
**1.2.1对齐前的长度**

**数据对应的问题**：对齐前的数据集是否注释过于冗余？导致信息密度降低？

**问题对应的目的**：证明原始数据集的长度普遍较长、信息密度普遍偏低。

路径：.{数据集名称}_original_length.json

文件是一个列表，里面是每个case的注释长度

整体的结果：
- pandasEval：
  - 总长度：13804
  - 平均长度：
- numpyEval：
  - 总长度：13444
  - 平均长度：
- classEval: 
  - 总长度：53461
  - 平均长度
- humanEval: 
  - 总长度：
  - 平均长度：36231

CodeSearchNet:
训练集：
- 总长度：119020886
- 平均长度：288.76088971269695
验证集：
- 总长度：7555789
- 平均长度：326.9913446141862
测试集：
- 总长度：6663684
- 平均长度：300.49080086580085

**1.2.2对齐后的长度**

**数据对应的问题**：对齐后的数据集是否信息密度有所上升？

**问题对应的目的**：证明我们数据集制作过程中对齐这一步是必要的。

路径：.{数据集名称}_aligned_length.json

整体结果：

numpyEval：
- 总长度：11090
- 减少的长度：2354
- 平均每个case减少的字符数：23.306930693069308

pandasEval：
- 总长度：11789
- 减少的长度：2015
- 平均每个case减少的字符数：19.95049504950495

classEval:
- 总长度:53554
- 减少的长度：-93(一定程度上能说明classEval的注释效果明显优质，classEval也比较新，也能说明我们对齐这个步骤的正确性)
- 平均每个case减少的字符数：-0.93

humanEval:
- 总长度：30904
- 减少的长度：5327
- 平均每个case减少的字符数：



**1.2.3对齐前后的长度方差（熵）**

**数据对应的问题**：对齐前原始数据集是否注释长度熵较大？对齐后这一现象是否得到了改善？

**问题对应的目的**：证明我们数据集制作过程中对齐这一步是必要的。

Before/After：对齐前后

熵、方差、标准差

pandasEval:
- Before: 5.335817813681178 (8541.21007744339, 92.41866736457192)
- After: 4.941006096007957 (5417.269679443194, 73.60210377049826)

numpyEval:
- Before: 5.226633285703219 (8381.542593863347, 91.55076511894015)
- After: 4.955769381904778 (5391.168708950104, 73.42457837093859)

classEval:
- Before: 6.518601124309857 (51232.277899999994, 226.34548349812505)
- After: 6.491587981778348 (47761.62839999998, 218.5443396658902)

humanEval:
- Before:5.885502424045776 (29339.426643367045, 171.28755542469233)
- After: 5.733674567323545 (14126.892623438429, 118.85660529999345)

CodeSearchNet
- Train: 6.2295117460186225 (292645.98909268936, 540.9676414469625)
- Val: 6.169694288662192 (332333.53590985085, 576.4837689908111)
- Test: 6.012762036275767 (348103.5250777136, 590.0029873464316)

说明的问题：
- 复杂任务的注释长度方差明显会大（反映在ClassEval >> humanEval >> Pandas/NumpyEval
- 对齐操作一定程度上可以减小注释的风格差异（不同数据集对齐后长度方差都会明显下降）

### 1.3 可读性
codeSearchNet：
- 详细的、每条case的可读性：
  - detailed_test_aligned_read_ability.json
  - detailed_train_aligned_read_ability.json
  - detailed_validation_aligned_read_ability.json
- 平均的可读性：
  - test_aligned_read_ability.json
  - train_aligned_read_ability.json
  - validation_aligned_read_ability.json

其他四个详细的、每条case的可读性：
- {数据集名称}_aligned_read_ability.json

### 1.4 待做
- HumanEval和ClassEval的结果当时没存，跑代码存一下
  - 长度分析 (√)
  - 对齐(√)
  - 可读性(√)
- 对CodeSearchNet数据集也做一套相同的流程分析(√)


## 2注释
### 2.1 模型&数据集介绍
**2.1.1 我们做的数据集包含**

**数据对应的问题**：如何选择数据作为我们数据集的原始数据？选择哪些大模型进行实验？

**问题对应的目的**：证明我们数据集的数据选取是合理的。证明我们的模型选取是合理的。

- 选用的模型：
  - Mistral 7B
  - Gpt3.5
  - Gpt4
  - Deepseek coder v2
- 数据集
  - HumanEval
  - NumpyEval
  - PandasEval
  - ClassEval

**2.1.2** 要跑过baseline的数据集
- CodeSearchNet

### 2.2 可读性分析

**数据对应的问题**：不同LLM生成的注释可读性与对齐后的数据相比，可读性如何？

**问题对应的目的**：验证不同LLM生成注释的效果（从可读性角度评估）

这里指的是不同模型在不同数据集上生成注释的可读性

**注意**：model为Dataset的时候，这个指标指的是原始数据集的注释的可读性（跟模型没关系）

路径：/results/readAbality/{指标名}.json

json文件解释：
 ```python
# Demo
[
    {
        "model": "deepseek_v2", # 生成注释的模型

        "dataset": "pandasEval", # 数据集名称
            
        "gunning_fog": 9.311980198019798 # 指标值
    }
    ...
]
 ```
### 2.3 数据集效果验证

**数据对应的问题**：不同LLM生成的注释是否贴合原始注释？使用三种指标评估。

**问题对应的目的**：验证不同LLM生成注释的效果（从综合评分角度评估）

Meeting明确的一点：我们要在我们的数据集上微调，实验我们的方法，然后迁移到CodeSearchNet上，证明我们的方法是有效的。

文件目录：
./results/evaluation/{模型名称}/{数据集名称}/{评价指标}.json
- 模型名称就是上面那四个
- 评价指标包含：
  - smoothed-bleu4
  - rouge
  - bert-score
- 数据集包含
  - pandaseval
  - numpyeval
  - classeval
  - humaneval

### 2.4 不同LLM生成结果的test_case分支覆盖率分析

**数据对应的问题**：不同LLM生成test_case的能力如何？是否有未完全覆盖或冗余现象？

**问题对应的目的**：验证不同LLM生成注释的效果

原始数据集中（classEval，humaneval）代码的分支数：
- results\testcases\classeval_branch.json
- results\testcases\humaneval_branch.json

不同大模型生成的test_case覆盖的分支数：
- results\testcases\cover\

classEval上各个大模型的test_case的分支覆盖率表现：
- gpt_4 0.6533180778032036
- deepseek_v2 0.7634285714285715
- gpt_3_5 0.2845714285714286
- mistral 0.3314285714285714

humanEval上各个大模型的test_case的分支覆盖率的表现：
- gpt_4 0.8997772828507795
- deepseek_v2 0.9487750556792873
- gpt_3_5 0.7149220489977728
- mistral 0.7527839643652561

### 2.4 具体的一些case/实验结果
**2.4.1 每个模型+数据集的结果**

**数据对应的问题**：具体展示每个LLM生成的注释的case，可以观察分析得到一些结论

**问题对应的目的**：验证不同LLM生成注释的效果

路径：./results/{模型名}/{数据集名称}/

**2.4.2 每个数据集对齐后的case**

**数据对应的问题**：具体展示对齐后的数据集，可以观察分析得到一些结论

**问题对应的目的**：证明我们数据集制作过程中对齐这一步是必要的。

路径：./prompt/aligned/{数据集名称}/

**2.4.3 prompt**

**数据对应的问题**：具体展示我们的prompt

**问题对应的目的**：证明我们的prompt设置是合理的。

路径：
- 对齐的prompt
  - ./prompt/test_prompts/alignment.txt
- 跑结果的prompt
  - ./prompt/test_prompts/comment.txt

## 2.5 一些图片

当时为了方便看效果临时画的

**2.5.1 可读性图**

路径：./figures/imgs/readability

**2.5.2 BertScore评分**

路径：./figures/imgs/bert_score.png

**2.5.3 长度比**

路径：./figures/imgs/CompressionRadio_bar.png

**2.5.4 rouge_s评分**

路径：./figures/imgs/rouge_s_score.png

## 2.6 待做

 - HumanEval与classEval的实验补上
 - 用微调的Deepseek和GPT做一下，得干过Baseline(codeT5+, Incoder)[!]
 - 各个实验集中在CodeSearchNet数据集上再做一下
 - Deepseek coder v2 7b/8b 再跑一遍
 - T5 用我们的方法做一下，看看能不能跑得过

## 3 代码生成



