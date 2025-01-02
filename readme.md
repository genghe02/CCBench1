# Nicked project

<p align="center">
<img src="https://img.shields.io/badge/benchmark-lightseagreen" style="height: 20px;"> <img src="https://img.shields.io/badge/contributions-welcome-lemonchiffon.svg" style="height: 20px;">
</p>

---------

Target

---------

Dataset(4 datasets mainly accepted):
- Refactored five widely used code generation datasets, enabling them to measure not only code generation standards but also annotation generation quality.（✅）
- contains <HumanEval, ClassEval, NumpyEval and pandasEval>（✅）
- Alignment of <HumanEval, numpyEval and pandasEval>（✅）
---------
LLMs in use:
- GPT-4(✅)
- GPT-4o-mini(✅)
- ChatGPT(✅)
- Llama3.1(In process)
- mistral(✅)
- deepseek-coder-v2(✅)
--------
注释生成效果评判
- GPT-4
  - humaneval
  - pandaseval(✅)
  - numpyeval(✅)
  - classeval
- GPT-4o-mini
  - humaneval
  - pandaseval
  - numpyeval
  - classeval
- ChatGPT
  - humaneval
  - pandaseval(✅)
  - numpyeval(✅)
  - classeval
- Llama3.1
  - humaneval
  - pandaseval
  - numpyeval
  - classeval
- mistral
  - humaneval
  - pandaseval(✅)
  - numpyeval(✅)
  - classeval
- deepseek-coder-v2
  - humaneval
  - pandaseval(✅)
  - numpyeval(✅)
  - classeval
--------
Analysis

PART 1 数据集

- 数据集代码风格差异
  - 长度
    - 对齐前后，查看数据集数据长度差异(✅)
      - 字符级别(✅)
      - 百分比级别 abs(a - b) / a(✅)
      - 多少条数据变短了？占整体的百分比？(✅)
    - 方差(✅)
  - 可读性
    - 对齐前后，数据集可读性差异
      - 对齐前后平均可读性差异
      - 多少条数据可读性⬆️，多少⬇️

PART 2 代码生成效果
- 代码质量（探究影响注释生成的关系）
  - 长度（上下文情景复杂程度）
  - ?
- 代码生成效果
  - pass@k
  - recall@k
- 代码风格
  - flake包辅助

PART 3 注释生成效果
- 测试样例设计
  - 全面性 cover的分支数/总分支数
  - 冗余性 测试样例数量/总分支数
- 注释生成效果
  - rouge(✅)(❌ 效果好像很差)
  - BLEU
  - Bert_Score(✅)
  - SBert
  - 注释可读性
    - 9个指标(✅)
    - 分析哪些llm的注释和原版最相似
  - 压缩率(✅?)
    - 生成注释长度/原注释长度(✅)
    - TBD
  - CodeBLEU——功能语义
  - EM——完全匹配度
  - 用基于熵和散毒的指标计算下信息增益或者说新信息比例
- Mask掉一些东西，分析哪部分prompt最work（探究如何改进prompt更好的生成注释，或者说哪部分prompt对LLM的理解作用最大）

PART 4 其它
- 指标效果：OR 
--------
最终指导意义：
- 提高大模型代码生成的可读性（这个是注释）
- 提高大模型代码生成的质量（这是代码本身）
--------
Conclusion
- 数据集方面
  - pandasEval, numpyEval以及humanEval都有语义没有对齐的情况（语言风格迥异）
  - 流行的数据集中对注释不够重视，注释中很多词汇拼写错误、格式不统一
- 实验方面
  - 似乎GPT4反而“不听话“或过于聪明，无法理解注释生成任务，会生成许多额外的内容。这一点上GPT3.5、deepseek等模型表现反而好一些；
  - Mistral无法理解“添加测试用例“的说法，仅能设计整个类的测试用例
- 指标方面