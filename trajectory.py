import json
import random
import sys

import requests
from langchain.chains import LLMChain
from langchain.llms import BaseLLM
from langchain_chroma.vectorstores import cosine_similarity
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from openai import OpenAI
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from tools import open_json, save_json


def get_client():
    # 设置向量化和检索
    api_key = "sk-fyw5AXxFDRzRHzJ344240f868d964f04Ba90BfBe75A08a73"

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.fast-tunnel.one/v1"
    )
    return client

def get_embedding(doc, client):
    # print(client)
    response = client.embeddings.create(model='text-embedding-3-large', input=doc)
    return response.data[0].embedding

def search_and_generate_answer(query, client, documents, embedding_matrix, top_k=2):
    # 获取查询的嵌入
    query_embedding = np.array(get_embedding(query, client)).reshape(1, -1)

    # 计算余弦相似度
    similarities = cosine_similarity(query_embedding, embedding_matrix)[0]

    # 获取最相关的文档索引
    top_indices = similarities.argsort()[-top_k:][::-1]
    retrieved_docs = [documents[i] for i in top_indices]

    # 将检索到的内容组合成上下文
    context = "\n".join(retrieved_docs)

    prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

Note: You only need to provide Python code, don't generate unnecessary content.

Next, I will provide you with some examples, including the process and results of analyzing and thinking about problems. I hope you can learn the problem-solving methods in the examples and solve the problems I have given you in this way.

### Examples:

{}

### Instruction:
You only need to learn the examples and thought process provided above, and think according to this process, but you don't need to output this process, just output Python code.   
Create a Python script for this problem:
{}


### Response:
    """
    example_str = ""
    questions = [item['question'] for item in doc]
    trajectories = [item['step'] for item in doc]
    results = [item['result'] for item in doc]
    for _index, __index in enumerate(top_indices):
        example_str += f'Example{str(_index)}\n'
        example_str += f'#### Question: {questions[__index]}\n'
        example_str += f'#### Trajectory: {trajectories[__index]}\n'
        example_str += f'#### Result: {results[__index]}\n'
        example_str += f'======== This is the end of Example {_index} =======\n'
    # print(example_str)

    prompt = prompt.format(example_str, query)
    # sys.exit()

    # 使用 OpenAI GPT 生成回答
    if target_model_name == 'gpt_4omini':
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            stream=False,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        answer = completion.choices[0].message.content
    else:
        messages = [
            {'role': 'user', 'content': prompt}
        ]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(
            model.device)
        # tokenizer.eos_token_id is the id of <｜end▁of▁sentence｜>  token
        outputs = model.generate(inputs, max_new_tokens=4096, do_sample=True, top_k=50, top_p=0.95,
                                 num_return_sequences=1,
                                 eos_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    return answer

def get_distance(query, client, sample_set):
    if len(sample_set) == 0:
        return 0

    embeddings = [get_embedding(doc, client) for doc in sample_set]

    embedding_matrix = np.array(embeddings)

    query_embedding = np.array(get_embedding(query, client)).reshape(1, -1)

    # 计算余弦相似度
    similarities = cosine_similarity(query_embedding, embedding_matrix)[0]

    return similarities[0]


def step_by_step_select(samples):
    """
    按步骤将 samples 中的样本加入 target_samples。
    每次都选取与当前 target_samples 中所有样本距离最大的那个样本。

    返回按贪心顺序生成的 target_samples 列表。
    """

    client = get_client()
    # 为了不破坏传入的 samples 原始数据，我们可以复制一份
    candidates = samples.copy()
    target_samples = []
    count = 1
    while candidates:
        print(f"{count} / {len(candidates)}")
        best_candidate = None
        best_distance = -1

        # 遍历所有可选样本，找出距离 target_samples 最远的
        for c in candidates:
            dist = get_distance(c, client, target_samples)
            if dist > best_distance:
                best_distance = dist
                best_candidate = c

        # 将该样本加入 target，并从候选中删除
        target_samples.append(best_candidate)
        candidates.remove(best_candidate)
        count += 1
        if count > 6:
            break

    return target_samples

def get_doc(target_model_name, test):
    # if test:
    #     return open_json("gpt4omini_trajectory.json")
    trajectory_list = []
    # 首先获取V3没做出来的样本名
    v3_failures = []
    v3_json = open_json("deepseek_v2_retry_log.json")
    for item in v3_json:
        if item['error'][-1] != 'Success':
            v3_failures.append(item['task_id'])
    v3_failures = list(set(v3_failures))
    # 然后获取目标模型没做出来的样本名
    target_failures = []
    if target_model_name == 'gpt_4omini':
        gpt4omini_json = open_json("gpt4o-mini_retry_log.json")
        for item in gpt4omini_json:
            if item['error'][-1] != 'Success':
                target_failures.append(item['task_id'])
    else:
        gpt4omini_json = open_json("deepseek_7b_retry_log.json")
        for item in gpt4omini_json:
            if item['error'][-1] != 'Success':
                target_failures.append(item['task_id'])
    target_failures = list(set(target_failures))
    # 筛选目标模型没做出来但是v3做出来的样本
    need_to_trajectory_failures = []
    for item in target_failures:
        if item not in v3_failures:
            need_to_trajectory_failures.append(item)
    print(need_to_trajectory_failures)
    # 23个样本
    print(len(need_to_trajectory_failures))
    # 整个嵌入算法，挨个嵌入，尽可能让样本之间的平均距离大
    doc = step_by_step_select(need_to_trajectory_failures)
    print(doc)
    save_json(f"{target_model_name}_trajectory.json", doc)
    return None



if __name__ == "__main__":
    """
    Preparation Part
    """
    # target_model_name = "gpt_4omini"
    target_model_name = 'deepseek_7b'
    # doc = get_doc(target_model_name, test=True)
    #
    # full_message_doc = []
    # #
    humaneval = open("human-eval-master/data/human-eval-v2-20210705.jsonl", 'r', encoding='utf-8')
    humaneval = humaneval.readlines()

    humaneval_json = []

    for item in humaneval:
        humaneval_json.append(eval(item))
    print(humaneval_json)
    #
    # for i, item in enumerate(doc):
    #     task_id = item['task_id']
    #     for j, item1 in enumerate(humaneval_json):
    #         if item1['task_id'] == task_id:
    #             doc[i]['question'] = item1['prompt']
    #
    # save_json(f"{target_model_name}_trajectory.json", doc)

    """
    RAG Part
    """
    doc = open_json(f"{target_model_name}_trajectory.json")
    # 代码或者任务描述
    # 代码
    documents = [item['question'] for item in doc]

    client = get_client()
    # 生成每个文档的嵌入
    embeddings = [get_embedding(doc, client) for doc in documents]

    if target_model_name == 'deepseek_7b':
        tokenizer = AutoTokenizer.from_pretrained("deepseek/", local_files_only=True, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("deepseek/", local_files_only=True, trust_remote_code=True,
                                                     torch_dtype=torch.bfloat16).cuda()

    # 将嵌入转换为 NumPy 数组
    embedding_matrix = np.array(embeddings)
    responses = []
    for i, item in enumerate(tqdm(humaneval_json)):
        query = item['prompt']
        answer = search_and_generate_answer(query, client, documents, embedding_matrix)
        responses.append({
            "task_id": item['task_id'],
            "solution_code": answer,
            "test": item['test'],
            "canonical_solution": item['canonical_solution'],
            "entry_point": item["entry_point"]
        })
        save_json(f"{target_model_name}_humaneval_code_trajectory.json", responses)
        # sys.exit()





