"""
代码生成实验
Geng He
"""
import argparse
import json
import sys

import requests
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftModel
from tqdm import tqdm

# 保存json格式文件
def save_json(filename, objects):
    """
        :param filename: 你要保存的文件名
        :param objects: 你要保存的内容
        :return: None

        Warning：会覆盖原有内容，谨慎！
    """
    f = open(filename, 'w')
    json.dump(objects, f)
    f.close()

def load_dataset(dataset_name):
    """
    加载数据集: 统一格式
    :return: dict_data
    样例：
    {
        "task_id": ...,
        "prompt": ....,
        "test_code": ...,
    }
    """

    dataset = []

    # prompt
    INSTRUCTION = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

        Note: You only need to provide Python code, don't generate unnecessary content.


        ### Instruction:
        Create a Python script for this problem:
        {}

        ### Response:"""

    # Humaneval
    if dataset_name == 'humaneval':

        file_path = "data/humaneval_comments_version/human-eval-v2-20210705.jsonl"
        f = open(file_path, 'r', encoding='utf-8')
        # f_lines = eval(f.read())
        f_lines = f.readlines()
        for line in f_lines:
            line = json.loads(line)
            dataset.append({
                "task_id": line['task_id'],
                "prompt": INSTRUCTION.format(line['prompt']),
                "test_code": line['test'],
            })
    # TODO: NumpyEval
    # TODO PandasEval
    # TODO: classEval



    return dataset

def load_model(model_name):
    if model_name == 'deepseek':
        tokenizer = AutoTokenizer.from_pretrained("deepseek/", local_files_only=True, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("deepseek/", local_files_only=True, trust_remote_code=True,
                                                     torch_dtype=torch.bfloat16).cuda()
    elif model_name == 'deepseek_finetune':
        model_path = 'deepseek/'
        lora_path = './output/deepseek_coder_v2/checkpoint-114'

        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16,
                                                     trust_remote_code=True).eval()

        # 加载lora权重
        model = PeftModel.from_pretrained(model, model_id=lora_path)
    elif model_name == 'mistral':
        # TODO
        model = None
        tokenizer = None
    else:
        model = None
        tokenizer = None
    return model, tokenizer

def large_model_api(model_name, prompt, model, tokenizer):
    if model_name == 'deepseek' or model_name == 'deepseek_finetune':
        messages = [
            {'role': 'user', 'content': prompt}
        ]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        # tokenizer.eos_token_id is the id of <｜end▁of▁sentence｜>  token
        outputs = model.generate(inputs, max_new_tokens=4096, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1,
                                 eos_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    elif model_name == 'gpt4o-mini':
        api_key = "e3e9e07414a84058ba9d9bce313854634412c6cc36294e98ba3a382093b3959c"
        api_base = "https://gpt-api.hkust-gz.edu.cn/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer e3e9e07414a84058ba9d9bce313854634412c6cc36294e98ba3a382093b3959c"
            # Please change your KEY. If your key is XXX, the Authorization is "Authorization": "Bearer XXX"
        }
        data = {
            "model": "gpt-3.5-turbo",
            # "gpt-3.5-turbo" version in gpt-3.5-turbo-1106,
            # "gpt-4" version in gpt-4-1106-version (gpt-4-vision-preview is NOT available in azure openai),
            # "gpt-3.5-turbo-16k", "gpt-4-32k"
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        response = requests.post(api_base, headers=headers, data=json.dumps(data))
        response_json = response.json()
        print(response_json)
        return response_json['choices'][0]['message']['content']
    elif model_name == 'gpt3_5':
        modelname = 'gpt-3.5-turbo'
        api_key = "sk-fyw5AXxFDRzRHzJ344240f868d964f04Ba90BfBe75A08a73"
        api_base = "https://api.fast-tunnel.one/v1"
        client = OpenAI(api_key=api_key, base_url=api_base)

        # save the whole response body
        completion = client.chat.completions.create(
            model=f"{modelname}",
            messages=[
                {"role": "user", "content": f"{prompt}"}
            ]
        )
        return completion.choices[0].message.content
        # return None
    elif model_name == 'gpt_4':
        modelname = 'gpt-4-turbo'
        api_key = "sk-fyw5AXxFDRzRHzJ344240f868d964f04Ba90BfBe75A08a73"
        api_base = "https://api.fast-tunnel.one/v1"
        client = OpenAI(api_key=api_key, base_url=api_base)

        # save the whole response body
        completion = client.chat.completions.create(
            model=f"{modelname}",
            messages=[
                {"role": "user", "content": f"{prompt}"}
            ]
        )
        return completion.choices[0].message.content
    elif model_name == 'mistral':
        from transformers import pipeline
        from huggingface_hub import login
        login(token='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')

        messages = [
            {"role": "user", "content": prompt},
        ]
        pipe = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2", max_new_tokens=5096)
        return pipe(messages)[0]['generated_text'][1]['content']
    elif model_name == 'deepseek_v2':
        client = OpenAI(api_key='sk-5780d51e99ed49f7a0c0696089b7dc62', base_url="https://api.deepseek.com")

        response = client.chat.completions.create(
            model="deepseek- ",
            messages=[
                {"role": "user", "content": prompt},
            ],
            stream=False
        )

        return response.choices[0].message.content
    # TODO:需要包含日志保存
    # 输入模型种类、prompt，返回模型的输出

    pass

def results_save():
    # TODO: 保存结果
    pass

if __name__ == '__main__':
    # 设置参数
    # parser = argparse.ArgumentParser(description='manual to this script')
    # parser.add_argument("--d", type=str, dest='dataset', required=True, default='humaneval')
    # parser.add_argument("--m", type=str, dest='model', required=True, default='deepseek')
    # args = parser.parse_args()
    # dataset_name, model_name = args.dataset, args.model

    # Test Settings
    dataset_name, model_name = "humaneval", "deepseek_v2"

    dataset_name = dataset_name.lower()
    assert dataset_name in ("humaneval", "classeval", "pandaseval", "numpyeval")
    # TODO: 加载数据集
    dataset = load_dataset(dataset_name)

    # TODO: 加载大模型(如果需要
    model, tokenizer = load_model(model_name)

    # TODO: 模型接口
    responses = []
    for _index, data in enumerate(tqdm(dataset)):
        for i in range(10):
            response = large_model_api(model_name, data['prompt'], model, tokenizer)
            # print(response)
            responses.append({
                "task_id": data['task_id'],
                "prompt": data['prompt'],
                "solution_code": response,
                "test_code": data['test_code']
            })
        # TODO:结果保存
        save_json(f"{model_name}_{dataset_name}_code.json", responses)

"""
Humaneval上的表现
gpt4omini: {'pass@1': 0.8689024390243902, 'pass@10': 0.9451219512195121}
deepseek 7b: {'pass@1': 0.823780487804878, 'pass@10': 0.8658536585365854}
deepseek 7b finetuned:{'pass@1': 0.8176829268292682, 'pass@10': 0.8536585365853658}(待定)
gpt 3.5:{'pass@1': 0.6579268292682927, 'pass@10': 0.9085365853658537}
gpt4: {'pass@1': 0.8975609756097562, 'pass@10': 0.9634146341463414}
mistral: {'pass@1': 0.35548780487804876,, 'pass@10': 0.6097560975609756 }
deepseek: {'pass@1': 0.89, 'pass@10': - }
"""
