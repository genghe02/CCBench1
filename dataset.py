import datetime
import json
import os
import sys

import requests
import torch
from openai import OpenAI
from tqdm import tqdm

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login


def mistral7B(prompt, test_case_name, model, tokenizer):
    messages = [
        {'role': 'user', 'content': prompt}
    ]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(
        'cuda')
    # tokenizer.eos_token_id is the id of <｜end▁of▁sentence｜>  token
    outputs = model.generate(inputs, max_new_tokens=4096, do_sample=True, top_k=50, top_p=0.95,
                             num_return_sequences=1,
                             eos_token_id=tokenizer.eos_token_id)
    return tokenizer.batch_decode(outputs)[0]


def deepseek_v2_handle(api_key, base_url, model_name, prompt, test_case_name):
    client = OpenAI(api_key=api_key, base_url=base_url)

    response = client.chat.completions.create(
        model="deepseek-coder",
        messages=[
            {"role": "user", "content": prompt},
        ],
        stream=False
    )

    save_log({
        "response body": response.to_json(),
        "test_case_name": test_case_name
    }, dataset_name)

    return response.choices[0].message.content


def gpt_3_5_handle(url, headers, data, model_name, test_case_name):
    response = requests.post(url, headers=headers, data=json.dumps(data))
    response_json = response.json()
    save_log({
        "response body": response_json,
        "test_case_name": test_case_name
    }, dataset_name)
    return response_json['choices'][0]['message']['content']


def dataset(dataset_name, prompt_dic):
    assert dataset_name in ['pandasEval', 'numpyEval', 'humanEval', 'classEval'], ValueError
    prompt_list = []
    dataset_filename = dataset_name.lower()
    dir_path = f"data/{dataset_filename}_comments_version/{dataset_filename}_without_comments/"
    pandas_file_list = os.listdir(dir_path)
    for filename in pandas_file_list:
        if not filename.endswith(".txt"):
            continue
        with open(dir_path + filename, 'r', encoding='utf-8') as f:
            prompt_list.append({
                "test_case_name": filename,
                "prompt": prompt_dic[dataset_name] + '\n' + f.read()
            })
    return prompt_list


def save_log(json_text, dir_name):
    """
    Save the json text to "./log/dirname"
    example:
    >> {"test_name_case": "test_name_case", "response body": "response body"}
    :param json_text: text of log
    :param dir_name: name of directory
    :return: None
    """
    try:
        os.mkdir(f"log/{dir_name}")
    except FileExistsError:
        pass
    with open(f"log/{dir_name}/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
        json.dump(json_text, f)


def gpt_handle(api_key, api_base, modelname, prompt, test_case_name, dataset_name):
    """
    :param api_key: OPENAI API key
    :param api_base: OPENAI API base url
    :param modelname: LLM model name
    :param prompt: prompt
    :param test_case_name: name of test_case
    :return: response name of LLM
    """
    client = OpenAI(api_key=api_key, base_url=api_base)

    # save the whole response body
    completion = client.chat.completions.create(
        model=f"{modelname}",
        messages=[
            {"role": "user", "content": f"{prompt}"}
        ]
    )

    save_log({
        "response body": str(completion),
        "test_case_name": test_case_name
    }, dataset_name)
    return completion.choices[0].message


def save_response(response_text, dataset, filename, modelname):
    assert dataset in ("pandasEval", "numpyEval", "humanEval", "classEval"), ValueError
    ans_save_path = f"new_results/{modelname}/{dataset}"
    with open(ans_save_path + "/" + filename, "w", encoding='utf-8') as f:
        # For GPT API
        f.write(response_text.content)
        # For Huggingface local model
        # f.write(response_text)

def deepseek_7b_run(prompt, test_case_name, model, tokenizer):
    messages = [
        {'role': 'user', 'content': prompt}
    ]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    # tokenizer.eos_token_id is the id of <｜end▁of▁sentence｜>  token
    outputs = model.generate(inputs, max_new_tokens=4096, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1,
                             eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

if __name__ == '__main__':
    # TODO: Set argparse
    # dataset_name = 'pandasEval'
    # for dataset_name in ['classEval']:
    for dataset_name in ('humanEval', 'classEval', "pandasEval", "numpyEval"):

        assert dataset_name in ['pandasEval', 'numpyEval', 'humanEval', 'classEval'], ValueError

        # modelname = 'gpt4omini'
        modelname = 'gpt3_5'
        # modelname = 'gpt_4'
        assert modelname in ['deepseek_v2', 'gpt3_5', 'gpt4omini', 'lamma', 'mistral', 'gpt_4', "deepseek_7b"], ValueError

        if modelname == 'mistral':
            model = AutoModelForCausalLM.from_pretrained("mistral/", local_files_only=True, device_map="auto")
            tokenizer = AutoTokenizer.from_pretrained("mistral/", local_files_only=True,
                                                      torch_dtype=torch.bfloat16, device_map="auto")

        if modelname == 'deepseek_7b':
            tokenizer = AutoTokenizer.from_pretrained("deepseek/", local_files_only=True, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained("deepseek/", local_files_only=True, trust_remote_code=True,
                                                         torch_dtype=torch.bfloat16).cuda()

        # set the default prompt template
        prompt_dic = {
            "pandasEval": "",
            "numpyEval": "",
            "humanEval": "",
            "classEval": ""
        }
        with open('prompt/new/require1.txt', 'r', encoding='utf-8') as f:
            prompt_dic["pandasEval"] = f.read()
        with open('prompt/new/require1.txt', 'r', encoding='utf-8') as f:
            prompt_dic["numpyEval"] = f.read()
        with open('prompt/new/humaneval_comment1.txt', 'r', encoding='utf-8') as f:
            prompt_dic["humanEval"] = f.read()
        with open('prompt/new/classeval_prompt.txt', 'r', encoding='utf-8') as f:
            prompt_dic["classEval"] = f.read()

        prompt_list = dataset(dataset_name, prompt_dic)

        for i in tqdm(range(len(prompt_list))):
            # if i < 63 and dataset_name == 'humanEval':
            #     continue

            prompt = prompt_list[i]
            """
            GPT 4o-mini
            """
            # from models import gpt4mini
            #
            # response_text = gpt_handle(
            #     gpt4mini.GPT4Mini.api_key, gpt4mini.GPT4Mini.api_base, gpt4mini.GPT4Mini.model_name, prompt,
            #     prompt["test_case_name"], dataset_name)
            # save_response(response_text, dataset_name, prompt["test_case_name"], modelname)
            """
            Mistral 7B
            """
            # response_text = mistral7B(prompt['prompt'], prompt["test_case_name"], model, tokenizer)
            # save_response(response_text, dataset_name, prompt["test_case_name"], modelname)
            """
            GPT 3.5
            """
            from models import gpt4mini

            response_text = gpt_handle(
                    gpt4mini.GPT4Mini.api_key, gpt4mini.GPT4Mini.api_base, "gpt-3.5-turbo", prompt['prompt'],
                    prompt["test_case_name"], dataset_name)
            save_response(response_text, dataset_name, prompt["test_case_name"], modelname)
            """
            deepseek coder v2
            """
            # from models import deepseek_coder_v2
            #
            # response_text = deepseek_v2_handle(deepseek_coder_v2.Deepseek_v2.api_key,
            #                                    deepseek_coder_v2.Deepseek_v2.base_url,
            #                                    deepseek_coder_v2.Deepseek_v2.model_name,
            #                                    prompt['prompt'], prompt["test_case_name"])
            #
            # save_response(response_text, dataset_name, prompt["test_case_name"], modelname)

            """
            GPT 4
            """
            # from models import gpt_3_5
            #
            # data = {
            #     "model": "gpt-4-32k",
            #     "messages": [{"role": "user", "content": prompt['prompt']}],
            #     "temperature": 0.7
            # }
            # response_text = gpt_3_5_handle(gpt_3_5.GPT_3_5.api_base, gpt_3_5.GPT_3_5.headers, data, dataset_name, prompt["test_case_name"])
            # save_response(response_text, dataset_name, prompt["test_case_name"], modelname)
            """
            deepseek 7b
            """

            # response_text = deepseek_7b_run(prompt['prompt'], prompt["test_case_name"], model, tokenizer)
            # save_response(response_text, dataset_name, prompt["test_case_name"], modelname)

            """
            Lamma 3.1
            """
            # sys.exit()


