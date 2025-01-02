"""
整理注释
"""
import os

from openai import OpenAI
from tqdm import tqdm


def get_data(modelname, datasetname):
    dir_path = f"./new_results/{modelname}/{datasetname}/"
    filenames = os.listdir(dir_path)
    prompt_list = []
    for filename in filenames:
        f = open(os.path.join(dir_path, filename), 'r', encoding='utf-8')
        prompt_l = f.read()
        prompt_list.append(prompt_p + prompt_l)
    return prompt_list

def gpt_handle(api_key, api_base, modelname, prompt):
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
    return completion.choices[0].message

def gpt4o_handler(prompt):
    from models import gpt4mini
    response_text = gpt_handle(
        gpt4mini.GPT4Mini.api_key, gpt4mini.GPT4Mini.api_base, gpt4mini.GPT4Mini.model_name, prompt)
    return response_text.content


if __name__ == '__main__':
    # Set model name
    model_name = 'gpt4omini'
    assert model_name in ['deepseek_v2', 'gpt3_5', 'gpt4omini', 'mistral', 'gpt_4', "deepseek_7b"]

    # Set dataset name
    dataset_name = 'numpyEval'
    assert dataset_name in ['pandasEval', 'numpyEval', 'humanEval', 'classEval'], ValueError

    for model_name in ['gpt_4']:
        for dataset_name in ['numpyEval']:

            if dataset_name == 'pandasEval' or dataset_name == 'numpyEval':
                # Set prompt
                with open('prompt/new/get_prompt_pandas_numpy.txt', 'r', encoding='utf-8') as f:
                    prompt_p = f.read()
                # Set data
                prompts = get_data(model_name, dataset_name)

            ans = ""

            for _index, prompt in enumerate(tqdm(prompts)):
                ans += gpt4o_handler(prompt)
                ans += '[Block]'
                with open(f"./new_results/{model_name}_{dataset_name}.txt", 'w', encoding='utf-8') as f:
                    f.write(ans)
