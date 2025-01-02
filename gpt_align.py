import datetime
import json
import os
import sys

from openai import OpenAI
from tqdm import tqdm
from models import gpt4mini
from prompt.alignment_comments_class_human import save_json
from tools import open_json


def gpt_handle(comment, filename, modelname, api_key, api_base):
    client = OpenAI(api_key=api_key, base_url=api_base)

    completion = client.chat.completions.create(
        model=f"{modelname}",
        messages=[
            {"role": "user", "content": f"{comment}"}
        ]
    )

    save_log({
        "response body": str(completion),
        "filename": filename
    })
    return completion.choices[0].message


def save_log(json_text):
    with open(f"log/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
        json.dump(json_text, f)


def get_comment(dataset):
    assert dataset in ("humanEval", "classEval"), ValueError
    if dataset == "classEval":
        file_path = "classeval_align_prompt.json"
        file = open_json(file_path)
        results = []
        for i in tqdm(range(len(file))):
            message = file[i]
            prompts = message['comments_prompt']
            response_message = {
                "filename": message['filename'],
                "comments": []
            }
            for prompt in prompts:
                response = gpt_handle(prompt, message['filename'], gpt4mini.GPT4Mini.model_name,
                                      gpt4mini.GPT4Mini.api_key, gpt4mini.GPT4Mini.api_base)
                response_message["comments"].append(response.content)
            results.append(response_message)
            save_json("aligned_classEval.json", results)
    else:
        file_path = "humaneval_align_prompt.json"
        file = open_json(file_path)
        results = []
        for i in tqdm(range(len(file))):
            message = file[i]
            prompts = message['comments_prompt']
            response_message = {
                "filename": message['filename'],
                "comments": []
            }
            response = gpt_handle(prompts, message['filename'], gpt4mini.GPT4Mini.model_name,
                                  gpt4mini.GPT4Mini.api_key, gpt4mini.GPT4Mini.api_base)
            response_message["comments"].append(response.content)
            results.append(response_message)
            save_json("aligned_humanEval.json", results)


if __name__ == "__main__":
    dataset = "humanEval"
    get_comment(dataset)

