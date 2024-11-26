import datetime
import json
import os
import sys

from openai import OpenAI
from tqdm import tqdm


def get_comment(dataset):
    assert dataset in ("pandasEval", "numpyEval", "humanEval", "classEval")
    comments_list = []
    if dataset == "pandasEval":
        dir_path = "prompt/to_align/pandasEval"
        ans_save_path = "prompt/aligned/pandasEval"
        save_files = os.listdir(ans_save_path)
        files = []
        for item in save_files:
            with open(ans_save_path + "/" + item, "r") as f:
                if f.read() != '':
                    continue
                files.append(item)
        files = sorted(files)
        for filename in files:
            with open(dir_path + "/" + filename, "r") as f:
                comments_list.append({
                    "filename": filename,
                    "comment": f.read()
                })
    elif dataset == "numpyEval":
        dir_path = "prompt/to_align/numpyEval"
        ans_save_path = "prompt/aligned/numpyEval"
        save_files = os.listdir(ans_save_path)
        files = []
        for item in save_files:
            with open(ans_save_path + "/" + item, "r") as f:
                if f.read() != '':
                    continue
                files.append(item)
        files = sorted(files)
        for filename in files:
            with open(dir_path + "/" + filename, "r") as f:
                comments_list.append({
                    "filename": filename,
                    "comment": f.read()
                })
    return comments_list

def gpt_handle(comment, filename):
    api_key = "sk-fyw5AXxFDRzRHzJ344240f868d964f04Ba90BfBe75A08a73"
    api_base = "https://api.fast-tunnel.one/v1"
    client = OpenAI(api_key=api_key, base_url=api_base)

    completion = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=[
        {"role": "user", "content": f"{comment}"}
      ]
    )

    print(completion)

    save_log({
        "response body": str(completion),
        "filename": filename
    })
    return completion.choices[0].message


def save_log(json_text):
    with open(f"log/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
        json.dump(json_text, f)

def save_response(response_text, dataset, filename):
    assert dataset in ("pandasEval", "numpyEval", "humanEval", "classEval")
    if dataset == "pandasEval":
        ans_save_path = "prompt/aligned/pandasEval"
        files = []
        save_files = os.listdir(ans_save_path)
        for item in save_files:
            with open(ans_save_path + "/" + item, "r") as f:
                if f.read() != '':
                    continue
                files.append(item)
        with open(ans_save_path + "/" + filename, "w") as f:
            f.write(response_text.content)
    elif dataset == "numpyEval":
        ans_save_path = "prompt/aligned/numpyEval"
        files = []
        save_files = os.listdir(ans_save_path)
        for item in save_files:
            with open(ans_save_path + "/" + item, "r") as f:
                if f.read() != '':
                    continue
                files.append(item)
        with open(ans_save_path + "/" + filename, "w") as f:
            f.write(response_text.content)


if __name__ == "__main__":
    dataset = "numpyEval"
    comments_list = get_comment(dataset)
    for i in tqdm(range(len(comments_list))):
        comment = comments_list[i]
        response = gpt_handle(comment['comment'], comment['filename'])
        save_response(response, dataset, comment['filename'])

