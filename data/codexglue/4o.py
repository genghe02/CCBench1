import json
import sys

import requests
from openai import OpenAI
from tqdm import tqdm

from run_code import save_json
from tools import open_json


def gpt4o(prompt):
    from openai import OpenAI

    client = OpenAI(api_key="sk-kKcQtH3Dh1DStsLW8e0fDe76F1B24dA2BaA1Ba6874D8E0Bf", base_url="https://vip.apiyi.com/v1")

    try:

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            stream=False,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        result = completion.choices[0].message.content
    except:
        print(completion.choices[0].message)
        return "[ERROR]"

    return result


if __name__ == '__main__':
    prompt = """Task: Next, I will provide you with a code comment, and I hope you can help me delete the content other than the code summary.
For example, input and output parameter descriptions, parameter types, test cases, etc.

Example1:
Input:
Dataset: str->list
    Convert XML to URL List.
    From Biligrab.

✅ Correct Output:
Convert XML to URL List.

Example2:
Input:
Write text at x, y top left corner position.

        By default the x and y coordinates represent the top left hand corner
        of the text. The text can be centered vertically and horizontally by
        using setting the ``center`` option to ``True``.

        :param text: text to write
        :param position: (row, col) tuple
        :param color: RGB tuple
        :param size: font size
        :param antialias: whether or not the text should be antialiased
        :param center: whether or not the text should be centered on the
                       input coordinate
✅ Correct Output:                    
Write text at x, y top left corner position.

Example3:
Input:
Returns a unique ID of a given length.
User `version=2` for cross-systems uniqueness.

✅ Correct Output:
Returns a unique ID of a given length.

This is the end of the examples.
Here is the task you should finish:
"""
    refs_file = "./test.jsonl"
    with open(refs_file, 'r', encoding='utf-8') as f:
        references = [line.strip() for line in f if line.strip()]

    new_references = []

    for item in references:
        js = eval(item)
        # nl = ' '.join(js['docstring_tokens']).replace('\n', '')
        # nl = ' '.join(nl.strip().split())
        nl = js['docstring']
        new_references.append(nl)

    references = new_references
    save_json("oral_codexglue.json", references)
    # codexglue_fixed_test = open_json("oral_codexglue.json")
    codexglue_fixed_test = []

    for i, item in enumerate(tqdm(references)):
        response = gpt4o(prompt + '\n' + item)
        if response == '[ERROR]':
            print(i)
        codexglue_fixed_test.append(response)
        save_json("codexglue_fixed_test.json", codexglue_fixed_test)