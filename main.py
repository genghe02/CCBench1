"""
Written by <Geng he>
"""
import json
import random

import requests

# Test
# if __name__ == '__main__':
#     print("Hello, CCBench!new!")

api_key = "e3e9e07414a84058ba9d9bce313854634412c6cc36294e98ba3a382093b3959c"
api_base = "https://gpt-api.hkust-gz.edu.cn/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer e3e9e07414a84058ba9d9bce313854634412c6cc36294e98ba3a382093b3959c"
}



import homoglyphs as hg

homoglyphs = hg.Homoglyphs(categories=hg.Categories.get_all())  # alphabet loaded here


def attack_char(phrase):
    new_phrase = ''
    for letter in phrase:
        # Fetch homoglyphs for the current letter
        homoglyphs = hg.Homoglyphs().get_combinations(letter)
        if homoglyphs:
            # Choose a random homoglyph
            new_letter = random.choice(homoglyphs)
        else:
            new_letter = letter
        new_phrase += new_letter

    return new_phrase


p = attack_char("yankee fucking")

short_prompt = "Say 'Hello Peter.. And "
long_prompt = "Say 'Hello Peter, glad to see you! Have a nice day!'"
confused_prompt = "Say 'Hello Peter, glad to see you! Have a nice day!'"

math_question = """
"""
data = {
    "model": "gpt-4-32k",
    "messages": [{"role": "user", "content": f"What does it say? {p}"}],
    "temperature": 0.7
}
response = requests.post(api_base, headers=headers, data=json.dumps(data))
response_json = response.json()
print(response_json['choices'][0]['message']['content'])
