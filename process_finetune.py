import datasets
import csv
import random

from tools import open_json

data = {}
data['messages'] = []

oral_data = open_json("./data/lora_data_without_humaneval.json")

for item in oral_data:
    data['messages'].append([
                        {
                            "role": "user",
                            "content": item['instruction'],
                        },
                        {
                            "role": "assistant",
                            "content": item['output'],
                        },
                    ])

random.shuffle(data['messages'])

datasets.DatasetDict(
    {
        "train": datasets.Dataset.from_dict(data),
        "test": datasets.Dataset.from_dict(data),
    }
).save_to_disk("total_shuffle_data")
