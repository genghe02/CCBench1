from evaluate.dataset_aligned_analysis import open_json

f = open_json("finetune_prompts.json")
print(len(f))

print(f[0])