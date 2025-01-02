"""
注释评价指标
"""
import os

from tqdm import tqdm

from evaluate import RougeEvaluate, BertScoreEvaluate, smoothed_bleu
from tools import save_json


def str_delete(target_str: str):
    while '#' in target_str or '\n' in target_str:
        target_str = target_str.replace('\n', '')
        target_str = target_str.replace('#', '')
    return target_str.strip()

if __name__ == '__main__':
    # Set model name
    model_name = 'deepseek_v2'
    assert model_name in ['deepseek_v2', 'gpt3_5', 'gpt4omini', 'lamma', 'mistral', 'gpt_4', "deepseek_7b"]

    # Set dataset name
    dataset_name = 'pandasEval'
    assert dataset_name in ['pandasEval', 'numpyEval', 'humanEval', 'classEval'], ValueError

    if dataset_name == 'pandasEval':
        dataset_path = './prompt/aligned/pandasEval'
        generated_path = f'./new_results/{model_name}_{dataset_name}.txt'
        with open(generated_path, 'r', encoding='utf-8') as f:
            generated_data = f.read()
        generated_data = generated_data.split('[Block]')
        oral_data = os.listdir(dataset_path)
        bert_score_list = []
        rouge_score_list = []
        bleu_score_list = []
        for i, item in enumerate(tqdm(oral_data)):
            # Bert Score
            bert_indicator = BertScoreEvaluate()
            with open(os.path.join(dataset_path, item), 'r', encoding='utf-8') as f_1:
                item_r = f_1.read()
            scores = bert_indicator.get_score(str_delete(item_r), str_delete(eval(generated_data[i])[-1]))
            bert_score = {
                "P": scores[0],
                "R": scores[1],
                "F1": scores[2]
            }
            bert_score_list.append(bert_score)
            # rouge
            # rouge_indicator = RougeEvaluate()
            # rouge_score = rouge_indicator.get_score(str_delete(item_r), str_delete(eval(generated_data[i])[-1]))
            # rouge_score_list.append(rouge_score)
            # smoothed bleu4
            bleu_score = smoothed_bleu.get_score(str_delete(item_r), str_delete(eval(generated_data[i])[-1]))
            bleu_score_list.append(bleu_score)
            save_json(f"./new_results/_evaluation/{model_name}_{dataset_name}_bert.json", bert_score_list)
            # save_json(f"./new_results/_evaluation/{model_name}_{dataset_name}_rouge.json", rouge_score_list)
            save_json(f"./new_results/_evaluation/{model_name}_{dataset_name}_bleu.json", bleu_score_list)
