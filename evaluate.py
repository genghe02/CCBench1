import argparse
import os
import sys

from tqdm import tqdm

from evaluate import RougeEvaluate, BertScoreEvaluate, smoothed_bleu
from tools import open_json, save_json


def str_delete(target_str: str):
    while '#' in target_str or '\n' in target_str:
        target_str = target_str.replace('\n', '')
        target_str = target_str.replace('#', '')
    return target_str.strip()

def pandasEval_evaluate(indicator_name, model_name, indicator):
    # path of dataset
    dataset_path = './prompt/aligned/pandasEval'
    # path of results
    result_path = f'./results/{model_name}/pandasEval'
    dataset_filenames = os.listdir(dataset_path)
    result_filenames = os.listdir(result_path)
    result = []
    _sum = []
    # pandasEval
    for i in tqdm(range(len(dataset_filenames))):
        dataset_filename = dataset_filenames[i]
        if not dataset_filename.endswith('.txt'):
            continue
        dataset_file = open(dataset_path + '/' + dataset_filename, 'r').read()
        result_filename = dataset_filename.split('.')[0] + '.txt'
        result_file = open(result_path + '/' + result_filename, 'r')
        comment = ""
        for line in result_file:
            if line.strip().startswith('#'):
                comment += str_delete(line)
        if indicator_name == 'bert':
            scores = indicator.get_score(str_delete(comment), str_delete(dataset_file))
            score = {
                "P": scores[0],
                "R": scores[1],
                "F1": scores[2]
            }
        elif indicator_name == 'rouge':
            scores = indicator.get_score(str_delete(comment), str_delete(dataset_file))
            score = scores
        else:
            scores = smoothed_bleu.get_score(str_delete(comment), str_delete(dataset_file))
            score = scores
            _sum.append(score)
            pass
        result.append({
            "id": int(dataset_filename.split('.')[0].split('_')[1]),
            "filename": dataset_filename,
            "dataset_comment": str_delete(dataset_file),
            "generated_comment": str_delete(comment),
            "indicator_name": indicator_name,
            "score": score
        })
        # Test Save
    print(sum(_sum) / len(_sum))
    result = sorted(result, key=lambda x: x["id"])
    save_json("./results/evaluation/" + model_name + "/" + 'pandasEval' + '/' + indicator_name + ".json", result)
    score = None
    return score

def numpyEval_evaluate(indicator_name, model_name, indicator):
    # path of dataset
    dataset_path = './prompt/aligned/numpyEval'
    # path of results
    result_path = f'./results/{model_name}/numpyEval'
    dataset_filenames = os.listdir(dataset_path)
    result_filenames = os.listdir(result_path)
    result = []
    comments_handled_json_list = []
    _sum = []
    # numpyEval
    for i in tqdm(range(len(dataset_filenames))):
        dataset_filename = dataset_filenames[i]
        if not dataset_filename.endswith('.txt'):
            continue
        dataset_file = open(dataset_path + '/' + dataset_filename, 'r').read()
        result_filename = dataset_filename.split('.')[0] + '.txt'
        result_file = open(result_path + '/' + result_filename, 'r')
        comment = ""
        for line in result_file:
            if line.strip().startswith('#'):
                comment += str_delete(line)

        comments_handled_json_list.append(str_delete(comment))
        if indicator_name == 'bert':
            scores = indicator.get_score(str_delete(comment), str_delete(dataset_file))
            score = {
                "P": scores[0],
                "R": scores[1],
                "F1": scores[2]
            }
        elif indicator_name == 'rouge':
            scores = indicator.get_score(str_delete(comment), str_delete(dataset_file))
            score = scores
        else:
            scores = smoothed_bleu.get_score(str_delete(comment), str_delete(dataset_file))
            score = scores
            _sum.append(score)
            pass
        result.append({
            "id": int(dataset_filename.split('.')[0].split('_')[1]),
            "filename": dataset_filename,
            "dataset_comment": str_delete(dataset_file),
            "generated_comment": str_delete(comment),
            "indicator_name": indicator_name,
            "score": score
        })
        # Test Save
    print(sum(_sum) / len(_sum))
    result = sorted(result, key=lambda x: x["id"])
    save_json("./results/evaluation/" + model_name + "/" + 'numpyEval' + '/' + indicator_name + ".json", result)
    save_json("./results/" + f"{model_name}_numpyEval.json", comments_handled_json_list)
    score = None
    return score

def humaneval_result_handler():
    # TODO
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--d", type=str, dest='dataset', required=True)
    parser.add_argument("--m", type=str, dest='model', required=True)
    parser.add_argument("--i", type=str, dest="indicator", required=True)
    args = parser.parse_args()
    dataset_name, indicator_name, model_name = args.dataset, args.indicator, args.model
    assert indicator_name in ('bert', 'rouge', 'bleu'), ValueError
    assert dataset_name in ('pandasEval', 'numpyEval', 'humanEval', 'classEval'), ValueError
    assert model_name in ('deepseek_v2', 'gpt_3_5', 'gpt4omini', 'lamma', 'mistral', 'gpt_4'), ValueError
    if indicator_name == 'rouge':
        indicator = RougeEvaluate()
    elif indicator_name == 'bert':
        indicator = BertScoreEvaluate()
    elif indicator_name == 'bleu':
        indicator = None
    else:
        indicator = None
    if dataset_name == 'pandasEval':
        pandasEval_evaluate(indicator_name, model_name, indicator)
    elif dataset_name == 'numpyEval':
        numpyEval_evaluate(indicator_name, model_name, indicator)
    elif dataset_name == 'humanEval':
        # TODO
        pass
    elif dataset_name == 'classEval':
        # TODO
        pass
