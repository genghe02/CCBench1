#!/bin/bash
#SBATCH -J CCBench
#SBATCH --partition=i64m1tga40u
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1
#SBATCH --qos=low
#SBATCH -o log/res_backup.out   # 作业运行log输出文件
#SBATCH -e log/res_backup.err   # 作业错误信息log输出文件
module load cuda/12.1
##################
# ABOVE SERVER SETTINGS
##################




##################
# Some messages
echo $(date +%Y-%m-%d\ %H:%M:%S)
source activate
source deactivate
conda activate CCBench
# cd /hpc2hdd/home/yxu409/genghe/CCBench  # 更改工作目录
echo "Use GPU ${CUDA_VISIBLE_DEVICES}"
##################



##################
# Get comment results from LLM
# Lamma 8b
# https://huggingface.co/meta-llama/Meta-Llama-3.1-8B
#python models/lamma_8B.py

# Mistral
# https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
#python models/mistral_7B.python
##################

# deepseek V2 Coder 8B
# https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct/tree/main
python models/deepseek_coder_v2_8b.py


# Get comment results from LLM
#python dataset.python



# Bar Graph
# Bert-Score
#python figure/bert_bar.py
# Rouge-Score
#python figure/rouge_bar.py
# Rouge-Score
#python figure/percentageLineChart.py --m deepseek_v2 --i bert

# Evaluate
#python evaluate.py --d 'numpyEval' --m 'gpt_4' --i 'bleu'



