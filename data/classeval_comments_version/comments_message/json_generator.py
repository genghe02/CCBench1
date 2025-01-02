import os
filePath = '../../ClassEval-master/data/benchmark_solution_code'
filename_list = os.listdir(filePath)
for filename in filename_list:
    new_name = filename.split('.')[0]
    file = open(f"{new_name}.json", "w")
    file.close()