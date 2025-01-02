import json
import os


def handle(filename):
    with open(filename, 'r') as f:
        new_file_name = filename
        new_file = open(f"pandaseval_without_comments/{new_file_name.split('.')[0]}.txt", "w")
        comments = ""
        for line in f.readlines():
            tmp = line.lstrip()
            if len(tmp) == len(line):
                if line.lstrip().startswith("#"):
                    comments += line.lstrip()
                    new_file.writelines('<Requirements for the code>\n')
                else:
                    new_file.writelines(line)
            else:
                blank = (len(line) - len(tmp)) * ' '
                if line.lstrip().startswith("#"):
                    comments += line.lstrip()
                    new_file.writelines(f'{blank}<Requirements for the code>\n')
                else:
                    new_file.writelines(line)
        dic = {
            "filename": f"{filename}",
            "comments": comments
        }
        with open(f"pandaseval_without_comments_json/{new_file_name.split('.')[0]}.json", "w") as outfile:
            json.dump(dic, outfile)


filelist = os.listdir(".")

# print(filelist)

for filename in filelist:
    if filename.endswith(".txt"):
        handle(filename)
