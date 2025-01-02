import os

from tools import open_json, save_json


# humaneval
# humaneval_comments = []
# all_comment = ""
# for i in range(164):
#     ans = {
#         "filename": f"HumanEval_{i}.txt",
#         "comments": [
#
#         ]
#     }
#     humaneval_comments.append(ans)
# save_json("humaneval_comments.json", humaneval_comments)

# classeval
original_classeval_filepath = "data/classeval_comments_version/classeval_without_comments"

classeval_only_comments_filepath = "data/classeval_comments_version/comments_message"

tokens = [
    "<description for whole class>",
    "<description for whole function>",
    "<description for all parameters>",
    "<description for return statement>",
    "<some test cases for the function>"
]

original_classeval_filenames = os.listdir(original_classeval_filepath)
classeval_comments = []
for filename in original_classeval_filenames:
    if not filename.endswith(".txt"):
        continue
    ans = {
        "filename": filename,
        "all_tokens": 0,
        "comment_tokens": [],
        "comments": []
    }
    with open(os.path.join(original_classeval_filepath, filename), "r") as f:
        line_number = 0
        f_lines = f.readlines()

        token_number = 0
        for line in f_lines:
            for token in tokens:
                if token in line:
                    ans["all_tokens"] += 1
                    if token == "<description for whole class>" or token == "<description for whole function>":
                        ans["comment_tokens"].append(line_number)
                        new_filename = filename.replace(".txt", ".json")
                        f1 = open_json(classeval_only_comments_filepath + "/" + new_filename)
                        ans["comments"].append(f1[token_number])
                    token_number += 1
            line_number += 1
    classeval_comments.append(ans)
classeval_comments = sorted(classeval_comments, key=lambda x: x["filename"])
save_json("classeval_comments.json", classeval_comments)
