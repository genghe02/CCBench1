import os

from tools import open_json, save_json

if __name__ == '__main__':
    # Check classeval due to its complex context
    classeval_comments = open_json('classeval_comments.json')
    dataset_dir_path = 'data/classeval_comments_version/comments_message/'
    error_filenames = []
    for item in classeval_comments:
        filename = item['filename'].split('.')[0] + '.json'
        file_full_path = os.path.join(dataset_dir_path, filename)
        print(file_full_path)
        f = open_json(file_full_path)
        if len(f) != item['all_tokens']:
            print(f)
            error_filenames.append(filename)
    save_json("classeval_errors.json", error_filenames)

