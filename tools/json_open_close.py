import json


# 打开json格式文件
def open_json(filename):
    """
        :param filename: 你要打开的json文件名
        :return: None
    """
    f = open(filename, encoding='utf-8')
    objects = json.load(f)
    f.close()
    return objects


# 保存json格式文件
def save_json(filename, objects):
    """
        :param filename: 你要保存的文件名
        :param objects: 你要保存的内容
        :return: None

        Warning：会覆盖原有内容，谨慎！
    """
    f = open(filename, 'w')
    json.dump(objects, f)
    f.close()