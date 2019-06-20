# coding: utf-8


def save_results(results, save_path):
    """
    保存json格式数据
    :param results: dict
    :param save_path: json文件绝对路径
    """
    import json
    if save_path is None:
        return
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)


def print_json(results):
    """
    打印json格式数据
    :param results: dict
    """
    import json
    print(json.dumps(results))


def load_image_list(image_list_path):
    """
    加载图像路径列表
    :param image_list_path: 图像列表文件路径
    :return: 图像路径列表
    """
    import os
    if os.path.exists(image_list_path):
        with open(image_list_path) as f:
            image_paths = [l.strip() for l in f.readlines()]
        return image_paths
    else:
        return None


def parse_image_list(raw_list):
    image_paths = raw_list.split('\n')
    return image_paths