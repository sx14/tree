from file_loader import *
import matplotlib.pyplot as plt


def eval(anno_path, result_path):
    if anno_path.endswith('txt'):
        anno_db = load_txt_anno(anno_path)
    elif anno_path.endswith('xlsx'):
        anno_db = load_excel_anno(anno_path)
    else:
        return

    results = load_measure_results(result_path)

    diff = []
    diff_r = []
    for tree_id in anno_db:
        if tree_id in results:
            width_measure = results[tree_id]['width']
            width_anno = anno_db[tree_id]['width'] * 10
            diff.append(width_anno - width_measure)
            diff_r.append((width_anno - width_measure) / width_anno)


    plt.hist(diff, 20)
    plt.show()

    plt.hist(diff_r, 20)
    plt.show()


if __name__ == '__main__':
    result_path = '../results.json'
    anno_path = '../data/anno/actual_values_2.txt'
    eval(anno_path, result_path)

