
def load_txt_anno(anno_path):
    with open(anno_path) as f:
        annos = f.readlines()
        annos = [line.strip().split(',') for line in annos]
    id2info = {}
    for anno in annos:
        id2info[anno[0]] = {
            'type': 'unknown',
            'width': float(anno[1])
        }
    return id2info


def load_excel_anno(anno_path):
    import xlrd
    wb = xlrd.open_workbook(filename=anno_path)
    sheet = wb.sheet_by_index(0)
    id2info = {}
    for r in range(1, sheet.nrows):
        raw_values = sheet.row_values(r)
        tree_type = raw_values[1]
        tree_id = raw_values[4][-4:]
        tree_width = float(raw_values[5])
        id2info[tree_id] = {
            'type': tree_type,
            'width': tree_width
        }
    return id2info


def load_measure_results(save_path):
    import json
    with open(save_path) as f:
        results = json.load(f)
        results = results['results']

    id2info = {}
    for result in results:
        image_path = result['image_path']
        tree_id = image_path.split('/')[-1][:-4]
        id2info[tree_id] = result

    return id2info