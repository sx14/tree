import os

data_path = '../data/tree_flipped'
# data_path = '/media/sunx/Data/dataset/tree-dataset/test'

image_list = []
for img_id in os.listdir(data_path):
    image_rel_path = os.path.join(data_path, img_id)
    image_abs_path = os.path.abspath(image_rel_path)
    image_list.append(image_abs_path+'\n')

image_list = image_list[:5]
save_path = '../image_list.txt'
with open(save_path, 'w') as f:
    f.writelines(image_list)