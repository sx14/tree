# coding: utf-8
from det.tag.seg_tag import *
from det.tag.cal_tag import *
from util.show_image import *
from util.resize_image import *
from config import IMG_MAX_HEIGHT

if __name__ == '__main__':
    """
    实验确定镜头焦距FOCAL_LENGTH
    单位是pixel
    """

    img_root = '../data/tag'

    focal_dis_arr = []
    focal_dis_w_arr = []
    focal_dis_h_arr = []

    print('Image height(resized): %d' % IMG_MAX_HEIGHT)
    print('dis mm: focal_w pix | focal_h pix')
    print('---------------------------------')
    for i, img_id in enumerate(os.listdir(img_root)):
        im_path = os.path.join(img_root, img_id)
        im = cv2.imread(im_path)
        im = resize_image(im)

        tag_mask = segment_tag(im)

        tag = BlueTag(tag_mask)
        show = tag.show_lines()
        # show_image('', show)

        real_width = tag.WIDTH
        pixel_width = tag.pixel_width()

        real_height = tag.HEIGHT
        pixel_height = tag.pixel_height()

        tag2lens = int(img_id.split('.')[0])*10.0

        focal_dis_w = tag2lens * (pixel_width / real_width)
        focal_dis_h = tag2lens * (pixel_height / real_height)
        print('%d mm: %.2f pix | %.2f pix' % (tag2lens, focal_dis_w, focal_dis_h))
        focal_dis_arr.append((focal_dis_w+focal_dis_h)/2.0)
        focal_dis_h_arr.append(focal_dis_h)
        focal_dis_w_arr.append(focal_dis_w)

    print('>>>>>>>>>>>>>>>>>>>>>')
    focal_dis_w_arr = np.array(focal_dis_w_arr)
    focal_dis_h_arr = np.array(focal_dis_h_arr)
    print('VAR: w(%.2f) | h(%.2f)' % (focal_dis_w_arr.var(), focal_dis_h_arr.var()))

    focal_dis = sum(focal_dis_arr) / len(focal_dis_arr)
    print('FOCAL_LENGTH=%.4f pix' % focal_dis)




