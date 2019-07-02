import os
import cv2
import argparse
import traceback

import config
from util.my_logger import save_log


def parse_args():
    parser = argparse.ArgumentParser(description='TreeMeasure(resize) V1.1')
    parser.add_argument('-i', '--input',    help='Please provide input image path.',    required=True)
    parser.add_argument('-o', '--output',   help='Please provide output image path.',   required=True)
    parser.add_argument('-W', '--width', help='Resize width.', type=int)
    parser.add_argument('-H', '--height', help='Resize width.', type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    try:
        args = parse_args()
        input_path = args.input
        output_path = args.output
        width = args.width
        height = args.height

        if width is None:
            width = config.SAVE_WIDTH
        if height is None:
            height = config.SAVE_HEIGHT

        if os.path.exists(input_path):
            im = cv2.imread(input_path)
            resize_size = (width, height)
            im_resized = cv2.resize(im, resize_size)
            cv2.imwrite(output_path, im_resized)
            print(0)
        else:
            print(-1)
    except:
        save_log(traceback.format_exc())