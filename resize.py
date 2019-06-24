import os
import cv2
import argparse

import config


def parse_args():
    parser = argparse.ArgumentParser(description='TreeMeasure(resize) V1.1')
    parser.add_argument('-i', '--input',    help='Please provide input image path.',    default='None', required=True)
    parser.add_argument('-o', '--output',   help='Please provide output image path.',   default='None', required=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    input_path = args.input
    output_path = args.output

    if os.path.exists(input_path):
        im = cv2.imread(input_path)
        resize_size = (config.SAVE_WIDTH, config.SAVE_HEIGHT)
        im_resized = cv2.resize(im, resize_size)

        cv2.imwrite(output_path, im_resized)
        print(0)
    else:
        print(-1)
