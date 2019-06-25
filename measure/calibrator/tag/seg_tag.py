# coding: utf-8
import os
import cv2
import numpy as np
from util.show_image import *


def segment_tag(im):
    im_h, im_w, _ = im.shape
    im = im.astype(np.int32)
    im_bin = im[:, :, 0] - im[:, :, 1] - im[:, :, 2]
    im_bin[im_bin < 10] = 0
    im_bin[im_bin > 0] = 1
    im_bin = im_bin.astype(np.uint8)
    # 连通区域
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(im_bin)

    # 假设目标在图像中心
    # 找到图像中心所在的连通区域
    im_h, im_w = im_bin.shape
    im_ct_y = int(im_h / 2.0)
    im_ct_x = int(im_w / 2.0)
    center_comp_label = labels[im_ct_y, im_ct_x]
    if center_comp_label > 0:
        labels[labels != center_comp_label] = 0
        labels[labels > 0] = 1
        tag_mask = labels.astype(np.uint8)
        return tag_mask
    else:
        return None


def crop_image(im, tag_mask, n_tag_w=4, n_tag_h=1):
    im_h, im_w, _ = im.shape

    tag_ys, tag_xs = np.where(tag_mask > 0)
    ymin = tag_ys.min()
    ymax = tag_ys.max()
    xmin = tag_xs.min()
    xmax = tag_xs.max()
    tag_h = ymax - ymin
    tag_w = xmax - xmin

    # default:
    # crop width:  9 * tag_width
    # crop height: 3 * tag_height
    crop_ymin = np.maximum(ymin - int(n_tag_h * tag_h), 0)
    crop_ymax = np.minimum(ymax + int(n_tag_h * tag_h), im_h)
    crop_xmin = np.maximum(xmin - int(n_tag_w * tag_w), 0)
    crop_xmax = np.minimum(xmax + int(n_tag_w * tag_w), im_w)

    im_patch = im[crop_ymin: crop_ymax+1, crop_xmin: crop_xmax+1, :]
    tag_mask_patch = tag_mask[crop_ymin: crop_ymax + 1, crop_xmin: crop_xmax + 1]

    # segmentation region box
    im_cand_box = tag_mask.copy()
    im_cand_box[crop_ymin: crop_ymax+1, crop_xmin: crop_xmax+1] = 255
    im_cand_box[crop_ymin+3: crop_ymax-2, crop_xmin+3: crop_xmax-2] = 0
    return im_patch, tag_mask_patch, im_cand_box


def remove_tag(im, tag_map):
    """
    用<蓝色标签>上方下方的树皮颜色
    掩盖<蓝色标签>
    :param im: 图像(cropped)
    :param tag_map: 标签掩码
    :return: 去除标签后的图像
    """
    im_h, im_w, _ = im.shape

    tag_ys, tag_xs = np.where(tag_map > 0)
    ymin = tag_ys.min()
    ymax = tag_ys.max()
    xmin = tag_xs.min()
    xmax = tag_xs.max()

    margin = 40
    ymin_above = np.maximum(ymin - margin, 0)
    region_above_tag = im[ymin_above:ymin, xmin:xmax, :]
    ymax_below = np.minimum(ymax + margin, im_h)
    region_below_tag = im[ymax:ymax_below, xmin:xmax, :]

    above_below_region = np.concatenate((region_above_tag, region_below_tag), axis=0)
    tree_color_b = np.mean(above_below_region[:, :, 0])
    tree_color_g = np.mean(above_below_region[:, :, 1])
    tree_color_r = np.mean(above_below_region[:, :, 2])
    im[tag_map>0, 0] = tree_color_b
    im[tag_map>0, 1] = tree_color_g
    im[tag_map>0, 2] = tree_color_r
    return im


def remove_tag1(im, tag_map):
    """
    用<蓝色标签>上方的树皮
    掩盖<蓝色标签>
    :param im: 图像(cropped)
    :param tag_map: 标签掩码
    :return: 去除标签后的图像
    """
    im_h, im_w, _ = im.shape

    tag_ys, tag_xs = np.where(tag_map > 0)
    ymin = tag_ys.min()
    ymax = tag_ys.max()
    xmin = tag_xs.min()
    xmax = tag_xs.max()

    margin = 40
    ymin_above = np.maximum(ymin - margin, 0)
    region_above_tag = im[ymin_above:ymin, xmin:xmax, :]
    ymax_below = np.minimum(ymax + margin, im_h)
    region_below_tag = im[ymax:ymax_below, xmin:xmax, :]

    above_below_region = np.concatenate((region_above_tag, region_below_tag), axis=0)
    tree_color_b = np.mean(above_below_region[:, :, 0])
    tree_color_g = np.mean(above_below_region[:, :, 1])
    tree_color_r = np.mean(above_below_region[:, :, 2])
    im[tag_map > 0, 0] = tree_color_b
    im[tag_map > 0, 1] = tree_color_g
    im[tag_map > 0, 2] = tree_color_r

    ymin_above = np.maximum(ymin-(ymax-ymin), 0)
    region_above = im[ymin_above:ymin, xmin:xmax, :]
    im[ymin:ymin+(ymin-ymin_above), xmin:xmax, :] = region_above

    return im