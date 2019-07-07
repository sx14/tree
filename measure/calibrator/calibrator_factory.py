import config
from measure.calibrator.laser.seg_laser import get_laser_points
from measure.calibrator.laser.laser import Laser
from measure.calibrator.tag.seg_tag import segment_tag
from measure.calibrator.tag.tag import BlueTag
from util.show_image import *


def get_laser(im, im_id, debug=False):

    pt_pair, pt_mask, pt_conf, laser_mask = get_laser_points(im, debug)

    if debug:
        visualize_image(im, name='img', im_id=im_id)
        visualize_image(pt_mask, name='pt', im_id=im_id)

    if len(pt_pair) == 2:
        calibrator = Laser(pt_pair, pt_mask, pt_conf, laser_mask)
    else:
        calibrator = None
    return calibrator


def get_tag(im, im_id, debug=False):
    tag_mask = segment_tag(im)

    if tag_mask is None:
        return None
    else:
        if debug:
            visualize_image(tag_mask, name='tag', im_id=im_id)
        tag = BlueTag(tag_mask)
        if tag.is_available():
            return tag
        else:
            return None


def get_calibrator(im, im_id, debug=False):
    if config.CALIBRATOR == 'laser':
        return get_laser(im, im_id, debug)
    elif config.CALIBRATOR == 'tag':
        return get_tag(im, im_id, debug)
    else:
        return None