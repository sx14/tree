import config
from det.calibrator.laser.seg_laser import get_laser_points
from det.calibrator.laser.laser import Laser
from det.calibrator.tag.seg_tag import segment_tag
from det.calibrator.tag.tag import BlueTag
from util.show_image import *


def get_laser(im, im_id, debug=False):

    pt_pair, pt_mask, pt_conf, laser_mask = get_laser_points(im, debug)

    if debug:
        visualize_image(im, name='img', im_id=im_id, show=debug)
        visualize_image(pt_mask, name='pt', im_id=im_id, show=debug)

    if len(pt_pair) == 2:
        calibrator = Laser(pt_pair, pt_mask, pt_conf, laser_mask)
    else:
        calibrator = None
    return calibrator


def get_tag(im, vis=False):
    return None


def get_calibrator(im, im_id, debug=False):
    if config.CALIBRATOR == 'laser':
        return get_laser(im, im_id, debug)
    elif config.CALIBRATOR == 'tag':
        return get_tag(im, debug)
    else:
        return None