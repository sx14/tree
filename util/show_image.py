
def show_masked_image(im, mask, win_name='show', show=False):
    import numpy as np
    im_red = mask.astype(np.uint8)
    im_red[im_red > 0] = 255

    im_show = im.copy().astype(np.uint8)
    im_show = im_show / 2
    im_red = im_red / 2
    im_show[:, :, 1] = im_show[:, :, 1] + im_red
    visualize_image(im_show, win_name, show=show)


def visualize_image(im, name='temp', im_id='temp', show=False):
    import cv2

    im_h = im.shape[0]
    im_w = im.shape[1]
    ratio = 800.0 / im_h

    if len(im.shape) == 2:
        im_show = im.copy()
        im_show[im_show > 0] = 255
    else:
        im_show = im

    if show:
        cv2.namedWindow(name, 0)
        cv2.resizeWindow(name, int(im_w * 1.0 * ratio), int(im_h * 1.0 * ratio))
        cv2.moveWindow(name, 200, 200)
        cv2.imshow(name, im_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        import os
        from config import PROJECT_ROOT
        save_dir = os.path.join(PROJECT_ROOT, 'vis', im_id)
        if im_show.shape == 2:
            suffix = 'png'
        else:
            suffix = 'jpg'
        if os.path.exists(save_dir):
            cv2.imwrite(os.path.join(save_dir, name + '.' + suffix), im_show)
        else:
            os.mkdir(save_dir)


def show_images(ims, name='show'):
    import cv2
    left_top_x = 0

    for i, im in enumerate(ims):
        win_name = 'name%s[%d]' % (name, i+1)
        im_h = im.shape[0]
        im_w = im.shape[1]
        ratio = 800.0 / im_h

        if len(im.shape) == 2:
            im_show = im.copy()
            im_show[im_show > 0] = 255
        else:
            im_show = im

        cv2.namedWindow(win_name, 0)
        cv2.resizeWindow(win_name, int(im_w * 1.0 * ratio), int(im_h * 1.0 * ratio))
        cv2.moveWindow(win_name, left_top_x, 0)
        cv2.imshow(win_name, im_show)
        left_top_x += int(im_w * 1.0 * ratio)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_pts(im, pts, name='final', im_id='0', show=False):
    h, w, _ = im.shape
    for pt in pts:
        x, y = pt
        box_xmin = max(0, x-8)
        box_ymin = max(0, y-8)
        box_xmax = min(x+8, w-1)
        box_ymax = min(y+8, h-1)
        im[box_ymin:box_ymax+1, box_xmin:box_xmax+1] = [147, 20, 255]

    visualize_image(im, name=name, im_id = im_id, show=show)