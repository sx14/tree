
def show_masked_image(im, mask, win_name='show'):
    import numpy as np
    im_red = mask.astype(np.uint8)
    im_red[im_red > 0] = 255

    im_show = im.copy().astype(np.uint8)
    im_show = im_show / 2
    im_red = im_red / 2
    im_show[:, :, 1] = im_show[:, :, 1] + im_red
    show_image(im_show, win_name)


def show_image(im, win_name='show'):
    import cv2
    im_h = im.shape[0]
    im_w = im.shape[1]
    ratio = 800.0 / im_h

    cv2.namedWindow(win_name, 0)
    cv2.resizeWindow(win_name, int(im_w * 1.0 * ratio), int(im_h * 1.0 * ratio))
    cv2.moveWindow(win_name, 200, 200)
    cv2.imshow(win_name, im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_images(ims, name='show'):
    import cv2
    left_top_x = 0

    for i, im in enumerate(ims):
        win_name = 'name%s[%d]' % (name, i+1)
        im_h = im.shape[0]
        im_w = im.shape[1]
        ratio = 800.0 / im_h

        cv2.namedWindow(win_name, 0)
        cv2.resizeWindow(win_name, int(im_w * 1.0 * ratio), int(im_h * 1.0 * ratio))
        cv2.moveWindow(win_name, left_top_x, 0)
        cv2.imshow(win_name, im)
        left_top_x += int(im_w * 1.0 * ratio)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_pts(im, pts):
    import matplotlib.pyplot as plt
    """Draw points"""
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in range(0, len(pts)):

        pt = pts[i]
        ax.add_patch(
            plt.Rectangle((pt[0]-1, pt[1]-1),
                          3,
                          3, fill=True,
                          edgecolor=[255.0/255, 20.0/255, 147.0/255], linewidth=1))
    plt.axis('off')
    plt.tight_layout()
    plt.show()
