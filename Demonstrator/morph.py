"""
This model contains the morphological functions.
"""
import cv2
import numpy as np


def opening_by_reconstruction(img, erosion_kernel, dilation_kernel):
    """
    Implements opening by reconstruction with given kernels.

    :param img: Source image
    :type img: InputArray
    :param erosion_kernel: Erosion kernel
    :type erosion_kernel: InputArray
    :param dilation_kernel: Dilation kernel
    :type dilation_kernel: InputArray
    :return: Processed image
    :return type: OutputArray
    """
    im_eroded = cv2.erode(img, erosion_kernel)
    im_opened = reconstruction_by_dilation(im_eroded, img, dilation_kernel)
    return im_opened


def reconstruction_by_dilation(img, mask, dilation_kernel):
    """
    Implements reconstruction by dilation with given kernel and mask.

    :param img: Eroded image
    :type img: InputArray
    :param mask: Source image
    :type mask: InputArray
    :param dilation_kernel: Dilation kernel
    :type dilation_kernel: InputArray
    :return: Processed image
    :return type: OutputArray
    """
    im_old = img.copy()
    while 1:
        img = cv2.dilate(img, dilation_kernel)
        img = cv2.bitwise_and(img, mask)
        if np.array_equal(im_old, img):
            break
        im_old = img.copy()
    return img
