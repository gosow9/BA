"""
This modul contains the morphological functions.
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


def get_edge_erroded(img, kernel):
    """
    Implements edge detection with an eroded image.

    :param img: 2d image
    :type img: InputArray
    :param kernel: errosion kernel
    :type kernel: InputArray
    :return: Processed image
    :return type: OutputArray
    """
    ret3, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    erode = cv2.erode(thresh, kernel, iterations=1)
    return cv2.subtract(thresh, erode)


def get_edge_dilated(img, kernel):
    """
    Implements edge detection with an dilated image.

    :param img: 2d image
    :type img: InputArray
    :param kernel: errosion kernel
    :type kernel: InputArray
    :return: Processed image
    :return type: OutputArray
    """
    ret3, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    dilat = cv2.dilate(thresh, kernel, iterations=1)
    return cv2.subtract(dilat, thresh)


def get_edge_grad(img, kernel):
    """
    Implements edge detection with an difference of a dilated and eroded image.

    :param img: 2d image
    :type img: InputArray
    :param kernel: errosion kernel
    :type kernel: InputArray
    :return: Processed image
    :return type: OutputArray
    """
    ret3, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    return cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)


def get_edge_errosion_dilation(img, kernel):
    """
    Implements edge detection with an difference of a dilated and eroded image.

    :param img: 2d image
    :type img: InputArray
    :param kernel: errosion kernel
    :type kernel: InputArray
    :return: Processed image
    :return type: OutputArray
    """
    ret3, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    erode = cv2.erode(thresh, kernel, iterations=1)
    dilat = cv2.dilate(thresh, kernel, iterations=1)
    return cv2.subtract(dilat, erode)
