'''
This modul contains functions, which are usefull to handle countours and points in order to get the geometry pattern.
'''

import cv2
import numpy as np
from scipy.spatial import distance as dist


def rect_center(tl, tr, br, bl):
    """"
    Computes centerpoint of a rectangle given it's corners

    :param tl: Point in the top left
    :type tl: InputArray
    :param tr: Point in the top right
    :type tr: InputArray
    :param br: Point in the bottom right
    :type br: InputArray
    :param bl: Point in the bottom left
    :type bl: InputArray
    :return: Centerpoint
    :return type: Output Array
    """
    A = np.array([[tl[0] - br[0], tr[0] - bl[0]],
                  [tl[1] - br[1], tr[1] - bl[1]]])
    b = np.array([[tr[0] - tl[0]],
                  [tr[1] - tl[1]]])
    s = np.linalg.inv(A) @ b

    p = (tl - br) * s[0] + tl

    return (p[0], p[1])


def remove_contours(cnts, minArea, maxArea):
    """
    Removes contours outside the definded to areas

    :param cnts: Input Contours
    :type cnts: InputArray
    :param minArea:
    :param maxArea:
    :return: Output Contours
    :return type: OutputArray
    """

    r = []
    for i in reversed(range(len(cnts))):
        area = cv2.contourArea(cnts[i])
        if area > maxArea or area < minArea:
            r.append(i)

    for i in r:
        cnts.pop(i)

    return cnts


def remap_contours(cnts, map_x, map_y):
    """
    Remaps the contours points

    :param cnts: Contours
    :type cnts: InputArray
    :param map_x: Undistortion map in x
    :type map_x: InputArray
    :param map_y: Undistortion map in y
    :type map_y: InputArray
    :return: Remaped Contours
    :return type: OutputArray
    """
    for c in cnts:
        x_tmp = c[0][0][0]
        y_tmp = c[0][0][1]

        c[0][0][0] = map_x[y_tmp][x_tmp]
        c[0][0][1] = map_y[y_tmp][x_tmp]

    return cnts


def sort_points(p):
    """
    Sorts a set of four points

    :param p: set of four points
    :type p: InputArray
    :return: Sorted set of points
    :return type: OutputArray
    """

    # sort with respect to x
    xsort = sorted(p, key=lambda a: a[0])

    # get left and right points
    l = xsort[:2]
    r = xsort[2:]

    # sort with respect to y
    tl, bl = sorted(l, key=lambda a: a[1])
    tr, br = sorted(r, key=lambda a: a[1])

    return [tl, bl, tr, br]
