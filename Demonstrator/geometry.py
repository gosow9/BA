'''
This modul contains functions, which are usefull to handle countours and points in order to get the geometry pattern.
'''

import cv2
import numpy as np
from scipy.spatial import distance as dist

def midpoint(ptA, ptB):
    """
    Computes the midpoint of to points

    :param ptA: First point
    :type ptA: InputArray
    :param ptB: Second Point
    :type ptB: InputArray
    :return: Midpoint
    :return type: OutputArray
    """
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


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
    A = np.array([[tl[0]-br[0], tr[0]-bl[0]],
                  [tl[1]-br[1], tr[1]-bl[1]]])
    b = np.array([[tr[0]-tl[0]],
                  [tr[1]-tl[1]]])
    s = np.linalg.inv(A)@b

    p = (tl-br)*s[0] + tl

    return (p[0], p[1])

def order_points(pts):
    """
    Orders four points clockwise (top left first)
    :param pts: Four points
    :type pts: InputArray
    :return: Ordered points
    :return type: OutputArray
    """
    xSorted = pts[np.argsort(pts[:, 0]), :]

    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    return np.array([tl, tr, br, bl], dtype="float32")

def combine_contours(cnts):
    """
    Combines a set of contours (points) into one single contour

    :param cnts: Contours
    :type cnts: InputArray
    :return: Combined Contour
    :return type: OutputArray
    """
    c = []
    for i in range(len(cnts)):
        for j in range(len(cnts[i])):
            c.append([[cnts[i][j][0][0], cnts[i][j][0][1]]])

    return np.array(c)

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