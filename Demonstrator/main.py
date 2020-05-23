# -*- coding: utf-8 -*-
"""
main module
"""
import cv2
import numpy as np
import time
from scipy.spatial import distance as dist
from morph import *
from geometry import *



def gstreamer_pipeline(
    capture_width=3280,
    capture_height=2464,
    display_width=3280,
    display_height=2464,
    framerate=20,
    flip_method=0,
):
    """
    Configures gstreamer string
    :param capture_width: capture width of camera
    :type capture_width: Int
    :param capture_height: capture height of camera
    :type capture_height: Int
    :param display_width: display width of the captured image
    :type display_width: Int
    :param display_height: display height of the captured image
    :type display_height: Int
    :param framerate: frame
    :type framerate: Int
    :param flip_method:
    :type flip_method: Int
    :return: String 
    """
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )



if __name__ == "__main__":
    # load calibration params
    mtx = np.eye(3,3)#np.loadtxt('intrinsics/mtx_lowdist.txt')
    dst = None#np.loadtxt('intrinsics/dist_lowdist.txt')
    w = 3280
    h = 2464
    # get calibration map
    map_x, map_y = cv2.initUndistortRectifyMap(mtx, dst, None, mtx, (w, h), cv2.CV_32FC1)

    # separation from edge
    sep = 1000

    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline(flip_method=0))
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_NORMAL)
        # Window
        t = time.time()
        background = np.ones((2464, 3280), np.uint8)
        im = np.ones((2464, 3280), np.uint8)
        while time.time()-t < 5 and cv2.getWindowProperty("CSI Camera", 0) >= 0:
            ret_val, img = cap.read()
            im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            print("Init mode")
            thresh_value = np.max(im)
            print(thresh_value)
            cv2.imshow("CSI Camera", im)
            keyCode = cv2.waitKey(30) & 0xFF
            # Stop the program on the ESC key
            if keyCode == 27:
                break
        background = background * thresh_value - im
        print("Entering measurment mode")
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        grad = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        fps = 0
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            t_ref = time.time()
            ret_val, img = cap.read()
            # Diffrent types to get the edge use one:
            #out = get_edge_errosion_dilation(img, grad)
            out = get_edge_erroded(img, kernel)
            #out = get_edge_dilated(img, kernel)
            #out = get_edge_grad(img, grad)

            # undistort edges
            edge = cv2.remap(out, map_x, map_y, cv2.INTER_LINEAR)

            # find geometry pattern (upper and lower)
            cnts_upper, _ = cv2.findContours(edge[0:sep, :], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnts_lower, _ = cv2.findContours(edge[h - sep:h, :], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # only keep valid contours
            cnts_upper = remove_contours(cnts_upper, 2000, 3000)
            cnts_lower = remove_contours(cnts_lower, 2000, 3000)

            # reject frame if amount of contours is invalid
            if len(cnts_upper) != 2 or len(cnts_lower) != 2:
                print('No calibration points detected')
                continue

            # collect image points (geometry pattern)
            imgp = np.zeros((4, 2))
            index = 0
            for c in cnts_upper:
                box = cv2.minAreaRect(c)
                box = cv2.boxPoints(box)

                (tl, tr, br, bl) = box
                p = rect_center(tl, tr, br, bl)
                imgp[index] = p
                index += 1

            for c in cnts_lower:
                box = cv2.minAreaRect(c)
                box = cv2.boxPoints(box)

                (tl, tr, br, bl) = box
                tl[1] += (h - sep)
                tr[1] += (h - sep)
                br[1] += (h - sep)
                bl[1] += (h - sep)
                p = rect_center(tl, tr, br, bl)
                imgp[index] = p
                index += 1

            # order image points
            imgp = order_points(imgp)

            # get pixels per metric unit
            ppm = dist.euclidean(imgp[0], imgp[1]) / 200
            fx = mtx[0][0]
            fy = mtx[1][1]
            mx = mtx[0][2]
            my = mtx[1][2]
            world_x = -mx / fx
            world_y = -my / fy

            # points in world coordiantes, format: (px*ppm-mx)/fx, (py*ppm-my)/fy
            world_p = np.array([[world_x, world_y, 0],
                                [(200 * ppm - mx) / fx, world_y, 0],
                                [(200 * ppm - mx) / fx, (140 * ppm - my) / fy, 0],
                                [world_x, (140 * ppm - my) / fy, 0]], dtype=np.float32)

            # project to image-plane
            world_p, _ = cv2.projectPoints(world_p, (0, 0, 0), (0, 0, 0), mtx, dst)

            # get objectpoints, top left point (imgp[0]) is reference
            objp = np.concatenate([world_p[0] + imgp[0], world_p[1] + imgp[0], world_p[2] + imgp[0], world_p[3] + imgp[0]])

            # get transformation matrix
            T = cv2.getPerspectiveTransform(imgp, objp)

            # warp edges
            edge = cv2.warpPerspective(edge, T, (w, h))

            # find object-contours inside the separation
            cnts_m, _ = cv2.findContours(edge[sep:(h - sep), :], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # reject frame if cnts_m is empty
            if not cnts_m:
                print('No Object found')
                continue

            # combine found contours to one
            cnts_m = combine_contours(cnts_m)

            # minimum area rectangle around cnts_m
            box = cv2.minAreaRect(cnts_m)
            box = cv2.boxPoints(box)
            box[0][1] += sep
            box[1][1] += sep
            box[2][1] += sep
            box[3][1] += sep

            # get the midpoints of the rectangle
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            # compute the euclidean distance between the midpoints
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            # compute the size of the object
            dimA = dA / ppm
            dimB = dB / ppm
            if fps == 0:
                fps = 1/(time.time()-t_ref)
            else:
                fps = 0.9*fps + 0.1/(time.time()-t_ref)
            # the larger is the lenght
            if dimA > dimB:
                print('l = {:.2f}, w = {:.2f}'.format(dimA, dimB))

            else:
                print('l = {:.2f}, w = {:.2f}'.format(dimB, dimA))

            # Show the edges for visual control
            cv2.resizeWindow("CSI Camera", 820, 616)
            cv2.putText(out, "FPS:{:.3f}".format(fps), (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)
            cv2.imshow("CSI Camera", out)

            # This also acts as
            keyCode = cv2.waitKey(30) & 0xFF
            # Stop the program on the ESC key
            if keyCode == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")
