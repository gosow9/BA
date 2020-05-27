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
        capture_height=1848,
        display_width=3280,
        display_height=1848,
        framerate=21,
        flip_method=0,
):
    """
    Configures gstreamer string
    :param capture_width: Capture width of camera
    :type capture_width: Int
    :param capture_height: Capture height of camera
    :type capture_height: Int
    :param display_width: Display width of the captured image
    :type display_width: Int
    :param display_height: Display height of the captured image
    :type display_height: Int
    :param framerate: Frames per second
    :type framerate: Int
    :param flip_method: Orientation of the image
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


def image_found(img):
    return True
    width, height = img.shape
    var1 = np.mean(img[:, 0])
    var2 = np.mean(img[:, width - 1])
    _, cross1 = cv2.threshold(img[:, int(width/2)], var1 / 2, 1, cv2.THRESH_BINARY)
    val = np.sum(cross1)
    val1 = np.sum(cross1)
    if width - val > 10:
        _, border1 = cv2.threshold(img[:, 0], var1 / 3, 1, cv2.THRESH_BINARY)
        _, border2 = cv2.threshold(img[:, width - 1], var2 / 3, 1, cv2.THRESH_BINARY)
        val1 = np.sum(border1)
        val2 = np.sum(border2)
        print(var1, var2, val1, val2, width)
        if width - val1 < 10 and width - val2 < 10:
            return True
        else:
            print("Over line")
            return False
    else:
        print("is empty")
        return False

if __name__ == "__main__":
    # load calibration params
    mtx = np.eye(3, 3)  # np.loadtxt('intrinsics/mtx1.txt')
    dst = None  # np.loadtxt('intrinsics/dist1.txt')
    w = 3280
    h = 1848
    # get calibration map
    map_x, map_y = cv2.initUndistortRectifyMap(mtx, dst, None, mtx, (w, h), cv2.CV_32FC1)

    # separation from edge
    sep = 700

    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline(flip_method=0))
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_NORMAL)
        # Window
        t = time.time()
        background = np.ones((2464, 3280), np.uint8)
        im = np.ones((2464, 3280), np.uint8)
        while time.time() - t < 5 and cv2.getWindowProperty("CSI Camera", 0) >= 0:
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
        #background = background * thresh_value - im
        print("Entering measurment mode")
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        grad = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        fps = 0
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            t_ref = time.time()
            ret_val, img = cap.read()
            imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if image_found(imgray):
                # Diffrent types to get the edge use one:
                #edge = get_edge_errosion_dilation(imgray, grad)
                edge = get_edge_erroded(imgray, kernel)
                #edge = get_edge_dilated(imgray, kernel)
                #edge = get_edge_grad(imgray, grad)

                # find geometry pattern (upper and lower)
                cnts_upper, _ = cv2.findContours(edge[0:sep, :], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cnts_lower, _ = cv2.findContours(edge[h - sep:h, :], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                # only keep valid contours
                cnts_upper = remove_contours(cnts_upper, 2000, 3000)
                cnts_lower = remove_contours(cnts_lower, 2000, 3000)

                # reject frame if amount of contours is invalid
                if len(cnts_upper) != 2 or len(cnts_lower) != 2:
                    print('No pattern detected')
                    #continue

                # undistort contours
                cnts_upper = remap_contours(cnts_upper, map_x, map_y)
                cnts_lower = remap_contours(cnts_lower, map_x, map_y)

                # collect image points (geometry pattern)
                imgp_upper = np.zeros((5, 2))
                index = 0

                for c in cnts_upper:
                    box = cv2.minAreaRect(c)
                    box = cv2.boxPoints(box)

                    (tl, tr, br, bl) = box
                    p = rect_center(tl, tr, br, bl)
                    imgp_upper[index] = p
                    index += 1

                imgp_lower = np.zeros((5, 2))
                index = 0

                for c in cnts_lower:
                    box = cv2.minAreaRect(c)
                    box = cv2.boxPoints(box)

                    (tl, tr, br, bl) = box
                    tl[1] += (h - sep)
                    tr[1] += (h - sep)
                    br[1] += (h - sep)
                    bl[1] += (h - sep)
                    p = rect_center(tl, tr, br, bl)
                    imgp_lower[index] = p
                    index += 1

                # sort the points
                imgp_upper.sort(axis=0)
                imgp_lower.sort(axis=0)

                # combine points into one array
                imgp = np.concatenate([imgp_lower, imgp_upper])

                # get pixels per metric unit
                ppm_array = [dist.euclidean(imgp[0], imgp[4]) / 240, dist.euclidean(imgp[5], imgp[9]) / 240,
                             dist.euclidean(imgp[1], imgp[3]) / 160, dist.euclidean(imgp[6], imgp[8]) / 160]

                for i in range(5):
                    ppm_array.append(dist.euclidean(imgp[i], imgp[5 + i]) / 160)

                ppm = np.mean(ppm_array)

                # get the principal point
                mx = mtx[0][2]
                my = mtx[1][2]

                # generate array with objectpoints
                objp = np.array([[-120 * ppm + mx, -80 * ppm + my],
                                 [-60 * ppm + mx, -80 * ppm + my],
                                 [mx, -80 * ppm + my],
                                 [60 * ppm + mx, -80 * ppm + my],
                                 [120 * ppm + mx, -80 * ppm + my],
                                 [-120 * ppm + mx, 80 * ppm + my],
                                 [-60 * ppm + mx, 80 * ppm + my],
                                 [mx, 80 * ppm + my],
                                 [60 * ppm + mx, 80 * ppm + my],
                                 [120 * ppm + mx, 80 * ppm + my]], dtype=np.float32)

                # get transformation matrix
                T, _ = cv2.findHomography(imgp, objp, method=0)

                # find object-contours inside the separation
                cnts_m, _ = cv2.findContours(edge[sep:(h - sep), :], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                # undistort contours
                cnts_m = remap_contours(cnts_m, map_x, map_y)

                # reject frame if cnts_m is empty
                if not cnts_m:
                    print('No object found')
                    #continue

                # combine found contours to one
                cnts_m = np.concatenate(cnts_m, axis=0).astype(np.float64)

                # warp perspective of the contours
                cnts_m = cv2.perspectiveTransform(cnts_m, T)

                # minimum area rectangle around cnts_m
                box = cv2.minAreaRect(cnts_m.astype(np.float32))
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

                # Test ppm
                ppm = 1
                # compute the size of the object
                dimA = dA / ppm
                dimB = dB / ppm
                if fps == 0:
                    fps = 1 / (time.time() - t_ref)
                else:
                    fps = 0.9 * fps + 0.1 / (time.time() - t_ref)
                # the larger is the lenght
                if dimA > dimB:
                    print('l = {:.2f}, w = {:.2f}'.format(dimA, dimB))

                else:
                    print('l = {:.2f}, w = {:.2f}'.format(dimB, dimA))

                # Show the edges for visual control
                cv2.resizeWindow("CSI Camera", 820, 616)
                cv2.putText(edge, "{:.2f} fps".format(fps), (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)
                cv2.imshow("CSI Camera", edge)

                # This also acts as
                keyCode = cv2.waitKey(30) & 0xFF
                # Stop the program on the ESC key
                if keyCode == 27:
                    break
            else:
                if fps == 0:
                    fps = 1 / (time.time() - t_ref)
                else:
                    fps = 0.9 * fps + 0.1 / (time.time() - t_ref)
                print("No Object found, FPS{}".format(fps))
                cv2.resizeWindow("CSI Camera", 820, 616)
                cv2.putText(imgray, "{:.2f} fps".format(fps), (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255),
                            3)
                cv2.imshow("CSI Camera", imgray)

                # This also acts as
                keyCode = cv2.waitKey(30) & 0xFF
                # Stop the program on the ESC key
                if keyCode == 27:
                    break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")
