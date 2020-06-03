"""
This script estimates the velocity of a moving object in pixel/sec
"""

import cv2
import numpy as np
import time
from scipy.spatial import distance as dist
from morph import *
from geometry import *

# Global variables
w = 3264
h = 1848

def gstreamer_pipeline(
        capture_width=w,
        capture_height=h,
        display_width=w,
        display_height=h,
        framerate=28,
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


if __name__ == "__main__":
    # separation from edge
    sep = 250

    # kernel for edge-detection
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_NORMAL)
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            t_ref = time.time()
            ret_val, img = cap.read()

            imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # variance of the first column
            var = np.var(imgray[sep:h - sep:10, 0])

            velocity = []
            imsave = []
            timestamp = []

            # check frames if the object is in the first column
            if var > 60:
                print("search...")

                # save the next 4 frames
                for i in range(4):
                    timestamp.append(time.time())
                    _, imgs = cap.read()
                    imsave.append(imgs)

                for i in range(len(imsave)):
                    cv2.imwrite('im{:}.png'.format(i), imsave[i])

                # edge detection
                edge_ref = get_edge_erroded(imgray, kernel)

                # find object-contours inside the separation
                cnts_ref, _ = cv2.findContours(edge_ref[sep:(h - sep), :], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                # remove invalid contours
                cnts_ref = remove_contours(cnts_ref, 4000, 2000000)

                # reject frame if cnts_m is empty
                if not cnts_ref:
                    print('No reference object found')
                    continue

                cnts_ref = np.concatenate(cnts_ref, axis=0).astype(np.float64)

                # minimum area rectangle around cnts_m
                box = cv2.minAreaRect(cnts_ref.astype(np.float32))
                box = cv2.boxPoints(box)

                box[0][1] += sep
                box[1][1] += sep
                box[2][1] += sep
                box[3][1] += sep

                # sort the points clockwise
                (tl_ref, bl_ref, tr_ref, br_ref) = sort_points(box)

                # search through the next frames for the object
                for i in range(len(imsave)):
                    gray = cv2.cvtColor(imsave[i], cv2.COLOR_BGR2GRAY)

                    # compute variance at the end
                    var_end = np.var(gray[sep:h - sep:4, w - 1])

                    # check if object is out of frame
                    if var_end < 60:
                        edge = get_edge_erroded(gray, kernel)

                        # find object-contours inside the separation
                        cnts, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                        # remove invalid contours
                        cnts = remove_contours(cnts, 10000, 2000000)

                        # reject frame if cnts_m is empty
                        if not cnts:
                            print('No object found, loop {:}'.format(i))
                            continue

                        cnts = np.concatenate(cnts, axis=0).astype(np.float64)

                        # minimum area rectangle around cnts_m
                        box = cv2.minAreaRect(cnts.astype(np.float32))
                        box = cv2.boxPoints(box)

                        # sort the points clockwise
                        (tl, bl, tr, br) = sort_points(box)

                        dist_t = tr[0] - tr_ref[0]
                        dist_b = br[0] - br_ref[0]

                        v_t = dist_t / (timestamp[i] - t_ref)
                        v_b = dist_b / (timestamp[i] - t_ref)

                        velocity.append((v_t + v_b) / 2)

                print(velocity)

        cap.release()
        cv2.destroyAllWindows()

    else:
        print("Unable to open camera")