"""
This script opens a window, which helps the user to set up the demostrator, make sure to set w, h, vec, sep to the same
values as later used in main.py.
"""
import cv2
import numpy as np
from morph import *
from geometry import *

# Global variable
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
    vec = 400

    # kernel for edge detection
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline(flip_method=0))
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_NORMAL)
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            ret_val, img = cap.read()
            imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            edge = get_edge_erroded(imgray, kernel)

            # find geometry pattern (upper and lower)
            cnts_upper, _ = cv2.findContours(edge[0:sep, vec:(w - vec)], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnts_lower, _ = cv2.findContours(edge[(h - sep):h, vec:(w - vec)], cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_NONE)
            # only keep valid contours
            cnts_upper = remove_contours(cnts_upper, 2700, 3400)
            cnts_lower = remove_contours(cnts_lower, 2700, 3400)

            # pattern area
            area = []

            # draw the contours, compute pattern area
            for c in cnts_upper:
                box = cv2.minAreaRect(c)

                a = cv2.contourArea(c)
                area.append(a)

                box = cv2.boxPoints(box)
                box = box + [vec, 0]
                cv2.drawContours(img, [box.astype("int")], -1, (0, 255, 255), 5)

            for c in cnts_lower:
                box = cv2.minAreaRect(c)

                a = cv2.contourArea(c)
                area.append(a)

                box = cv2.boxPoints(box)
                box = box + [vec, h - sep]
                cv2.drawContours(img, [box.astype("int")], -1, (0, 255, 255), 5)

            # print out the mean pattern area
            print('Mean pattern area: {:.0f}'.format(np.mean(area)))

            # draw lines to postion pattern
            cv2.line(img, (vec, 0), (vec, sep), (0, 0, 255), thickness=5, lineType=8, shift=0)
            cv2.line(img, (vec, h-sep), (vec, h), (0, 0, 255), thickness=5, lineType=8, shift=0)
            cv2.line(img, (w-vec, 0), (w-vec, sep), (0, 0, 255), thickness=5, lineType=8, shift=0)
            cv2.line(img, (w - vec, h-sep), (w - vec, h), (0, 0, 255), thickness=5, lineType=8, shift=0)
            cv2.line(img, (vec, sep), (w-vec, sep), (0, 0, 255), thickness=5, lineType=8, shift=0)
            cv2.line(img, (vec, h-sep), (w-vec, h-sep), (0, 0, 255), thickness=5, lineType=8, shift=0)

            cv2.resizeWindow("CSI Camera", 1632, 924)
            cv2.imshow("CSI Camera", img)

            keyCode = cv2.waitKey(30) & 0xFF
            if keyCode == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        print("Unable to open camera")
