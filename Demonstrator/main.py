# -*- coding: utf-8 -*-
"""
main module
"""
import cv2
import numpy as np
import time
from morph import *


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
    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline(flip_method=0))
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_NORMAL)
        # Window
        t = time.time()
        background = np.ones((2464, 3280), np.int8)
        im = np.ones((2464, 3280), np.int8)
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
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            t = time.time()
            ret_val, img = cap.read()
            #Diffrent types to get the edge use one:
            #out = get_edge_errosion_dilation(img, grad)
            out = get_edge_erroded(img, kernel)
            #out = get_edge_dilated(img, kernel)
            #out = get_edge_grad(img, grad)
            t = t-time.time()
            print(t)
            # Show the edges for visual control
            cv2.resizeWindow("CSI Camera", 820, 616)
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
