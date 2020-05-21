# -*- coding: utf-8 -*-
"""
main module
"""
import cv2
import numpy as np
import time
import morph


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
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
        # Window
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            t = time.time()
            ret_val, img = cap.read()
            imggray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            (thresh, blackAndWhiteImage) = cv2.threshold(imggray, 127, 255, cv2.THRESH_BINARY)
            t = t-time.time()
            print(t)
            #cv2.imshow("CSI Camera", img)
            # This also acts as
            keyCode = cv2.waitKey(30) & 0xFF
            # Stop the program on the ESC key
            if keyCode == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")
