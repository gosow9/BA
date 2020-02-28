import cv2
import numpy as np
import imutils
from imutils import contours
from scipy.spatial import distance as dist

import time

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
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

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
if cap.isOpened():
    window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
    # Window
    while cv2.getWindowProperty("CSI Camera", 0) >= 0:
        t_ref = time.time()

        ret_val, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        #edge detection
        edge = cv2.Canny(gray, 50, 100)
        edge = cv2.dilate(edge, None, iterations=1)
        edge = cv2.erode(edge, None, iterations=1)

        #contour detection
        cnts = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        #sort contours from left to right (reference-object left)
        (cnts, _) = contours.sort_contours(cnts)

        #init pixels/metric-variable
        ppm = None

        #loop over contours and match rectangles over objects
        for c in cnts:
            # if the contour is not sufficiently large, ignore it
            if cv2.contourArea(c) < 500:
                continue
            # compute the rotated bounding box of the contour, then
            # draw the contours
            box = cv2.minAreaRect(c)
            box = cv2.boxPoints(box)
            cv2.drawContours(img, [box.astype('int')], -1, (0, 255, 0), 2)

            # loop over the original points and draw them
            for (x, y) in box:
                cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)

            #unpack box and compute midpoints
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)

            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            # draw connecting lines
            cv2.line(img, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
            cv2.line(img, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

            # draw midpoints
            cv2.circle(img, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
            cv2.circle(img, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
            cv2.circle(img, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
            cv2.circle(img, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

            # compute the Euclidean distance between the midpoints
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            if ppm is None:
                ppm = dB/8

            # compute the size of the object
            dimA = dA / ppm
            dimB = dB / ppm

            # draw the object sizes on the image
            cv2.putText(img, "{:.3f}".format(dimA), (int(tltrX - 15), int(tltrY - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(img, "{:.3f}".format(dimB),
                        (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        cv2.imshow("box", img)

        #print fps
        print(int(np.round(1/(time.time()-t_ref))))

        # This also acts as
        keyCode = cv2.waitKey(30) & 0xFF
        # Stop the program on the ESC key
        if keyCode == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
else:
    print("Unable to open camera")