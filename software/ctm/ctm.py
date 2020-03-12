import cv2
import numpy as np
import time

# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Defaults to 1280x720 @ 60fps
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of the window on the screen

def gstreamer_pipeline(
    capture_width=3280,
    capture_height=2464,
    display_width=820,
    display_height=616,
    framerate=20,
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

mtx = np.loadtxt('mtx.txt')
dst = np.loadtxt('dist.txt')
w = 3280
h = 2464

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dst,(w,h),1,(w,h))

mean_fps = 0
n = 1

# To flip the image, modify the flip_method parameter (0 and 2 are the most common)
print(gstreamer_pipeline(flip_method=0))
cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
if cap.isOpened():
    window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
    # Window
    while cv2.getWindowProperty("CSI Camera", 0) >= 0:
        t_ref = time.time()

        ret_val, img = cap.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # edge detection
        edge = cv2.Canny(gray, 50, 150)

        # corner
        # corner = cv2.cornerHarris(gray, 2, 3, 0.04)
        # img[corner > 0.01 * corner.max()] = [0, 0, 255]

        # contour detection
        #cnts, h = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)

        minLineLength = 100
        maxLineGap = 10

        lines = cv2.HoughLinesP(edge, 1, np.pi / 180, 100, minLineLength,maxLineGap)

        for l in lines:
            rho = l[0][0]
            theta = l[0][1]

            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imshow("img", img)

        # print fps
        mean_fps += int(np.round(1 / (time.time() - t_ref)))

        print(mean_fps/n)

        n += 1

        # This also acts as
        keyCode = cv2.waitKey(30) & 0xFF
        # Stop the program on the ESC key
        if keyCode == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
else:
    print("Unable to open camera")

