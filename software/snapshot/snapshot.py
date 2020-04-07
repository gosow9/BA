import cv2
import numpy as np

def gstreamer_pipeline(
    capture_width=3280,
    capture_height=2464,
    display_width=3280,
    display_height=2464,
    framerate=20,
    flip_method=2,
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

# To flip the image, modify the flip_method parameter (0 and 2 are the most common)
print(gstreamer_pipeline(flip_method=0))
cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
if cap.isOpened():
    window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
    # Window

    nr = 0

    while cv2.getWindowProperty("CSI Camera", 0) >= 0:

        ret_val, img = cap.read()
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cv2.namedWindow('snapshot', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('snapshot', 2*820, 2*616)
        cv2.imshow('snapshot', img)

        keyCode = cv2.waitKey(30) & 0xFF

        # Creata snap if s is pressed
        if keyCode == 115:
            cv2.imwrite('im{:}.png'.format(nr), img)

            print('snapshot created')

            nr = nr + 1

        # Stop the program on the ESC key
        if keyCode == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
else:
    print("Unable to open camera")