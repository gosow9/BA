import cv2
import numpy as np

def gstreamer_pipeline(
    capture_width=3280,
    capture_height=2464,
    display_width=3280,
    display_height=2464,
    framerate=1,
    flip_method=0,
):
    return (
        "nvarguscamerasrc wbmode=0 ! "
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
# image resolution
w = 3280
h = 2464

# open camera
cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0, capture_width=w, capture_height=h,
                                          display_width=w, display_height=h), cv2.CAP_GSTREAMER)
if cap.isOpened():
    window_handle = cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Camera', 820, 616)

    # Window
    nr = 0
    while cv2.getWindowProperty('Camera', 0) >= 0:
        ret_val, img = cap.read()


        cv2.imshow('Camera', img)

        # get key code
        keyCode = cv2.waitKey(30) & 0xFF

        # Process image if s is pressed
        if keyCode == 115:
            print('processing...')
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCornersSB(gray, (8, 7), flags=cv2.CALIB_CB_ACCURACY)

            cv2.putText(img, 'average brightness: {:.1f}'.format(np.mean(gray)), (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 0, 255), 4)
            if ret == True:
                retval_h, sharpness_h = cv2.estimateChessboardSharpness(gray, (8, 7), corners)
                retval_v, sharpness_v = cv2.estimateChessboardSharpness(gray, (8, 7), corners, vertical=True)

                cv2.putText(img, 'average brightness: {:.1f}'.format(np.mean(gray)), (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.5 ,(255, 0, 255), 4)
                cv2.putText(img, 'sharpness horizontal: {:.1f}'.format(retval_h[0]), (100, 230), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 0, 255), 4)
                cv2.putText(img, 'sharpness vertical: {:.1f}'.format(retval_v[0]), (100, 310), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 0, 255), 4)
                cv2.putText(img, 'brightness black: {:.1f}'.format((retval_h[1]+retval_v[1])/2), (100, 390), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 0, 255), 4)
                cv2.putText(img, 'brightness white: {:.1f}'.format((retval_h[2]+retval_v[2])/2), (100, 470), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 0, 255), 4)

                cv2.imwrite('im{:}.png'.format(nr), img)
                print('image written')

                nr += 1

            else:
                print('no chessboard corners detected')

            cv2.namedWindow('sharp', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('sharp', 820, 616)
            cv2.imshow('sharp', img)

        # Stop the program on the ESC key
        if keyCode == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

else:
    print("Unable to open camera")