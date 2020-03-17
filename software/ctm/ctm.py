import cv2
import numpy as np
import time
from scipy.spatial import distance as dist

# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Defaults to 1280x720 @ 60fps
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of the window on the screen

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

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def rect_center(tl, tr, br, bl):
    A = np.array([[tl[0]-br[0], tr[0]-bl[0]],
                  [tl[1]-br[1], tr[1]-bl[1]]])
    b = np.array([[tr[0]-tl[0]],
                  [tr[1]-tl[1]]])
    s = np.linalg.inv(A)@b

    p = (tl-br)*s[0] + tl

    return (p[0], p[1])


def order_points(pts):
	xSorted = pts[np.argsort(pts[:, 0]), :]

	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]

	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost

	D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
	(br, tr) = rightMost[np.argsort(D)[::-1], :]

	return np.array([tl, tr, br, bl], dtype="float32")

mtx = np.loadtxt('chess3280x2464/mtx.txt')
dst = np.loadtxt('chess3280x2464/dist.txt')
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

        # undistort and crop image
        img = cv2.undistort(img, mtx, dst, None, newcameramtx)
        x,y,w,h = roi
        img = img[y:y+h, x:x+w]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # edge detection
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        edge = cv2.Canny(gray, 30, 50)

        edge = cv2.dilate(edge, None, iterations=2)

        # contour detection
        cnts, h = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        imgp = np.zeros((4, 2))

        index = 0
        for c in cnts:
            area = cv2.contourArea(c)

            if area > 2600 or area < 2200:
                continue

            box = cv2.minAreaRect(c)
            box = cv2.boxPoints(box)

            # unpack box and compute center
            (tl, tr, br, bl) = box
            p = rect_center(tl, tr, br, bl)

            imgp[index] = p


            cv2.circle(img, (int(p[0]), int(p[1])), 7, (0, 255, 255), -1)

            index += 1

            if index == 4:
                break

        # order image points
        imgp = order_points(imgp)

        #pixels/metric
        ppm = dist.euclidean(imgp[0], imgp[1])/200
        fx = mtx[0][0]
        fy = mtx[1][1]
        mx = mtx[0][2]
        my = mtx[1][2]
        world_x = -mx/fx
        world_y = -my/fy

        # Welt-Koordianten in der form: (px*ppm-mx)/fx, (py*ppm-my)/fy
        # float32 für projectPoints (float64 würde auch gehen) und getperspectiveTransform (float32 notwendig)
        world_p = np.array([[world_x,             world_y,             0],
                            [(200*ppm - mx) / fx, world_y,             0],
                            [(200*ppm - mx) / fx, (140*ppm - my) / fy, 0],
                            [world_x,             (140*ppm - my) / fy, 0]], dtype=np.float32)

        world_p, _ = cv2.projectPoints(world_p, (0, 0, 0), (0, 0, 0), mtx, 0)

        # Punkt oben links (imgp[0]) ist Referenz
        objp = np.concatenate([world_p[0]+imgp[0], world_p[1]+imgp[0], world_p[2]+imgp[0], world_p[3]+imgp[0]])

        # Bildebene Transformieren
        M = cv2.getPerspectiveTransform(imgp, objp)
        h, w = img.shape[:2]
        img = cv2.warpPerspective(img, M, (w, h))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # edge detection
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        edge = cv2.Canny(gray, 30, 50)

        edge = cv2.dilate(edge, None, iterations=1)

        # contour detection
        cnts, h = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # loop over contours and match rectangles over objects
        for c in cnts:
            # if the contour is not sufficiently large, ignore it
            if cv2.contourArea(c) < 20000:
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

            # draw the object sizes on the image
            cv2.putText(img, "{:.3f}".format(dA), (int(tltrX - 15), int(tltrY - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            cv2.putText(img, "{:.3f}".format(dB),
                        (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('img', 2*820, 2*616)
        cv2.imshow("img", img)

        # print fps
        mean_fps += int(np.round(1 / (time.time() - t_ref)))

        print(mean_fps/n)

        n += 1

        # This also acts as
        keyCode = cv2.waitKey(30) & 0xFF

        nr = 0
        # Creat a snap if s is pressed
        if keyCode == 115:
            cv2.imwrite('p{:}.jpg'.format(nr), img)

            nr += 1

            print('snapshot created')

        # Stop the program on the ESC key
        if keyCode == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
else:
    print("Unable to open camera")

