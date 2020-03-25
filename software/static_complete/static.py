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

def getContours(cnts):
    c = []
    for i in range(len(cnts)):
        for j in range(len(cnts[i])):
            c.append([cnts[i][j][0][0], cnts[i][j][0][1]])

    return np.array(c)

mtx = np.loadtxt('intrinsics/mtx.txt')
dst = np.loadtxt('intrinsics/dst.txt')
w = 3280
h = 2464

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dst,(w,h),1,(w,h))

# define variables for fps
mean_fps = 0
n = 1

# separation from edge
sep = 750

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

        # edge detection
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        edge = cv2.Canny(gray, 50, 80)
        edge = cv2.dilate(edge, None, iterations=2)

        # find rectangles
        cnts_upper, _ = cv2.findContours(edge[0:sep,:], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts_lower, _ = cv2.findContours(edge[h-sep:h,:], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        imgp = np.zeros((4, 2))

        index = 0
        for c in cnts_upper:
            area = cv2.contourArea(c)
            if area > 3000 or area < 2000:
                continue

            box = cv2.minAreaRect(c)
            box = cv2.boxPoints(box)

            # unpack box and compute center
            (tl, tr, br, bl) = box
            p = rect_center(tl, tr, br, bl)

            imgp[index] = p
            cv2.circle(img, (int(p[0]), int(p[1])), 7, (0, 255, 255), -1)

            index += 1

            if index == 2:
                break

        for c in cnts_lower:
            area = cv2.contourArea(c)
            if area > 3000 or area < 2000:
                continue

            box = cv2.minAreaRect(c)
            box = cv2.boxPoints(box)

            # unpack box and compute center
            (tl, tr, br, bl) = box
            tl[1] += (h-sep)
            tr[1] += (h - sep)
            br[1] += (h - sep)
            bl[1] += (h - sep)
            p = rect_center(tl, tr, br, bl)

            imgp[index] = p
            cv2.circle(img, (int(p[0]), int(p[1])), 7, (0, 255, 255), -1)

            index += 1

            if index == 4:
                break

        # order image points
        imgp = order_points(imgp)

        # undistort imgp
        imgp = cv2.undistortPoints(imgp, mtx, dst)

        # pixels/metric
        ppm = dist.euclidean(imgp[0], imgp[1])/200
        fx = mtx[0][0]
        fy = mtx[1][1]
        mx = mtx[0][2]
        my = mtx[1][2]
        world_x = -mx/fx
        world_y = -my/fy

        # world coorinates: (px*ppm-mx)/fx, (py*ppm-my)/fy
        # float32 for projectPoints (float64 ok) and getperspectiveTransform (float32 required)
        world_p = np.array([[world_x,             world_y,             0],
                            [(200*ppm - mx) / fx, world_y,             0],
                            [(200*ppm - mx) / fx, (140*ppm - my) / fy, 0],
                            [world_x,             (140*ppm - my) / fy, 0]], dtype=np.float32)

        world_p, _ = cv2.projectPoints(world_p, (0, 0, 0), (0, 0, 0), mtx, dst)

        # top left point (imgp[0]) is reference
        objp = np.concatenate([world_p[0]+imgp[0], world_p[1]+imgp[0], world_p[2]+imgp[0], world_p[3]+imgp[0]])

        # get transformation matrix
        T = cv2.getPerspectiveTransform(imgp, objp)

        # find object
        edge = cv2.erode(edge, None, iterations=2)
        cnts_m, _ = cv2.findContours(edge[sep:(h-sep),:], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        cnts_m = getContours(cnts_m)

        # undistort and warp perspective
        cnts_trans = cv2.undistortPoints(cnts_m.astype(np.float32), mtx, dst) #float32 important
        cnts_trans = cv2.perspectiveTransform(cnts_trans, T, (w, h))

        box = cv2.minAreaRect(cnts_trans)
        box = cv2.boxPoints(box)
        box[0][1] += sep
        box[1][1] += sep
        box[2][1] += sep
        box[3][1] += sep

        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # compute the Euclidean distance between the midpoints
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        # compute the size of the object
        dimA = dA / ppm
        dimB = dB / ppm

        # print fps
        mean_fps += 1 / (time.time() - t_ref)
        print(mean_fps / n)
        n += 1

        # draw to image
        box = cv2.minAreaRect(cnts_m)

        box = cv2.minAreaRect(cnts_m)
        box = cv2.boxPoints(box)
        box[0][1] += sep
        box[1][1] += sep
        box[2][1] += sep
        box[3][1] += sep

        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        cv2.drawContours(img, [box.astype('int')], -1, (0, 255, 0), 2)

        # draw connecting lines
        cv2.line(img, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
        cv2.line(img, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

        # draw midpoints
        cv2.circle(img, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(img, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(img, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(img, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

        # draw the object sizes on the image
        cv2.putText(img, "{:.3f}".format(dimA), (int(tltrX - 15), int(tltrY - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        cv2.putText(img, "{:.3f}".format(dimB),
                    (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

        # draw section into img
        img = cv2.line(img, (0, sep), (3280, sep), (255, 0, 0), 4)
        img = cv2.line(img, (0, h - sep), (3280, h - sep), (255, 0, 0), 4)

        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('img', 2*820, 2*616)
        cv2.imshow("img", img)

        # get key code
        keyCode = cv2.waitKey(30) & 0xFF

        nr = 0
        # Creat a snap if s is pressed
        if keyCode == 115:
            cv2.imwrite('p{:}.jpg'.format(nr), edge)
            nr += 1
            print('snapshot created')

        # Stop the program on the ESC key
        if keyCode == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
else:
    print("Unable to open camera")