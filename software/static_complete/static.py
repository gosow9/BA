import cv2
import numpy as np
import time
from scipy.spatial import distance as dist

# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Defaults to 1280x720 @ 60fps
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of the window on the screen

def gstreamer_pipeline(
    capture_width=3264,
    capture_height=1848,
    display_width=3264,
    display_height=1848,
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
    '''computes midpoint of to points'''
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def rect_center(tl, tr, br, bl):
    '''computes centerpoint of a rectangle (order matters)'''
    A = np.array([[tl[0]-br[0], tr[0]-bl[0]],
                  [tl[1]-br[1], tr[1]-bl[1]]])
    b = np.array([[tr[0]-tl[0]],
                  [tr[1]-tl[1]]])
    s = np.linalg.inv(A)@b

    p = (tl-br)*s[0] + tl

    return (p[0], p[1])


def order_points(pts):
    '''orders four pounts clockwise'''
    xSorted = pts[np.argsort(pts[:, 0]), :]

    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    return np.array([tl, tr, br, bl], dtype="float32")

def getContours(cnts):
    '''combines a set of contours (points) into one single contour'''
    c = []
    for i in range(len(cnts)):
        for j in range(len(cnts[i])):
            c.append([[cnts[i][j][0][0], cnts[i][j][0][1]]])

    return np.array(c)

# load camera intrinsics
mtx = np.loadtxt('mtx_normal.txt')
dst = np.loadtxt('dist_normal.txt')

# define resolution
w = 3264
h = 1848

map_x, map_y = cv2.initUndistortRectifyMap(mtx, dst, None, mtx, (w, h), cv2.CV_32FC1)

# define variables for fps
fps = 0

# separation from edge
sep = 700

# open camera
cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0, capture_width=w, capture_height=h,
                                          display_width=w, display_height=h), cv2.CAP_GSTREAMER)
if cap.isOpened():
    window_handle = cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Camera', 1632, 924)
    # Window
    while cv2.getWindowProperty('Camera', 0) >= 0:
        t_ref = time.time()
        ret_val, img = cap.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # edge detection
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        edge = cv2.Canny(gray, 50, 80)
        edge = cv2.dilate(edge, None, iterations=1)

        # undistort edge
        edge = cv2.remap(edge, map_x, map_y, cv2.INTER_LINEAR)

        # find rectangles
        cnts_upper, _ = cv2.findContours(edge[0:sep,:], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts_lower, _ = cv2.findContours(edge[h-sep:h,:], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # remove invalid contours
        r = []
        for i in reversed(range(len(cnts_upper))):
            area = cv2.contourArea(cnts_upper[i])
            if area > 3000 or area < 2000:
                r.append(i)

        for i in r:
            cnts_upper.pop(i)

        r = []
        for i in reversed(range(len(cnts_lower))):
            area = cv2.contourArea(cnts_lower[i])
            if area > 3000 or area < 2000:
                r.append(i)

        for i in r:
            cnts_lower.pop(i)

        # reject frame if amount of points is invalid
        if len(cnts_upper) != 5 or len(cnts_lower) != 5:
            print('No calibration points detected')
            continue

        # undistort cnts_upper
        # for c in cnts_upper:
        #     x_tmp = c[0][0][0]
        #     y_tmp = c[0][0][1]
        #
        #     c[0][0][1] = map_x[y_tmp][x_tmp]
        #     c[0][0][1] = map_y[y_tmp][x_tmp]
        #
        # # undistort cnts_lower
        # for c in cnts_upper:
        #     x_tmp = c[0][0][0]
        #     y_tmp = c[0][0][1]
        #
        #     c[0][0][1] = map_x[y_tmp][x_tmp]
        #     c[0][0][1] = map_y[y_tmp][x_tmp]

        # collect calibration points
        imgp_upper = np.zeros((5,2))
        index = 0
        for c in cnts_upper:
            box = cv2.minAreaRect(c)
            box = cv2.boxPoints(box)

            (tl, tr, br, bl) = box
            p = rect_center(tl, tr, br, bl)

            imgp_upper[index] = p

            index += 1

        imgp_lower = np.zeros((5,2))
        index = 0
        for c in cnts_lower:
            box = cv2.minAreaRect(c)
            box = cv2.boxPoints(box)

            (tl, tr, br, bl) = box
            tl[1] += (h-sep)
            tr[1] += (h - sep)
            br[1] += (h - sep)
            bl[1] += (h - sep)
            p = rect_center(tl, tr, br, bl)

            imgp_lower[index] = p

            index += 1

        # Sort the points
        imgp_upper.sort(axis=0)
        imgp_lower.sort(axis=0)

        # combine points into one array
        imgp = np.concatenate([imgp_lower, imgp_upper])

        # pixels/metric
        ppm_array = [dist.euclidean(imgp[0], imgp[4])/240, dist.euclidean(imgp[5], imgp[9])/240, dist.euclidean(imgp[1], imgp[3])/160, dist.euclidean(imgp[6], imgp[8])/160]

        for i in range(5):
            ppm_array.append(dist.euclidean(imgp[i], imgp[5+i])/160)

        ppm = np.mean(ppm_array)

        mx = mtx[0][2]
        my = mtx[1][2]

        # world coorinates: (px*ppm-mx)/fx, (py*ppm-my)/fy
        # float32 for projectPoints (float64 ok) and getperspectiveTransform (float32 required)
        objp = np.array([[-120*ppm+mx, -80*ppm+my],
                         [ -60*ppm+mx, -80*ppm+my],
                         [         mx, -80*ppm+my],
                         [  60*ppm+mx, -80*ppm+my],
                         [ 120*ppm+mx, -80*ppm+my],
                         [-120*ppm+mx,  80*ppm+my],
                         [ -60*ppm+mx,  80*ppm+my],
                         [         mx,  80*ppm+my],
                         [  60*ppm+mx,  80*ppm+my],
                         [ 120*ppm+mx,  80*ppm+my]], dtype=np.float32)

        # get transformation matrix
        T, _ = cv2.findHomography(imgp, objp, method=0)
        edge = cv2.warpPerspective(edge, T, (w, h))

        # find object
        edge = cv2.erode(edge, None, iterations=1)
        cnts_m, _ = cv2.findContours(edge[sep:(h-sep),:], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # reject frame if cnts_m is empty
        if not cnts_m:
            print('No Object found')
            continue

        cnts_m = getContours(cnts_m)

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

        # compute the Euclidean distance between the midpoints
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        # compute the size of the object
        dimA = dA / ppm
        dimB = dB / ppm

        print('Frame ok')



        # compute fps
        if fps == 0:
            fps = 1 / (time.time() - t_ref)
        else:
            fps = 0.9 * fps + 0.1*1 / (time.time() - t_ref)

        # undistort and warp the complete image
        img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
        img = cv2.warpPerspective(img, T, (w, h))

        #cnts_m = cv2.perspectiveTransform(cnts_m.astype(np.float32), T, (w, h))
        for p in objp:
            cv2.circle(img, (int(p[0]), int(p[1])), 7, (0, 255, 255), -1)

        for p in imgp:
            cv2.circle(img, (int(p[0]), int(p[1])), 7, (0, 0, 255), -1)

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
        cv2.putText(img, '{:.3f}'.format(dimA), (int(tltrX - 15), int(tltrY - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        cv2.putText(img, '{:.3f}'.format(dimB),
                    (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

        # show mean_fps on screen
        cv2.putText(img, '{:.3f} fps'.format(fps), (100,150) ,cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)

        # draw section into img
        img = cv2.line(img, (0, sep), (w, sep), (255, 0, 0), 4)
        img = cv2.line(img, (0, h - sep), (w, h - sep), (255, 0, 0), 4)

        cv2.imshow('Camera', img)

        # get key code
        keyCode = cv2.waitKey(30) & 0xFF

        nr = 0
        # Creat a snap if s is pressed
        if keyCode == 115:
            cv2.imwrite('p{:}.png'.format(nr), edge)
            nr += 1
            print('snapshot created')

        # Stop the program on the ESC key
        if keyCode == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
else:
    print("Unable to open camera")