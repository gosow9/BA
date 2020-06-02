"""
Main module, used for measuring
"""

import cv2
import numpy as np
import time
from scipy.spatial import distance as dist
from morph import *
from geometry import *
import collections

# Global variables
w = 3264
h = 1848

# separation from edge
sep = 250
vec = 400

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
    :return: The gstreamer pipeline
    :return type: String
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


def trigger(imgray, var_r):
    """
    Checks for an object in the frames, and returns the image in which the object doesn't touch the edges.

    :param imgray: Images to check
    :type imgray: InputArray
    :param var_r: Reference variance
    :type var_r: Float
    :return: frame with object
    :return type: OutputArray
    """

    var = np.var(imgray[sep:h - sep:10, 0])

    if var > 3*var_r:
        place = 1
        imsave = []
        iter = 0
        ret = False

        for i in range(4):
            _, imgs = cap.read()
            imsave.append(imgs)

        for i in range(len(imsave)):
            cv2.imwrite('im{:}.png'.format(i), imsave[i])

        while iter < len(imsave):

            if place < 0 or place > len(imsave)-1:
                return False, None

            imgray = cv2.cvtColor(imsave[place], cv2.COLOR_BGR2GRAY)

            var1 = np.var(imgray[sep:h-sep:4, 0])
            var2 = np.var(imgray[sep:h-sep:4, int(w / 3)])
            var3 = np.var(imgray[sep:h-sep:4, int(w / 3 * 2)])
            var4 = np.var(imgray[sep:h-sep:4, w - 1])

            if var1 < 3*var_r and var4 < 3*var_r and var2 > 3*var_r and var3 > 3*var_r:
                ret = True
                break

            if var1 > 60:
                place += 1
                iter += 1

            if var4 > 60:
                place -= 1
                iter += 1


        print('iter = {:}'.format(iter))
        print('place = {:}'.format(place))
        return ret, imgray

    else:
        return False, None


if __name__ == "__main__":
    # camera parameters (mm)
    focal_lenght = 3.04
    pixel_size = 1.12/1000

    # estimated diameter of the object (mm)
    D_est = 50

    # estimated velocity of the object (m/s)
    v_est = 0

    # load calibration parameters
    mtx = np.loadtxt('mtx_normal.txt')
    dst = np.loadtxt('dist_normal.txt')

    # get calibration map
    map_x, map_y = cv2.initUndistortRectifyMap(mtx, dst, None, mtx, (w, h), cv2.CV_32FC1)

    # ringbuffer for geometry pattern
    buf = collections.deque(maxlen=10)

    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_NORMAL)
        # Window
        t_ref = time.time()

        # wait for exposure control
        while time.time() - t_ref < 8 and cv2.getWindowProperty("CSI Camera", 0) >= 0:
            ret_val, img = cap.read()
            im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresh_value = np.max(im)
            print("Init mode, max value = {:.0f}".format(thresh_value), end="\r", flush=True)

        print("\nEntering measurment mode")

        # kernel for edge-detection
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # get image
        ret_val, img = cap.read()
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # compute a reference variance
        var = []
        var.append(np.var(imgray[sep:h - sep:4, 0]))
        var.append(np.var(imgray[sep:h - sep:4, int(w / 3)]))
        var.append(np.var(imgray[sep:h - sep:4, int(w / 3 * 2)]))
        var.append(np.var(imgray[sep:h - sep:4, w - 1]))

        fps_trigger = 0
        fps_process = 0

        time_old = time.time()
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            time_old = time.time() - time_old
            t_ref = time.time()
            ret_val, img = cap.read()
            imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # check trigger
            #ret, imgray = trigger(imgray, np.mean(var))

            t_process = time.time()
            if True:
                # edge detection
                edge = get_edge_erroded(imgray, kernel)

                # find geometry pattern (upper and lower)
                cnts_upper, _ = cv2.findContours(edge[0:sep, vec:(w - vec)], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cnts_lower, _ = cv2.findContours(edge[(h - sep):h, vec:(w - vec)], cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_NONE)

                # only keep valid contours
                cnts_upper = remove_contours(cnts_upper, 2700, 3200)
                cnts_lower = remove_contours(cnts_lower, 2700, 3200)

                if len(cnts_upper) == 5 and len(cnts_lower) == 5:
                    # undistort contours
                    cnts_upper = remap_contours(cnts_upper, map_x, map_y)
                    cnts_lower = remap_contours(cnts_lower, map_x, map_y)

                    # collect image points (geometry pattern)
                    imgp_upper = np.zeros((5, 2))
                    index = 0

                    for c in cnts_upper:
                        box = cv2.minAreaRect(c)
                        box = cv2.boxPoints(box)

                        (tl, tr, br, bl) = box
                        tl[0] += vec
                        tr[0] += vec
                        br[0] += vec
                        bl[0] += vec

                        p = rect_center(tl, tr, br, bl)
                        imgp_upper[index] = p
                        index += 1

                    imgp_lower = np.zeros((5, 2))
                    index = 0

                    for c in cnts_lower:
                        box = cv2.minAreaRect(c)
                        box = cv2.boxPoints(box)

                        (tl, tr, br, bl) = box
                        tl[0] += vec
                        tr[0] += vec
                        br[0] += vec
                        bl[0] += vec
                        tl[1] += (h - sep)
                        tr[1] += (h - sep)
                        br[1] += (h - sep)
                        bl[1] += (h - sep)
                        p = rect_center(tl, tr, br, bl)
                        imgp_lower[index] = p
                        index += 1

                    # sort the points
                    imgp_upper = sorted(imgp_upper, key=lambda p: p[0])
                    imgp_lower = sorted(imgp_lower, key=lambda p: p[0])

                    # append the buffer with a combined array
                    buf.append(np.concatenate([imgp_upper, imgp_lower]))

                else:
                    print('No pattern detected')

                # reject frame if buffer is empty
                if len(buf) == 0:
                    print('No image points in buffer')
                    continue

                # use the mean of the buffer as image points
                imgp = np.mean(buf, axis=0)

                # get pixels per metric unit
                ppm_array = [dist.euclidean(imgp[0], imgp[4]) / 220, dist.euclidean(imgp[5], imgp[9]) / 220,
                             dist.euclidean(imgp[1], imgp[3]) / 110, dist.euclidean(imgp[6], imgp[8]) / 110]

                for i in range(5):
                    ppm_array.append(dist.euclidean(imgp[i], imgp[5 + i]) / 150)

                ppm = np.mean(ppm_array)

                # get the principal point
                mx = mtx[0][2]
                my = mtx[1][2]

                # generate array with objectpoints
                objp = np.array([[-110 * ppm + mx, -75 * ppm + my],
                                 [-55 * ppm + mx, -75 * ppm + my],
                                 [mx, -75 * ppm + my],
                                 [55 * ppm + mx, -75 * ppm + my],
                                 [110 * ppm + mx, -75 * ppm + my],
                                 [-110 * ppm + mx, 75 * ppm + my],
                                 [-55 * ppm + mx, 75 * ppm + my],
                                 [mx, 75 * ppm + my],
                                 [55 * ppm + mx, 75  * ppm + my],
                                 [110 * ppm + mx, 75 * ppm + my]], dtype=np.float32)

                # get transformation matrix
                T, _ = cv2.findHomography(imgp, objp, method=0)

                # find object-contours inside the separation
                cnts_m, _ = cv2.findContours(edge[sep:(h - sep), :], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                # remove invalid contours
                cnts_m = remove_contours(cnts_m, 10000, 2000000)

                # undistort contours
                cnts_m = remap_contours(cnts_m, map_x, map_y)

                # reject frame if cnts_m is empty
                if not cnts_m:
                    print('No object found')
                    continue

                # combine contours to one
                cnts_m = np.concatenate(cnts_m, axis=0).astype(np.float64)

                # warp perspective of the contours
                cnts_m = cv2.perspectiveTransform(cnts_m, T)

                # minimum area rectangle around cnts_m
                box = cv2.minAreaRect(cnts_m.astype(np.float32))
                box = cv2.boxPoints(box)
                box[0][1] += sep
                box[1][1] += sep
                box[2][1] += sep
                box[3][1] += sep

                # sort the points clockwise
                (tl, bl, tr, br) = sort_points(box)

                # compute the center of the rectangle
                c = rect_center(tl, tr, br, bl)

                # compute the rotation matrix
                phi = np.arctan((tr[1]-tl[1]) / (tl[0] - tr[0]))
                D = np.array([[np.cos(phi), -np.sin(phi)],
                              [np.sin(phi),  np.cos(phi)]])

                # rotate the points
                tl = D @ (tl-c)
                tr = D @ (tr-c)
                br = D @ (br-c)
                bl = D @ (bl-c)

                # compute camera-plane distance (in mm)
                height = focal_lenght*(1/(ppm*pixel_size)-1)

                # correction factors (theorem of intersecting lines)
                x_cor = 1 - (height - D_est) / height
                y_cor = 1 - (height - D_est /2 ) / height

                # readjust x values
                tl[0] = tl[0] + (mx - tl[0]) * x_cor
                tr[0] = tr[0] + (mx - tr[0]) * x_cor
                br[0] = br[0] + (mx - br[0]) * x_cor
                bl[0] = bl[0] + (mx - bl[0]) * x_cor

                # readjust y values
                tl[1] = tl[1] + (my - tl[1]) * y_cor
                tr[1] = tr[1] + (my - tr[1]) * y_cor
                br[1] = br[1] + (my - br[1]) * y_cor
                bl[1] = bl[1] + (my - bl[1]) * y_cor

                # add the center back
                tl += c
                tr += c
                br += c
                bl += c

                # get the midpoints of the rectangle
                (tltrX, tltrY) = midpoint(tl, tr)
                (blbrX, blbrY) = midpoint(bl, br)
                (tlblX, tlblY) = midpoint(tl, bl)
                (trbrX, trbrY) = midpoint(tr, br)

                # compute the euclidean distance between the midpoints
                dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

                # compute the size of the object
                dimA = dA / ppm
                dimB = dB / ppm

                # compute amount of rows
                n_rows = (bl[1] - tl[1] + br[1] - tr[1]) / 2

                # get the time delay between the rows
                t_delay = 1 / 28 / h * n_rows

                # the larger is the lenght
                if dimA > dimB:
                    # correct the rolling shutter
                    dimA -= t_delay * v_est * 1000 / ppm
                    print('l = {:.2f}, w = {:.2f}'.format(dimA, dimB))

                else:
                    # correct the rolling shutter
                    dimB -= t_delay * v_est * 1000 / ppm
                    print('l = {:.2f}, w = {:.2f}'.format(dimB, dimA))

                if fps_process == 0:
                    fps_process = 1 / (time.time() - t_process)
                    time_old = time.time()
                else:
                    fps_process = 0.9 * fps_process + 0.1 / (time.time() - t_process)
                    time_old = time.time()

                print('fps = {:.1f}'.format(fps_process))

                cv2.namedWindow("Contours", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Contours", 1632, 924)
                cv2.drawContours(imgray, [np.array([tl, tr, br, bl]).astype("int")], -1, (255, 255, 0), 10)
                cv2.imshow("Contours", imgray)

                # Show the edges for visual control
                # cv2.resizeWindow("CSI Camera", 1632, 924)
                # cv2.putText(edge, "{:.2f} fps".format(fps), (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)
                # cv2.imshow("CSI Camera", edge)

                # This also acts as
                keyCode = cv2.waitKey(30) & 0xFF
                # Stop the program on the ESC key
                if keyCode == 27:
                    break
            else:
                fps_trigger = 0.9 * fps_trigger + 0.1 / (time.time() - t_ref + time_old)
                time_old = time.time()
                print('{:.1f}'.format(fps_trigger), end="\r", flush=True)

        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")
