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

    # variance of the first column
    var = np.var(imgray[sep:h - sep:10, 0])

    # check frames if the object is in the first column
    if var > 3*var_r:
        print('get frames...')

        place = 1
        imsave = []
        iter = 0
        ret = False

        # save the next 4 frames
        for i in range(4):
            _, imgs = cap.read()
            imsave.append(imgs)

        # for i in range(len(imsave)):
        #     cv2.imwrite('im{:}.png'.format(i), imsave[i])

        # search through the next frames for the object
        while iter < len(imsave):
            if place < 0 or place > len(imsave)-1 or iter > len(imsave):
                print('Object not in frame')
                return False, None

            imgray = cv2.cvtColor(imsave[place], cv2.COLOR_BGR2GRAY)

            # compute variances of different columns
            var1 = np.var(imgray[sep:h-sep:4, 0])
            var2 = np.var(imgray[sep:h-sep:4, int(w / 3)])
            var3 = np.var(imgray[sep:h-sep:4, int(w / 3 * 2)])
            var4 = np.var(imgray[sep:h-sep:4, w - 1])

            if var1 < 3*var_r and var4 < 3*var_r and var2 > 3*var_r and var3 > 3*var_r:
                ret = True
                break

            # go to next frame
            if var1 > 60:
                place += 1

            # go to previous frame
            if var4 > 60:
                place -= 1

            iter += 1

        return ret, imgray

    else:
        return False, None


if __name__ == "__main__":
    # camera parameters (mm)
    focal_length = 3.04
    pixel_size = 1.12/1000

    # estimated velocity of the object (m/s)
    v_est = 0

    # load calibration parameters
    mtx = np.loadtxt('mtx_normal.txt')
    dst = np.loadtxt('dist_normal.txt')

    # get calibration map
    map_x, map_y = cv2.initUndistortRectifyMap(mtx, dst, None, mtx, (w, h), cv2.CV_32FC1)

    # separation from edge
    sep = 250
    vec = 400

    # ringbuffer for geometry pattern
    buf = collections.deque(maxlen=10)

    # to flip the image, modify the flip_method parameter (0 and 2 are the most common)
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_NORMAL)

        t_ref = time.time()

        # wait for exposure control
        while time.time() - t_ref < 8 and cv2.getWindowProperty("CSI Camera", 0) >= 0:
            ret_val, img = cap.read()
            im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresh_value = np.max(im)
            print("Init mode, max value = {:.0f}".format(thresh_value), end="\r", flush=True)

        print("\nEntering measurement mode...")

        # kernel for edge detection
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

            # get the image
            ret_val, img = cap.read()
            imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # get an image from the trigger
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
                cnts_upper = remove_contours(cnts_upper, 2600, 3400)
                cnts_lower = remove_contours(cnts_lower, 2600, 3400)

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
                cx = mtx[0][2]
                cy = mtx[1][2]

                # generate array with objectpoints
                objp = np.array([[-110 * ppm + cx, -75 * ppm + cy],
                                 [-55 * ppm + cx, -75 * ppm + cy],
                                 [cx, -75 * ppm + cy],
                                 [55 * ppm + cx, -75 * ppm + cy],
                                 [110 * ppm + cx, -75 * ppm + cy],
                                 [-110 * ppm + cx, 75 * ppm + cy],
                                 [-55 * ppm + cx, 75 * ppm + cy],
                                 [cx, 75 * ppm + cy],
                                 [55 * ppm + cx, 75  * ppm + cy],
                                 [110 * ppm + cx, 75 * ppm + cy]], dtype=np.float32)

                # get transformation matrix
                T, _ = cv2.findHomography(imgp, objp, method=0)

                # find object-contours inside the separation
                cnts_m, _ = cv2.findContours(edge[sep:(h - sep), :], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                # remove invalid contours
                cnts_m = remove_contours(cnts_m, 10000, 900000)

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

                # rotate the rectangle
                tl = D @ (tl-c)
                tr = D @ (tr-c)
                br = D @ (br-c)
                bl = D @ (bl-c)

                # compute camera-plane distance (in mm)
                height = focal_length*(1/(ppm*pixel_size)-1)

                # compute the triangle around object
                a_l = np.sqrt(height ** 2 + (tl[1]/ppm) ** 2)
                b_l = np.sqrt(height ** 2 + (bl[1]/ppm) ** 2)
                c_l = (bl[1] - tl[1]) / ppm
                a_r = np.sqrt(height ** 2 + (tr[1]/ppm) ** 2)
                b_r = np.sqrt(height ** 2 + (br[1]/ppm) ** 2)
                c_r = (br[1] - tr[1]) / ppm

                # compute the radius (incircles)
                s_l = (a_l + b_l + c_l) / 2
                r_l = np.sqrt(((s_l - a_l) * (s_l - b_l) * (s_l - c_l)) / s_l)
                s_r = (a_r + b_r + c_r) / 2
                r_r = np.sqrt(((s_r - a_r) * (s_r - b_r) * (s_r - c_r)) / s_r)

                # compute Diameter
                D = r_r + r_l

                # correction factors (theorem of intersecting lines)
                x_cor = 1 - (height - D) / height

                # readjust x values
                tl[0] = tl[0] + (cx - tl[0]) * x_cor
                tr[0] = tr[0] + (cx - tr[0]) * x_cor
                br[0] = br[0] + (cx - br[0]) * x_cor
                bl[0] = bl[0] + (cx - bl[0]) * x_cor

                # compute length
                L = (tr[0] - tl[0] + br[0] - bl[0]) / 2 / ppm

                # compute amount of rows
                n_rows = (bl[1] - tl[1] + br[1] - tr[1]) / 2

                # get the time delay between the rows
                t_delay = 1 / 28 / h * n_rows

                # correct the rolling shutter
                L -= t_delay * v_est * 1000 / ppm

                # print the result
                print('l = {:.2f}, w = {:.2f}'.format(L, D))

                if fps_process == 0:
                    fps_process = 1 / (time.time() - t_process)
                    time_old = time.time()
                else:
                    fps_process = 0.9 * fps_process + 0.1 / (time.time() - t_process)
                    time_old = time.time()

                print('fps = {:.1f}'.format(fps_process))

                # show the measured contour (remove for full fps)
                cv2.namedWindow("Contours", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Contours", 1632, 924)
                cv2.putText(imgray, "L = {:.3f}, D = {:.3f}".format(L, D), (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)
                cv2.drawContours(imgray, [np.array([tl+c, tr+c, br+c, bl+c]).astype("int")], -1, (255, 255, 0), 10)
                cv2.imshow("Contours", imgray)

                # Show the edges for visual control (remove for full fps)
                # cv2.resizeWindow("CSI Camera", 1632, 924)
                # cv2.putText(edge, "{:.2f} fps".format(fps_process), (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)
                # cv2.imshow("CSI Camera", edge)

                keyCode = cv2.waitKey(30) & 0xFF
                # stop the program on the ESC key
                if keyCode == 27:
                    break

        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")
