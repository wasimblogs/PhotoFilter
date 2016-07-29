"""
Min value - Discards edges
Max value - sure edges
Between them, connectedness is checked to make sure edges are correct

Does Gaussian BLur to suppress noise
Finds intensity gradient of the image
Non-Maxima suppression - takes hints to suppress some edges
Hysteresis Thresholding - determines which edges are true and which not

For a trackbar to work a window is required before declaration of trackbar

Blurring produces the most dramatic effect
I am not sure between min and max which produces more effect
"""

import cv2
import numpy as np
import sys, getopt
import os

win = "Drawing"
canny = np.zeros((400,400),np.uint8)

def addTexture(image, sketch):
    """Takes image name of image in which texture is to be added"""

    # Texture Parameters
    imageW = 0.9
    sketchW = 1 - imageW

    res = cv2.addWeighted(image, imageW, sketch, sketchW, 1)
    return res

def cannyEditor(x):
    min = cv2.getTrackbarPos('Min', win)
    max = cv2.getTrackbarPos('Max', win)
    k = cv2.getTrackbarPos('K', win)

    print min, max, k
    if k%2 == 0:
        k += 1

    temp = cv2.blur(image,(k,k))
    global canny
    canny = cv2.Canny(temp, min, max)

    canny = cv2.bitwise_not(canny)
    # canny = cv2.GaussianBlur(canny, (3,3), 5)
    cv2.imshow(win,canny)

def draw(image):
    # Create Trackbar
    opts, args = getopt.getopt(sys.argv[1:], '', ['Min','Max', 'K'])
    opts = dict(opts)
    cv2.namedWindow(win)
    cv2.createTrackbar('Min',win, int(opts.get('--Min', 40)), 255, cannyEditor)
    cv2.createTrackbar('Max',win, int(opts.get('--Max', 40)), 255, cannyEditor)
    cv2.createTrackbar('K',win, int(opts.get('--K', 3)), 10, cannyEditor)

    kk =  cv2.waitKey(0)
    res =addTexture(image, canny)
    if kk == ord('s'):
        dir, ext = os.path.splitext(file)
        newName = dir+"D"+ext
        newName2 = dir + "DC"+ext
        newName3 = dir + "DCW"+ext
        cv2.imwrite(newName, canny)
        row, col = canny.shape
        stacked = np.zeros_like(canny)
        if row > col :
            stacked = np.hstack((image, canny))
            stacked = np.hstack((image, res))
        else :
            stacked = np.vstack((image, canny))
            stacked = np.vstack((image, res))

        cv2.imwrite(newName, canny)
        cv2.imwrite(newName2, stacked)
        test = cv2.imread(newName)
        if test.size>0: print 'Image Written'

    cv2.destroyAllWindows()

file = "G:/Filters/alia.jpg"
image = cv2.imread(file, 0)

draw(image)
