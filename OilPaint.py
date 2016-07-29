"""
This module aims at converting an image into its oil-paint
ALGORITHM-LOCAL CLUSTERING WITH K =1
"""

import Image
import cv2
import numpy as np
import collections

def group(list):
    """
    :param list:
    :return: center value
    """

    d = list
    d.sort()
    c = collections.Counter(list)
    a = c.most_common(int(0.2*len(c)))
    a = sorted(a, key=lambda a:a[0])
    if len(a):
        center, freq = 0,0
        for i in xrange(0, len(a)):
            center += a[i][0] * a[i][1]
            freq += a[i][1]
        #
        center = np.ceil(center/freq)
        return center

    else:
        return np.average(list)


def windowExplore(image, winSize = 40):
    """
    def windowExplore(image, win)->None
    Divides an image into grids of size win*win
    """

    WAIT_TIME = 50
    WIN = winSize

    image = Image.border(image, WIN)
    image = cv2.GaussianBlur(image, (5,5), 5)
    oilPaint = np.zeros_like(image, np.uint8)
    avgPaint = oilPaint.copy()
    # cv2.imshow("OIL PAINT", oilPaint)
    row, col = image.shape[:2]

    startH = 0
    for i in xrange(0, row/WIN):
        startW = 0
        for j in xrange(0, col/WIN):
            roiQ = image[startH: startH + WIN, startW : startW + WIN]
            startK = 0
            # startK = startW

            colors = []
            colorsRed = []
            colorsBlue = []
            colorsGreen = []
            # for k in xrange(0, col/WIN):
            roi = image[startH : startH + WIN, startW : startW + WIN]

            # colors = [roi[i][j] for i in WIN for j in WIN]
            for i in xrange(0,WIN):
                for j in xrange(0,WIN):
                    colors.append(roi[i][j])

            center = group(colors)
            avg = np.average(roi)
            # print center, avg
            oilWindow = np.ones(roi.shape, np.uint8)*center
            avgWindow = np.ones(roi.shape, np.uint8)*avg

            oilPaint[startH : startH + WIN, startW : startW + WIN] = oilWindow
            avgPaint[startH : startH + WIN, startW : startW + WIN] = avgWindow

            # cv2.imshow("OIL", oilPaint)
            # cv2.imshow("AVERAGE", avgPaint)
            # cv2.imshow("INPUT", image)
            # cv2.waitKey(WAIT_TIME)

            startW = startW + WIN
        startH = startH + WIN

    # cv2.imshow("OIL", oilPaint)
    # cv2.imshow("AVERAGE", avgPaint)
    # cv2.imshow("INPUT", image)
    # cv2.imshow("BLURRED", cv2.GaussianBlur(oilPaint, (5,5), 5))
    # cv2.waitKey(WAIT_TIME*0)
    blurred = cv2.GaussianBlur(oilPaint, (5,5), 5)
    print oilPaint.shape
    return oilPaint, blurred

imageName = "G:/Filters/wasim.jpg"
image = cv2.imread(imageName)
# help(Image.windowExplore)
size = 5
r,br =windowExplore(image[:,:,0],size)
print r.shape
print "Euta Vayo"
g,bg =windowExplore(image[:,:,1],size)
print "Duitai Vo"
b,bb =windowExplore(image[:,:,2],size)

cv2.imshow("rr", r)
cv2.waitKey(0)
res = cv2.merge((r,g,b))
resB = cv2.merge((br,bg,bb))

cv2.imwrite("G:/Filters/wasimO.jpg", res)
cv2.imshow("RES", res)
cv2.imshow("B RES", resB)
cv2.imshow("INPUT", image)
cv2.waitKey(0)