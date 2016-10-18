"""
This module takes two images and stitches them to create panorama

Stitching Algorithm
First place the right shifted image on empty canvas
Then place the left image down.
"""

import cv2
import numpy as np


def drawMatches(image1, point1, image2, point2):
    "Connects two matching keypoints by a line"
    output = np.hstack((image1, image2))
    row, col = image1.shape[:2]
    x, y = point2
    x = x + col

    cv2.line(output, point1, (x, y), (255, 0, 255), 5)
    cv2.imshow("OUTPUT", output)
    cv2.waitKey(10)
    pass


# Goodness bias should matter more than recent bias
def findHomography(image1, image2, Match=0.6):
    """Finds the matrix which relates two images. H = [R|T]"""
    # FLANN parameters for matching features

    # Calcuate keypoints in an image
    sift = cv2.SIFT(2000)
    kp1, des1 = sift.detectAndCompute(image1, None)
    kp2, des2 = sift.detectAndCompute(image2, None)

    kp1.sort()

    # Define a matcher to match keypoints in image pairs
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    points1 = []
    points2 = []

    # ratio test as per Lowe's paper
    # 0.8 is default
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.6 * n.distance:
            mm = m.distance / n.distance

            # Keep a list of good points from both images to compute homography
            # Points1 = Matrix * Points2
            points1.append(kp1[m.queryIdx].pt)
            points2.append(kp2[m.trainIdx].pt)

            # Location of keypoints
            point1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1]))
            point2 = (int(kp2[m.trainIdx].pt[0]), int(kp2[m.trainIdx].pt[1]))

            # useful only for debugging
            drawMatches(image1, point1, image2, point2)

        else:
            # Bad match
            pass

    # Data type conversion
    points1 = np.float32(points1)
    points2 = np.float32(points2)

    # Are two images touching?
    isAdjacentImage = False

    # Two images are touching if they have common keypoints / regions
    if len(points1) and len(points2):
        H, mask = cv2.findHomography(points1, points2, cv2.FM_RANSAC, 10)
        isAdjacentImage = True
        return isAdjacentImage, H

    else:
        return isAdjacentImage, None


def stitch(imageL, imageR, result):
    row, col = imageR.shape[:2]
    # temp1 = result.copy()
    # temp2 = result.copy()
    print "RESULT : ", result.shape
    result[0:row, 0:col] = imageR
    row, col = imageL.shape[:2]
    result[0:row, 0:col] = imageL

    # result = cv2.addWeighted(temp2,0.5,temp1,0.5,0)
    # cv2.imshow("TEMP1", temp1)
    # cv2.imshow("TEMP2", temp2)
    cv2.imshow("RES", result)
    cv2.imwrite("G:/Result.jpg", result)
    cv2.waitKey(0)

    return result


def findOrderAndStitch(retval, image1, image2):
    translate_X, translate_Y = int(retval[0][2]), int(retval[1][2])

    # width = image1.shape[1]+image2.shape[1]-translate_X
    # height = image1.shape[0]+ image2.shape[0]-translate_Y
    row = np.max((image1.shape[0], image2.shape[0])) + abs(translate_Y)
    col = np.max((image1.shape[1], image2.shape[1])) + abs(translate_X)
    result = np.zeros((row, col, 3), np.uint8)

    print("Image1 shape : {}").format(image1.shape)
    print("Image2 shape : {}").format(image2.shape)
    print("Result shape : {}").format(result.shape)
    print("Translate Y, X {} {}").format(translate_Y, translate_X)

    # If left image is in left and right image at is in right
    if translate_X >= 0:

        row, col = image1.shape[:2]
        M = np.float32([[1, 0, translate_X], [0, 1, translate_Y]])
        r, c = row + abs(translate_Y), col + abs(translate_X)
        image1 = cv2.warpAffine(image1, M, (c, r))
        print "After affine image1 :", image1.shape

        imageR = image1
        imageL = image2

    else:
        row, col = image2.shape[:2]
        M = np.float32([[1, 0, -translate_X], [0, 1, -translate_Y]])
        r, c = row + abs(translate_Y), col + abs(translate_X)
        image2 = cv2.warpAffine(image2, M, (c, r))
        print "After affine image2 :", image2.shape

        imageR = image2
        imageL = image1

    result = stitch(imageL, imageR, result)

    cv2.imshow('INPUT 1', image1)
    cv2.imshow('INPUT 2', image2)
    cv2.imshow('RESULT', result)
    cv2.waitKey(0)


def stitchImage(image1, image2):
    """Stitches two images if they have common regions"""
    ret, H = findHomography(image1, image2)
    if ret:
        res = findOrderAndStitch(H, image1, image2)
    else:
        print("Only images with common area can be stitched!")


if __name__ == "__main__":
    image1 = cv2.imread("G:/Filters/Stitch/a1.jpg")
    image2 = cv2.imread("G:/Filters/Stitch/a3.jpg")
    stitchImage(image1, image2)
    help(drawMatches)
