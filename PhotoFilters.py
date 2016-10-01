import cv2
import numpy as np

def channel(imageName):
    """Takes image name of image whose channel is to be removed"""
    image = cv2.imread(imageName)
    cv2.imshow("IMAGE", image)
    r,g,b = cv2.split(image)
    r[:] = 0
    # g[:] =0
    merged = cv2.merge((r,g,b))
    cv2.imshow("OUTPUT", merged)
    cv2.waitKey(0)
    # return merged

def dilate(imageName,iter):
    """Takes image name of image and dilates it"""
    image = cv2.imread(imageName)
    dilated = cv2.dilate(image,None, iterations=iter)
    cv2.imshow("OUTPUT", dilated)
    cv2.imshow("INPUT", image)
    cv2.waitKey(0)
    # return merged

def erode(imageName,iter):
    """Takes image name of image and erodes it"""
    image = cv2.imread(imageName)
    dilated = cv2.erode(image,None, iterations=iter)
    cv2.imshow("OUTPUT", dilated)
    cv2.imshow("INPUT", image)
    cv2.waitKey(0)
    # return merged

def opening(imageName,iter):
    """Takes image name of image and erodes it"""
    image = cv2.imread(imageName)
    dilated = cv2.erode(image,None, iterations=iter)
    dilated = cv2.dilate(image,None, iterations=iter)
    cv2.imshow("OUTPUT", dilated)
    cv2.imshow("INPUT", image)
    cv2.waitKey(0)
    # return merged

def closing(imageName,iter):
    """Takes image name of image and erodes it"""
    image = cv2.imread(imageName)
    dilated = cv2.dilate(image,None, iterations=iter)
    dilated = cv2.erode(image,None, iterations=iter)
    cv2.imshow("OUTPUT", dilated)
    cv2.imshow("INPUT", image)
    cv2.waitKey(0)
    # return merged


def addTexture(imageName):
    """Takes image name of image in which texture is to be added"""

    # Texture Parameters
    imageW = 0.7
    textureW = 1 - imageW

    image = cv2.imread(imageName)
    cv2.imshow("IMAGE", image)
    textureName = "G:/Filters/Texture1.jpg"
    texture = cv2.imread(textureName)
    row, col = image.shape[:2]
    texture = texture[0:row, 0:col]
    texturized = cv2.addWeighted(image, imageW, texture, textureW, 1)

    cv2.imshow("OUTPUT", texturized)
    cv2.waitKey(0)
    # return merged

def sharpen(imageName):
    """Takes image name of image where edges is to be highlighted as input"""

    # Texture Parameters
    imageW = 0.9
    edgeW = 1 - imageW

    image = cv2.imread(imageName)
    cv2.imshow("IMAGE", image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 40, 100)
    edge = cv2.merge((edge, edge, edge))
    mask = cv2.bitwise_and(edge, image)

    sharpened = cv2.addWeighted(image, imageW, mask, edgeW, 1)


    cv2.imshow("OUTPUT", sharpened)
    cv2.waitKey(0)
    # return sharpened

def smooth(imageName):
    """Takes image name of image where edges is to be highlighted as input"""

    # Texture Parameters
    image = cv2.imread(imageName)
    cv2.imshow("IMAGE", image)
    smooth = cv2.GaussianBlur(image, (11,11), 1)
    cv2.imshow("OUTPUT", smooth)
    cv2.waitKey(0)
    # return merged

def changeColorSet(imageName):
    """Takes image name as input and changes the color"""

    image = cv2.imread(imageName)
    cv2.imshow("INPUT", image)
    for i in xrange(0,12):
        res = cv2.applyColorMap(image, i)
        cv2.imshow("OUTPUT", res)
        cv2.waitKey(0)

def threshold(imageName):
    image = cv2.imread(imageName)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # print thresh
    cv2.imshow("IMAGE", image)
    cv2.imshow("OUTPUT", thresh)
    cv2.waitKey(0)

def stack(imageName):
    image = cv2.imread(imageName)
    stack = np.hstack((image, image))
    stack = np.vstack((stack, stack))
    cv2.imshow("IMAGE", stack)
    cv2.waitKey(0)

def coolBlack():
    IMAGE_WEIGHT = 0.5

    image = cv2.imread("G:/Filters/wasim.jpg",0)
    black = cv2.imread("G:/Filters/black5.jpg",0)
    black = cv2.resize(black, image.shape[::-1])

    res1 = cv2.addWeighted(image, IMAGE_WEIGHT, black, 1 - IMAGE_WEIGHT, 1)


    #NORMALIZE IMAGES
    image = np.float32(image)
    black = np.float32(black)

    image /= 255
    black /= 200

    res = image*black

    cv2.imshow("RES", res)
    cv2.waitKey(0)

    fname = "G:/Filtes/temp.jpg"
    cv2.imwrite(fname, res)
    res = cv2.imread(fname, 0)

    cv2.imshow("BLACK", res)
    cv2.waitKey(0)


# imageName = "G:\Filters/hazard.jpg"
# imageName = "G:\Filters/wasim.jpg"
imageName = "G:\Filters/aliaDQ.jpg"
# help(channel)
# channel(imageName, )
# changeColorSet(imageName)
# threshold(imageName)
# stack(imageName)
# coolBlack()
closing(imageName,2)
# opening(imageName,2)
erode(imageName,2)
# dilate(imageName,2)