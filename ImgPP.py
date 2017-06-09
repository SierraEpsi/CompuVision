import numpy as np
import cv2

def PPimg(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img,7)
    img = cv2.bilateralFilter(img,9,500,500)

    imgX1 = cv2.Scharr(img,-1,1,0)
    imgX2 = cv2.Scharr(cv2.flip(img,1),-1,1,0)
    imgX2 = cv2.flip(imgX2,1)
    imgX = (imgX1 + imgX2)/2

    imgY1 = cv2.Scharr(img,-1,0,1)
    imgY2 = cv2.Scharr(cv2.flip(img,0),-1,0,1)
    imgY2 = cv2.flip(imgY2,0)
    imgY = (imgY1 + imgY2)/2

    img = (imgX + imgY)/2
    img = cv2.threshold(img, 16, 256, cv2.THRESH_BINARY)[1]

    return img


if __name__ == '__main__':
    # read an image
    img = cv2.imread('_Data/Radiographs/04.tif')
    img = PPimg(img)

    # Show image
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img', 1500, 1000)
    cv2.imshow('img', img)
    cv2.waitKey(0)