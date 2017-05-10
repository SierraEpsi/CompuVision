
import cv2
import math

# this part of the code is only executed if the file is run stand-alone
if __name__ == '__main__':
    # read an image
    img = cv2.imread('_Data/Radiographs/01.tif')

    # show the image, and wait for a key to be pressed
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.imshow('img', img)
    cv2.waitKey(0)

    # remove spickle noise, quantum noise?
    img = cv2.medianBlur(img, 7)

    cv2.imshow('img', img)
    cv2.waitKey(0)

    img = cv2.bilateralFilter(img, 9, 200, 150)

    cv2.namedWindow("img2", cv2.WINDOW_NORMAL)
    cv2.imshow('img2', img)
    cv2.waitKey(0)

    # look for edges
    img = cv2.Canny(img, 20, 10)

    # show the smoothed image, and wait for a key to be pressed
    cv2.imshow('img', img)
    cv2.waitKey(0)