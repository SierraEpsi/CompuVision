
import cv
import cv2
import math

# this part of the code is only executed if the file is run stand-alone
if __name__ == '__main__':
    ddepth = cv2.CV_16S;
    # read an image
    img = cv2.imread('_Data/Radiographs/02.tif')
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # show the image, and wait for a key to be pressed
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.imshow('img', img)
    cv2.waitKey(0)

    # remove spickle noise, quantum noise?
    img = cv2.medianBlur(img, 7)

    img = cv2.bilateralFilter(img, 9, 200, 150)

    img = cv2.adaptiveThreshold(img, 256, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 0)

    cv2.namedWindow("img2", cv2.WINDOW_NORMAL)
    cv2.imshow('img2', img)
    cv2.waitKey(0)