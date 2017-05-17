
import numpy as np
import cv
import cv2
import math

# this part of the code is only executed if the file is run stand-alone
if __name__ == '__main__':
    ddepth = cv2.CV_64F;
    cv2.namedWindow("img1", cv2.WINDOW_NORMAL)
    cv2.namedWindow("img2", cv2.WINDOW_NORMAL)
    cv2.namedWindow("img3", cv2.WINDOW_NORMAL)
    cv2.namedWindow("img4", cv2.WINDOW_NORMAL)


    # read an image
    img = cv2.imread('_Data/Radiographs/02.tif')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('img1', img)

    img_Blurred = cv2.medianBlur(img, 11)
    #cv2.GaussianBlur(img,(5,5),0,img,0,cv2.BORDER_DEFAULT)
    cv2.imshow('img2', img_Blurred)

    ret,img_thresh = cv2.threshold(img_Blurred,100,235,cv2.THRESH_BINARY)
    cv2.imshow('img3', img_thresh)

    #img = cv2. equalizeHist(img)
    #cv2.imshow('img3', img)

    #img2 = cv2.Laplacian(img_Blurred,ddepth=-1,ksize=7,scale=-1,delta=10)
    img_Sobel = cv2.Sobel(img_thresh,ddepth,1,1,3)
    cv2.imshow('img4', img_Sobel)

    # Convert to 8 bit, doesnt work..
    img_8U = np.array(img_thresh, dtype=np.uint8)
    cv2.imshow('img5', img_8U)

    # 1. Do canny (determine the right parameters) on the gray scale image
    t1 = 100
    t2 = 25
    edges = cv2.Canny(img_8U, t1, t2)

    # Show the results of canny
    canny_result = np.copy(img)
    canny_result[edges.astype(np.bool)] = 0
    cv2.imshow('img1', canny_result)
    cv2.waitKey(0)



    #img = cv2.medianBlur(img, 11)
    #img = cv2.bilateralFilter(img, 15, 200, 150)
    #cv2.imshow('img1', img)

    #img = cv2. equalizeHist(img)
    #cv2.imshow('img2', img)
    #cv2.waitKey(0)

    #img = cv2.adaptiveThreshold(img, 256, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 0)
    #cv2.imshow('img1', img)
    #cv2.waitKey(0)

    #  img = cv2.Laplacian(img,cv2.CV_64F,3)
    #  cv2.imshow('img1', img)
    #  cv2.waitKey(0)

    # # show the image, and wait for a key to be pressed
    # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    #
    # # remove spickle noise, quantum noise?
    # img = cv2.medianBlur(img, 7)
    #
    # img = cv2.bilateralFilter(img, 9, 200, 150)
    #
    # img = cv2.adaptiveThreshold(img, 256, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 0)
    #
    # cv2.namedWindow("img2", cv2.WINDOW_NORMAL)
    # cv2.imshow('img2', img)
    # cv2.waitKey(0)
    #
    # img = cv2.HoughCircles(img, cv.CV_HOUGH_GRADIENT, 10, 10, minRadius = 0, maxRadius = 100)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)