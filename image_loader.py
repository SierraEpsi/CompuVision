
import numpy as np
import cv
import cv2
import math

def filterImage(img):


    ddepth = cv2.CV_8U;
    cv2.namedWindow("img1", cv2.WINDOW_NORMAL)
    cv2.namedWindow("img2", cv2.WINDOW_NORMAL)
    cv2.namedWindow("img3", cv2.WINDOW_NORMAL)
    cv2.namedWindow("img4", cv2.WINDOW_NORMAL)
    cv2.namedWindow("img5", cv2.WINDOW_NORMAL)
    #cv2.namedWindow("img6", cv2.WINDOW_NORMAL)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('img1', img)

    img_Eq = cv2.equalizeHist(img)
    cv2.imshow('img2', img_Eq)

    img_Blurred = cv2.medianBlur(img_Eq, 11)
    cv2.imshow('img3', img_Blurred)

    # 1. Do canny (determine the right parameters) on the gray scale image
    t1 = 50
    t2 = 0
    img_Edges = cv2.Canny(img_Blurred, t1, t2, apertureSize=3)
    cv2.imshow('img4', img_Edges)

    # Show the results of canny
    canny_result = np.copy(img)
    canny_result[img_Edges.astype(np.bool)] = 0
    cv2.imshow('img5', canny_result)
    cv2.destroyAllWindows()

    return img_Edges

# this part of the code is only executed if the file is run stand-alone
if __name__ == '__main__':

    # read an image
    img = cv2.imread('_Data/Radiographs/01.tif')

    edges =  filterImage(img)

    # img_Blurred0 = cv2.GaussianBlur(img,(5,5),sigmaX=0,sigmaY=0,borderType=cv2.BORDER_DEFAULT)
    # cv2.imshow('img3', img_Blurred0)

    # img_Sobelx = cv2.Sobel(img_Eq,ddepth,1,0,None,3,borderType=cv2.BORDER_DEFAULT)
    # img_Sobelxy = cv2.Sobel(img_Sobelx,ddepth,0,1,None,3,borderType=cv2.BORDER_DEFAULT)
    # cv2.imshow('img4', img_Sobelxy)

    # img_Eq2 = cv2. equalizeHist(img_Sobelxy)
    # cv2.imshow('img5', img_Eq2)

    # ret,img_thresh = cv2.threshold(img_Eq2,63,73,type=cv2.THRESH_BINARY)
    # cv2.imshow('img6', img_thresh)

