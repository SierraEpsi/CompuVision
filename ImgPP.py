import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from ACM import ACM as ACM
from ASM import ASM as ASM

def enhance(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.bilateralFilter(img, 9, 500, 500)
    return img;

def PPimg(img):
    img = enhance(img)

    imgX1 = cv2.Scharr(img, -1, 1, 0)
    imgX2 = cv2.Scharr(cv2.flip(img, 1), -1, 1, 0)
    imgX2 = cv2.flip(imgX2, 1)
    imgX = cv2.addWeighted(imgX1, 0.5, imgX2, 0.5, 0)

    imgY1 = cv2.Scharr(img,-1,0,1)
    imgY2 = cv2.Scharr(cv2.flip(img,0),-1,0,1)
    imgY2 = cv2.flip(imgY2,0)
    imgY = cv2.addWeighted(imgY1, 0.5, imgY2, 0.5, 0)

    img = cv2.addWeighted(imgX, 0.5, imgY, 0.5, 0)

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16))
    img = clahe.apply(img)
    img = cv2.medianBlur(img,7)

    #img = cv2.threshold(img, 30, 256, cv2.THRESH_BINARY)[1]
    t1 = 100
    t2 = 50
    img = cv2.Canny(img, t1, t2, apertureSize=3)

    return img


# Used to track mouse
def getMouseCoord(event, x, y, flags, params):
    global mouseX, mouseY, refPt, mouseAcrion
    if event == cv2.EVENT_LBUTTONDBLCLK:
        mouseX, mouseY = x, y
    elif event == cv2.EVENT_LBUTTONDOWN:
        if len(refPt) == 2:
            refPt[0] = refPt[1]
            refPt[1] = (x,y)
        else:
            refPt.append((x,y))
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img', 1500, 1000)
    cv2.setMouseCallback('img', getMouseCoord)
    cv2.imshow('img', img)

def selectSqr(img):
    global mouseX, mouseY, refPt
    mouseX = -1
    mouseY = -1
    refPt = []
    folder = '_Data/landmarks/original/'
    tooth = 1
    nbImgs = 14
    nbDims = 40
    asm = ASM(folder, nbImgs, nbDims, tooth)
    model = np.zeros((nbDims, 2))
    model[:, 0] = asm.mu[:nbDims]
    model[:, 1] = asm.mu[nbDims:]

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('img', getMouseCoord)
    cv2.resizeWindow('img', 1500, 1000)
    cv2.imshow('img',img)

    doClose = False
    while True:
        k = cv2.waitKey(20) & 0xFF
        # if the window is closed or 'esc' is pressed stop
        if k == 27:
            doClose = True
            break
        else:
            if mouseX != -1 and mouseY != -1:
                b_error = float('inf')
                b_point = -1
                for j in range(-50,50,5):
                    for i in range(-50,50,5):
                        pnt1, pnt2 = asm.get_search_box((mouseX+i,mouseY+j))
                        roi = img[pnt1[1]:pnt2[1],pnt1[0]:pnt2[0]]
                        pntsYX = np.where(roi > 0)  # contains all possible landmarks at estimated location
                        pntsYX = np.asarray(pntsYX).T
                        pnts = np.zeros(pntsYX.shape, int)
                        pnts[:, 0] = pntsYX[:, 1]
                        pnts[:, 1] = pntsYX[:, 0]
                        target, error = findClosestPoints(model, pnts, asm)
                        if error < b_error:
                            b_error = error
                            b_point = (i, j)
                pnt1, pnt2 = asm.get_search_box((mouseX + b_point[0], mouseY + b_point[1]))
                cv2.rectangle(img,pnt1,pnt2,(100,100,100),thickness=5)
                cv2.imshow('img',img)
                print 'found'
                cv2.waitKey(0)
    if doClose:
        return

def findClosestPoints(model, pnts, asm):
    if len(pnts) < len(model):
        return None, float('inf')
    pntsC = asm.center_landmarks(pnts)
    model = asm.center_landmarks(model)
    target = np.zeros(model.shape)

    error = 0
    for m in xrange(len(model)):
        dif = pntsC - model[m]
        px = np.power(dif[:, 0], 2)
        py = np.power(dif[:, 1], 2)
        d = np.sqrt(px + py)
        iP = np.argmin(d).astype(np.int32)
        target[m] = pnts[iP]
        error += d[iP]
    return target.astype(int), error


if __name__ == '__main__':
    # read an image
    img = cv2.imread('_Data/Radiographs/01.tif')
    img = PPimg(img)

    # Select square
    selectSqr(img)