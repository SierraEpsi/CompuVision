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
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(25, 25))
    img = clahe.apply(img)
    img = cv2.medianBlur(img, 11)
    img = cv2.threshold(img, 30, 256, cv2.THRESH_BINARY)[1]
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img', 1500, 1000)

    img = cv2.erode(img, (5, 5), iterations=1)

    t1 = 100
    t2 = 0
    img = cv2.Canny(img, t1, t2, apertureSize=3)


    return img

def getMouseCoord(event, x, y, flags, params):
    global mouseX, mouseY, refPt
    if event == cv2.EVENT_LBUTTONDBLCLK:
        mouseX, mouseY = x, y
    elif event == cv2.EVENT_LBUTTONDOWN:
        refPt.append((x, y))

def selectSqr(img):
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img', 1500, 1000)
    cv2.imshow('img',img)
    while True:
        k = cv2.waitKey(0) & 0xFF
        # if the window is closed or 'esc' is pressed stop
        if k == 255 or k == 27:
            break
        elif k == ord('a'):
            print mouseX, mouseY

if __name__ == '__main__':
    # read an image
    img = cv2.imread('_Data/Radiographs/01.tif')
    img = PPimg(img)

    # Select square
    selectSqr(img)