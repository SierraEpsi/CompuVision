import numpy as np
import matplotlib.pyplot as plt
import cv2

def PPimg(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img,5)
    img = cv2.bilateralFilter(img,9,500,500)

    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(16, 16))
    img = clahe.apply(img)

    imgX1 = cv2.Scharr(img,-1,1,0)
    imgX2 = cv2.Scharr(cv2.flip(img,1),-1,1,0)
    imgX2 = cv2.flip(imgX2,1)
    imgX = cv2.addWeighted(imgX1, 0.5, imgX2, 0.5, 0)

    imgY1 = cv2.Scharr(img,-1,0,1)
    imgY2 = cv2.Scharr(cv2.flip(img,0),-1,0,1)
    imgY2 = cv2.flip(imgY2,0)
    imgY = cv2.addWeighted(imgY1, 0.5, imgY2, 0.5, 0)

    img = cv2.addWeighted(imgX, 0.5, imgY, 0.5, 0)
    img = cv2.equalizeHist(img)
    # img = cv2.threshold(img, 30, 256, cv2.THRESH_BINARY)[1]
    t1 = 100
    t2 = 50
    img = cv2.Canny(img, t1, t2, apertureSize=3)

    return img

# Used to track mouse
def getMouseCoord(event, x, y, flags, params):
    global mouseX, mouseY, refPt
    if event == cv2.EVENT_LBUTTONDBLCLK:
        mouseX, mouseY = x, y
    elif event == cv2.EVENT_LBUTTONDOWN:
        if len(refPt) == 2:
            refPt[0] = refPt[1]
            refPt[1] = (x,y)
        else:
            refPt.append((x,y))

def selectSqr(img):

    global mouseX, mouseY, refPt
    mouseX = -1
    mouseY = -1
    refPt = []

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img', 1500, 1000)
    cv2.setMouseCallback('img', getMouseCoord)
    cv2.imshow('img', img)

    doClose = False
    while True:
        k = cv2.waitKey(0) & 0xFF
        # if the window is closed or 'esc' is pressed stop
        if k == 255 or k == 27:
            doClose = True
            break
        # check if spacebar is pressed
        else:
            cv2.imshow('img', img)
            # press space to continue
            if k == 32:
                if len(refPt) == 2 or (mouseX != -1 and mouseY != -1):
                    break
    if doClose:
        return

    if len(refPt) == 2:
        x1 = np.min((refPt[0][1], refPt[1][1]))
        x2 = np.max((refPt[0][1], refPt[1][1]))
        y1 = np.min((refPt[0][0], refPt[1][0]))
        y2 = np.max((refPt[0][0], refPt[1][0]))
        roi = img[x1:x2, y1:y2]
        if roi.shape[0] > 0 and roi.shape[1] > 0:
            cv2.imshow('roi', roi)
            k = cv2.waitKey(0) & 0xFF
            pnts = np.where(roi > 0)  # contains all possible landmarks at estimated location
            plt.plot(pnts[1], pnts[0], 'r.')
            plt.gca().invert_xaxis()
            plt.gca().invert_yaxis()
            plt.show()
        refPt = []

if __name__ == '__main__':
    # read an image
    img = cv2.imread('_Data/Radiographs/01.tif')
    img = PPimg(img)

    # Select square
    selectSqr(img)