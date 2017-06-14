import cv2.cv as cv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ASM import ASM as ASM

tooth = []
tmpTooth = []
dragging = False
start_point = (0, 0)


def init(pts, img):
    global tooth
    BG_img = np.array(img)

    # transform model points to image coord
    tooth = pts
    if len(pts.shape) == 1:
        pts = np.zeros((pts.size/2,2))
        pts[:,0] = tooth[:pts.size/2]
        pts[:,1] = tooth[pts.size/2:]
        pts = pts.astype('int32')

    m0, m1 = np.min(pts,0)
    pts[:, 0] = np.add(pts[:,0],-m0)
    pts[:, 1] = np.add(pts[:,1],-m1)
    tooth = pts

    pimg = np.reshape(pts,(-1, 1, 2))
    cv2.polylines(img, [pimg], True, (0, 256, 0),thickness=5)

    # show gui
    cv2.namedWindow('choose', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('choose', 1200, 800)
    cv2.imshow('choose', img)
    cv.SetMouseCallback('choose', mouse_func, BG_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    centroid = np.mean(tooth, axis=0)
    return centroid


def mouse_func(ev, x, y, flags, img):
    global tooth
    global dragging
    global start_point

    if ev == cv.CV_EVENT_LBUTTONDOWN:
        dragging = True
        start_point = (x, y)
    elif ev == cv.CV_EVENT_LBUTTONUP:
        tooth = tmpTooth
        dragging = False
    elif ev == cv.CV_EVENT_MOUSEMOVE:
        if dragging and tooth != []:
            mouse_move(x, y, img)


def mouse_move(x, y, img):
    global tmpTooth
    h = img.shape[0]
    tmp = np.array(img)
    dx = x-start_point[0]
    dy = y-start_point[1]

    pts = [(p[0]+dx, p[1]+dy) for p in tooth]
    tmpTooth = pts

    pimg = np.reshape(pts,(-1, 1, 2)).astype('int32')
    cv2.polylines(tmp, [pimg], True, (0, 256, 0),thickness=5)
    cv2.imshow('choose', tmp)

if __name__ == '__main__':
    img = cv2.imread('_Data/Radiographs/01.tif')
    folder = '_Data/landmarks/original/'
    nbImgs = 14
    nbDims = 40
    tooth = 1
    active_shape_model = ASM(folder, nbImgs, nbDims, tooth)
    pts = active_shape_model.mu
    print init(pts,img)