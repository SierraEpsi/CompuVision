import cv2.cv as cv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ASM import ASM as ASM
from ACM import ACM as ACM
from Landmarks import Landmarks as LMS
import ImgPP

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
    img = cv2.imread('_Data/Radiographs/05.tif')
    img2 = img.copy()
    G_img = ImgPP.enhance2(img)
    G_img = ImgPP.GRimg2(G_img)
    folder = '_Data/landmarks/original/'
    nbImgs = 14
    nbDims = 40
    tooth = 1
    asm = ASM(folder, nbImgs, nbDims, 1)
    pts = asm.mu
    landmarks = LMS(pts)
    landmarks = landmarks.scale_to_window(asm.mW)
    pts = landmarks.as_matrix().astype('int32')
    pimg = landmarks.translate(init(pts,img)).as_matrix().astype('int32')
    while True:
        acm = ACM(-0.01, -0.1, 25.0, G_img, pimg)
        diff = -151
        while diff < -150:
            diff = acm.greedy_step(5)
            print diff
            img3 = cv2.cvtColor(G_img, cv2.COLOR_GRAY2BGR)
            pimg = np.reshape(acm.pts, (-1, 1, 2))
            cv2.polylines(img3, [pimg], True, (0, 0, 256), thickness=5)
            cv2.namedWindow('choose', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('choose', 1200, 800)
            cv2.imshow('choose', img3)
            cv2.waitKey(0)
        Tx, Ty, sf, angle, b, error = asm.estimate_trans(acm.pts)
        cX = LMS(acm.pts).get_centroid()
        pts = LMS(asm.reconstruct(b))
        pimg = pts.T([Tx,Ty],sf,angle).translate(cX).as_matrix().astype('int32')

        img3 = cv2.cvtColor(G_img, cv2.COLOR_GRAY2BGR)
        pts = np.reshape(pimg, (-1, 1, 2))
        cv2.polylines(img3, [pts], True, (0, 0, 256), thickness=5)
        cv2.namedWindow('choose', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('choose', 1200, 800)
        cv2.imshow('choose', img3)
        cv2.waitKey(0)