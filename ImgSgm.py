import ImgPP as iPP
import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('_Data/Radiographs/01.tif')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.bilateralFilter(img,9,500,500)
#img = iPP.PPimg(img)

nbPts = 100
height, width = img.shape
window = width/15
cP = 0
nP = window
jaw_points = []
while nP < width:
    mean_line = np.mean(img[:,cP:nP],1)
    max_points = [[],[]]
    r = 50
    # find local maxima
    for i in range(r,len(mean_line)-r):
        point = mean_line[i]
        maPoint = np.max(mean_line[i-r:i+r])
        miPoint = np.min(mean_line[i-r:i+r])
        if point == maPoint or point == miPoint:
            max_points[0].append(i)
            max_points[1].append(point)

    # make em better
    for i in range(1,len(max_points[0])-2):
        point = max_points[1][i]
        nPoint = max_points[1][i+1]
        if point - nPoint > 70:
            nnPoint = max_points[1][i+2]
            if np.abs(nnPoint - point) < 40:
                jaw_points.append([cP + window/2,max_points[0][i+1]])
                i += 2
    cP = nP
    nP += window

pts = np.asarray(jaw_points).reshape((-1,1,2))
cv2.polylines(img,[pts],False,(0,255,255))
cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.resizeWindow('img', 1500, 1000)
cv2.imshow('img', img)
cv2.waitKey(0)