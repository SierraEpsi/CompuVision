import cv2
import cv2.cv as cv
import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack
import scipy.signal
from scipy.ndimage import morphology

import ImgPP as iPP
from old.TthMdl import ToothModel as TML


class JPath:
    def __init__(self, point, index, w=25):
        self.w = w
        self.start = index
        self.pointsX = [w*index]
        self.pointsY = [point]
        self.intensity = 0

    def add_best(self, some_points, w, img):
        lX = self.pointsX[-1]
        lY = self.pointsY[-1]
        min_intensity = float('inf')
        for point in some_points:
            cv2.rectangle(img, (w,point), (w+5,point+5), (0, 255, 255))
            intensity = cv.InitLineIterator(cv.fromarray(img), (lX,lY), (w,point))
            intensity = sum(intensity)
            if intensity < min_intensity:
                min_intensity = intensity
                best_point = point
        self.pointsX.append(w)
        self.pointsY.append(best_point)
        self.intensity += min_intensity

    def return_points(self):
        pts = []
        fit = np.polyfit(self.pointsX, self.pointsY, 2)
        self.pointsY = np.add(np.multiply(np.power(self.pointsX,2),fit[0]) + np.multiply(self.pointsX,fit[1]), fit[2])
        for i in range(0,len(self.pointsX)):
            pts.append([self.pointsX[i],int(self.pointsY[i])])
        pts = np.asarray(pts).reshape((-1, 1, 2))
        return pts

    def return_path(self):
        fit = np.polyfit(self.pointsX, self.pointsY, 2)
        self.pointsY = np.add(np.multiply(np.power(self.pointsX,2),fit[0]) + np.multiply(self.pointsX,fit[1]), fit[2])
        return [self.start, self.w, self.pointsY]

def find_jawline(img, window = 10):
    img = iPP.jaw_enhance(img)
    h, w = img.shape

    # Finding path points
    pot_jaw_pts = []
    for x in range(window,w,window):
        offset = int(0.4*h)
        wdw_img = img[offset:int(0.75*h),x-window:x+window]
        img_vals = np.sum(wdw_img,1)

        # only keep the 30 highest freqs to remove noise
        fft = scipy.fftpack.rfft(img_vals)
        fft[30:] = 0
        smoothed = scipy.fftpack.irfft(fft)

        # find local maxima
        loc_maxs_i = scipy.signal.argrelmax(smoothed)[0]
        loc_maxs_i = loc_maxs_i.tolist()

        # sort the indexes
        loc_maxs = []
        for max_i in loc_maxs_i:
            loc_maxs.append(smoothed[max_i])
        if len(loc_maxs) > 0:
            _, loc_maxs_i = zip(*sorted(zip(loc_maxs, loc_maxs_i),reverse=True))

            # keep best 3 with a minimal distance
            best_n = [offset + loc_maxs_i[0]]
            count = 1
            for i in range(1,len(loc_maxs_i)):
                max_i = loc_maxs_i[i]
                if all(abs(max_i - other_i) > 150 for other_i in best_n):
                    best_n.append(offset + max_i)
                    count += 1
                    if count == 3:
                        break
            pot_jaw_pts.append(best_n)

    paths = []
    for best_i in pot_jaw_pts[int(0.35*len(pot_jaw_pts))]:
        paths.append(JPath(best_i, int(0.35*len(pot_jaw_pts)), window))

    for i in range(int(0.35*len(pot_jaw_pts))+1,int(0.65*len(pot_jaw_pts))):
        for path in paths:
            path.add_best(pot_jaw_pts[i],(i+1)*window,img)

    min_intensity = float('inf')
    for path in paths:
        if path.intensity < min_intensity:
            min_intensity = path.intensity
            best_path = path

    return best_path.return_path()

def find_POI(img, window, isUp):
    img_w = img[window[0][1]:window[1][1], window[0][0]:window[1][0]].copy()

    imgX1 = cv2.Scharr(img_w, -1, 1, 0)
    imgX2 = cv2.Scharr(cv2.flip(img_w, 1), -1, 1, 0)
    imgX2 = cv2.flip(imgX2, 1)
    img2 = cv2.addWeighted(imgX1, 0.5, imgX2, 0.5, 0)
    h, w = img2.shape

    w1 = int(h/25)
    w2 = int(w/50)
    points = []

    # search for some points
    if isUp:
        rU = h
        rD = int(0.6*h)
    else:
        rU = int(0.6*h)
        rD = w1

    for i1 in range (rD,rU,w1):
        vals = []
        for i2 in range(w2,w,w2):
            val = np.sum(img2[i1-int(w1/2):i1+int(w1/2),i2-int(w2/2):i2+int(w2/2)])
            vals.append(val)
        fft = scipy.fftpack.rfft(vals)
        fft[30:] = 0
        smoothed = scipy.fftpack.irfft(fft)

        loc_maxs_i = scipy.signal.argrelmax(smoothed)[0]

        for point in loc_maxs_i:
            x1 = (1+point)*w2
            y1 = i1
            points.append((x1, y1))

    # group
    wg1 = int(0.125*w)
    wg2 = int(0.375*w)
    wg3 = int(0.625*w)
    wg4 = int(0.875*w)
    g1 = []
    g2 = []
    g3 = []
    for point in points:
        if point[0] > wg1 and point[0] < wg2:
            g1.append(point[0])
        elif point[0] > wg2 and point[0] < wg3:
            g2.append(point[0])
        elif point[0] > wg3 and point[0] < wg4:
            g3.append(point[0])

    g1 = int(np.mean(g1))
    g2 = int(np.mean(g2))
    g3 = int(np.mean(g3))


    img3 = img_w.copy()
    cv2.line(img3,(g1,0),(g1,h),(255,0,0),thickness=2)
    cv2.line(img3,(g2,0),(g2,h),(255,0,0),thickness=2)
    cv2.line(img3,(g3,0),(g3,h),(255,0,0),thickness=2)

    cv2.imshow('img',img3)
    cv2.waitKey(0)

    POI = [int((0+g1)/2),int((g1+g2)/2),int((g2+g3)/2),int((g3+w)/2)]

    # find the top of the tooth
    imgY1 = cv2.Scharr(img_w,-1,0,1)
    imgY2 = cv2.Scharr(cv2.flip(img_w,0),-1,0,1)
    imgY2 = cv2.flip(imgY2,0)
    img2 = cv2.addWeighted(imgY1, 0.5, imgY2, 0.5, 0)
    POI2 = []

    for x in POI:
        vals = []
        for i1 in range(w1, h, w1):
            val = np.sum(img2[i1 - int(w1 / 2):i1 + int(w1 / 2), x - int(w2 / 2):x + int(w2 / 2)])
            vals.append(val)

        fft = scipy.fftpack.rfft(vals)
        fft[30:] = 0
        smoothed = scipy.fftpack.irfft(fft)
        i_max1 = np.argmax(smoothed)
        loc_maxs_i = scipy.signal.argrelmax(smoothed)[0]
        if isUp:
            i_max2 = sorted(loc_maxs_i)[0]
            i_max = min(i_max1,i_max2)
            y = int(0.45*(h + i_max*w1))
            POI2.append((x+window[0][0],y+window[0][1]))
        else:
            i_max2 = sorted(loc_maxs_i)[-1]
            i_max = max(i_max1,i_max2)
            y = int(0.55*i_max*w1)
            POI2.append((x+window[0][0],y+window[0][1]))

    return POI2