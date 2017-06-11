import ImgPP as iPP
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import scipy.signal
from scipy.ndimage import morphology
import cv2
import cv2.cv as cv

def Idunno():
    img = cv2.imread('_Data/Radiographs/01.tif')
    img = iPP.enhance(img)
    h,w = img.shape
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16))
    img = clahe.apply(img)

    edges = cv2.Canny(img, 50, 0, apertureSize=3)
    kernel = np.ones((5,5),np.uint8)
    edges = cv2.dilate(edges,kernel,iterations = 1)
    edges = np.divide(edges,255)
    img2 = img*edges

    hist = cv2.calcHist([img2],[0],None,[256],[0,256])
    hist = hist[1:]
    mHist = np.argmax(hist)/2
    img2 = cv2.threshold(img, mHist, 256, cv2.THRESH_BINARY)[1]

    window = 10
    wP = 0
    nP = wP + window
    while nP < w:
        cImgWdw = img[:,wP:nP]
        Hedge = np.sum(cImgWdw, 1)
        point = Hedge[0]
        pI = 0
        hD = 0
        lD = 0
        points = [[],[]]
        thr1 = 10000
        i = 0
        while i < len(Hedge)-1:
            i += 1
            nPoint = np.int64(Hedge[i])
            dis = np.subtract(point,nPoint)
            if dis > 0:
                if dis > lD:
                    lD = dis
                else:
                    if lD-dis > thr1:
                        points[0].append(pI)
                        points[1].append(point)
                        point = np.min(Hedge[pI:i])
                        pI = pI + np.argmin(Hedge[pI:i])
                        i = pI
                        hD = 0
                        lD = 0
            else:
                if dis < hD:
                    hD = dis
                else:
                    if dis-hD > thr1:
                        points[0].append(pI)
                        points[1].append(point)
                        point = np.max(Hedge[pI:i])
                        pI = pI + np.argmax(Hedge[pI:i])
                        i = pI
                        hD = 0
                        lD = 0
        points[0].append(pI)
        points[1].append(point)
        plt.plot(points[0],points[1],'*')
        plt.plot(Hedge)
        cv2.imshow('img', cImgWdw)
        plt.show()
        wP = nP
        nP = wP + window

    for point in points[0]:
        img[point, :] = np.add(np.zeros((1, w)), 100)
        img[point, :] = np.add(np.zeros((1, w)), 100)
        img[point, :] = np.add(np.zeros((1, w)), 100)

class JPath:
    def __init__(self, point, w=25):
        self.pointsX = [w]
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
        for i in range(0,len(self.pointsX)):
            pts.append([self.pointsX[i],self.pointsY[i]])
        pts = np.asarray(pts).reshape((-1, 1, 2))
        return pts

# enhance bright structures
def do_it(img):
    img = iPP.enhance(img)
    h,w = img.shape

    # inverse img and apply gaussian to give more importance to center
    img2 = morphology.white_tophat(img, size=500)
    img2 = 255-img2
    img2 = cv2.GaussianBlur(img2, (111,25), 5)

    # Finding path points
    window = 10
    pot_jaw_pts = []
    for x in range(window,w,window):
        offset = int(0.4*h)
        wdw_img = img2[offset:int(0.75*h),x-window:x+window]
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
        paths.append(JPath(best_i,int(0.35*len(pot_jaw_pts))*window))

    for i in range(int(0.35*len(pot_jaw_pts))+1,int(0.65*len(pot_jaw_pts))):
        for path in paths:
            path.add_best(pot_jaw_pts[i],(i+1)*window,img)

    min_intensity = float('inf')
    for path in paths:
        if path.intensity < min_intensity:
            min_intensity = path.intensity
            best_path = path

    pts = best_path.return_points()
    cv2.polylines(img,[pts],False,(255,255,255))
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img', 1500, 1000)
    cv2.imshow('img', img)
    cv2.waitKey(0)

for i in range(1,10):
    img = cv2.imread('_Data/Radiographs/0' + str(i) + '.tif')
    do_it(img)
