import ImgPP as iPP
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import scipy.signal
from scipy.ndimage import morphology
import cv2
import cv2.cv as cv

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
        paths.append(JPath(best_i, int(0.35*len(pot_jaw_pts)), window))

    for i in range(int(0.35*len(pot_jaw_pts))+1,int(0.65*len(pot_jaw_pts))):
        for path in paths:
            path.add_best(pot_jaw_pts[i],(i+1)*window,img)

    min_intensity = float('inf')
    for path in paths:
        if path.intensity < min_intensity:
            min_intensity = path.intensity
            best_path = path

    pts = best_path.return_points()
    cv2.polylines(img2,[pts],False,(255,255,255), thickness=5)
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img', 1500, 1000)
    cv2.imshow('img', img2)
    cv2.waitKey(0)
    return best_path.return_path()

def do_it2(img, path):
    img = iPP.enhance(img)
    h,w = img.shape
    start = path[0]
    window = path[1]

    imgX1 = cv2.Scharr(img, -1, 1, 0)
    imgX2 = cv2.Scharr(cv2.flip(img, 1), -1, 1, 0)
    imgX2 = cv2.flip(imgX2, 1)
    img2 = cv2.addWeighted(imgX1, 0.5, imgX2, 0.5, 0)

    # lower
    img_vals = []
    for i in range(0,len(path[2])):
        x = (start + i) * window
        y = int(path[2][i])
        img_vals.append(np.sum(img2[y+50:y+200,x-window:x+window]))
    fft = scipy.fftpack.rfft(img_vals)
    fft[30:] = 0
    smoothed = scipy.fftpack.irfft(fft)
    loc_maxs_i = scipy.signal.argrelmax(smoothed)[0]

    for max_i in loc_maxs_i:
        x = (start + max_i) * window
        y = int(path[2][max_i]) + 50
        cv2.rectangle(img2,(x,y),(x+5,y+5),(255,255,255),thickness=-1)

    plt.plot(smoothed)
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img', 1500, 1000)
    cv2.imshow('img', img2)
    plt.show()

for i in range(1,10):
    img = cv2.imread('_Data/Radiographs/0' + str(i) + '.tif')
    best_path = do_it(img)
    do_it2(img, best_path)
