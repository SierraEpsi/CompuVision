import ImgPP as iPP
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import scipy.signal
import scipy.spatial.distance as spDst
import scipy.cluster.hierarchy as spHrch
from scipy.ndimage import morphology
import cv2
import cv2.cv as cv
from TthMdl import ToothModel as TML

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

class TPath:
    def __init__(self, point):
        self.points = [point]
        self.intensity = 0

    def add_best(self, some_points, img):
        min_intensity = float('inf')
        best_point = -1
        index = -1
        for i in range(0,len(some_points)):
            point = some_points[i]
            intensity = cv.InitLineIterator(cv.fromarray(img), self.points[-1], point)
            if intensity < min_intensity:
                min_intensity = intensity
                best_point = point
                index = i
        if best_point != -1:
            self.points.append(best_point)
            self.intensity += min_intensity
        return index

def find_jawline(img):
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

def find_lower_pairs(img, path):
    img = iPP.enhance(img)
    h,w = img.shape
    start = path[0]
    window = path[1]

    imgX1 = cv2.Scharr(img, -1, 1, 0)
    imgX2 = cv2.Scharr(cv2.flip(img, 1), -1, 1, 0)
    imgX2 = cv2.flip(imgX2, 1)
    img2 = cv2.addWeighted(imgX1, 0.5, imgX2, 0.5, 0)

    # lower
    w2 = 10
    points = []
    for w1 in range (50,160,w2):
        off1 = w1
        off2 = w1 + w2
        img_vals = []
        for i in range(0,len(path[2])):
            x = (start + i) * window
            y = int(path[2][i])
            img_val = np.sum(img2[y+off1:y+off2,x-window:x+window])
            img_vals.append(img_val)
        fft = scipy.fftpack.rfft(img_vals)
        fft[30:] = 0
        smoothed = scipy.fftpack.irfft(fft)
        loc_maxs_i = scipy.signal.argrelmax(smoothed)[0]
        for point in loc_maxs_i:
            x1 = (start + point) * window
            y1 = int(path[2][point]) + w1
            points.append((x1,y1))

    paths = []
    thr = 25
    for point in points:
        path404 = True
        for path in paths:
            if abs(path[-1][0] - point[0]) < thr:
                path.append(point)
                path404 = False
        if path404:
            paths.append([point])

    g_paths = []
    for path in paths:
        if len(path) > 4:
            g_paths.append(path)

    POI = []
    g_paths = sorted(g_paths)
    cPath = g_paths[0]
    cY = np.mean(cPath,0)[0]
    for i in range(1,len(g_paths)-1):
        nPath = g_paths[i]
        nY = np.mean(nPath,0)[0]
        if abs(nY-cY) < 25:
            pass
        else:
            y = int((cY + nY)/2)
            cPath.extend(nPath)
            x = int(np.mean(cPath,0)[1])
            cY = nY
            cPath = nPath
            POI.append((y,x))

    return POI

def find_upper_pairs(img, path):
    img = iPP.enhance(img)
    h,w = img.shape
    start = path[0]
    window = path[1]

    imgX1 = cv2.Scharr(img, -1, 1, 0)
    imgX2 = cv2.Scharr(cv2.flip(img, 1), -1, 1, 0)
    imgX2 = cv2.flip(imgX2, 1)
    img2 = cv2.addWeighted(imgX1, 0.5, imgX2, 0.5, 0)

    # lower
    w2 = 20
    points = []
    for w1 in range (40,200,w2):
        off1 = w1
        off2 = w1 + w2
        img_vals = []
        for i in range(0,len(path[2])):
            x = (start + i) * window
            y = int(path[2][i])
            img_val = np.sum(img2[y-off2:y-off1,x-window:x+window])
            img_vals.append(img_val)
        fft = scipy.fftpack.rfft(img_vals)
        fft[30:] = 0
        smoothed = scipy.fftpack.irfft(fft)
        loc_maxs_i = scipy.signal.argrelmax(smoothed)[0]
        for point in loc_maxs_i:
            x1 = (start + point) * window
            y1 = int(path[2][point]) - w1
            points.append((x1,y1))

    paths = []
    thr = 25
    for point in points:
        path404 = True
        for path in paths:
            if abs(path[-1][0] - point[0]) < thr:
                path.append(point)
                path404 = False
        if path404:
            paths.append([point])

    g_paths = []
    for path in paths:
        if len(path) > 4:
            g_paths.append(path)

    POI = []
    g_paths = sorted(g_paths)
    cPath = g_paths[0]
    cY = np.mean(cPath,0)[0]
    for i in range(1,len(g_paths)-1):
        nPath = g_paths[i]
        nY = np.mean(nPath,0)[0]
        if abs(nY-cY) < 25:
            pass
        else:
            y = int((cY + nY)/2)
            cPath.extend(nPath)
            x = int(np.mean(cPath,0)[1])
            cY = nY
            cPath = nPath
            POI.append((y,x))

    return POI

def refine_POI(pois, img, k=10):
    img = iPP.enhance2(img)
    print 'start'
    for poi in pois:
        P00 = poi[0] - k
        P01 = poi[0] + k + 1
        P10 = poi[1] - k
        P11 = poi[1] + k + 1
        img_window = img[P10:P11,P00:P01]
        img_val = np.sqrt(np.sum(np.power(img_window,2)))
        print img_val

def find_POI(img,pairs):
    POI = []
    cPair = pairs[0]
    i = 1
    while i < len(pairs) - 1:
        nPair = pairs[i]
        if abs(cPair[0][0] - nPair[0][0]) < 25 and abs(cPair[1][0] - nPair[1][0]) < 25:
            dis1 = sqrt(np.pow(cPair[0][0]-cPair[1][0],2) + np.pow(cPair[0][1]-cPair[1][1],2))
            dis2 = sqrt(np.pow(nPair[0][0]-nPair[1][0],2) + np.pow(nPair[0][1]-nPair[1][1],2))
            if dis2 < dis1:
                cPair = nPair
        else:
            x1 = np.min((cPair[0][0],cPair[1][0],nPair[0][0],nPair[1][0]))
            x2 = np.max((cPair[0][0],cPair[1][0],nPair[0][0],nPair[1][0]))
            y1 = np.min((cPair[0][1],cPair[1][1],nPair[0][1],nPair[1][1]))
            y2 = np.max((cPair[0][1],cPair[1][1],nPair[0][1],nPair[1][1]))
            roI = img[y1:y2,x1:x2]
            POI.append((int((x1+x2)/2),int((y1+y2)/2)))
            cPair = nPair
        i += 1
    return POI


def test1():
    for i in range(1, 10):
        img = cv2.imread('_Data/Radiographs/0' + str(i) + '.tif')
        img2 = img.copy()
        best_path = find_jawline(img)

        POIL = find_lower_pairs(img, best_path)
        POIU = find_upper_pairs(img, best_path)

        refine_POI(POIU,img)
        for poi in POIU:
            cv2.rectangle(img2, poi, (poi[0] + 5, poi[1] + 5), (0, 255, 0), thickness=-1)

        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('img', 1500, 1000)
        cv2.imshow('img', img2)
        cv2.waitKey(0)
    for i in range(10, 15):
        img = cv2.imread('_Data/Radiographs/' + str(i) + '.tif')
        img2 = img.copy()
        best_path = find_jawline(img)

        POI = find_lower_pairs(img, best_path)
        POI.extend(find_upper_pairs(img, best_path))

        for poi in POI:
            cv2.rectangle(img2, poi, (poi[0] + 5, poi[1] + 5), (0, 255, 0), thickness=-1)

        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('img', 1500, 1000)
        cv2.imshow('img', img2)
        cv2.waitKey(0)




if __name__ == '__main__':
    img = cv2.imread('_Data/Radiographs/01.tif')
    img2 = img.copy()
    G_IMG = iPP.enhance2(img)
    best_path = find_jawline(img)
#
    POIL = find_lower_pairs(img, best_path)
    POIU = find_upper_pairs(img, best_path)
#
    print 'done 1'
#
    img_path = '_Data/Radiographs/'
    lmk_path = '_Data/Landmarks/original/landmarks'
    tModel = TML(img_path,lmk_path,4)
#
    print 'done 2'
#
    tModel.find_best_match(G_IMG,POIU)