from Landmarks import Landmarks as LMS
import ImgPP as IPP
import cv2
import numpy as np
import MnlSlctn
from ASM import ASM as ASM
from numpy import linalg as la

class InsrModel:
    def __init__(self, img_path, lmk_path, isUp, nr=14):
        self.mu = []
        self.pcm = []
        self.w = 0
        self.h = 0
        self.isUp = isUp
        X = self.make_X(img_path, isUp, lmk_path, nr)
        self.compute_model(X,True)

    def make_X(self, img_path, isUp, lmk_path, nr):
        X = np.zeros((14,250000))
        for i in range(1, nr + 1):
            if isUp:
                p00 = -1
                p01 = -1
                p10 = -1
                p11 = -1
                for j in range(1, 5):
                    landmarks = LMS(lmk_path + str(i) + '-' + str(j) + '.txt')
                    p0, p1 = landmarks.get_window()
                    if p00 == -1 or p0[0] < p00:
                        p00 = p0[0]
                    if p01 == -1 or p0[1] < p01:
                        p01 = p0[1]
                    if p10 == -1 or p1[0] > p10:
                        p10 = p1[0]
                    if p11 == -1 or p1[1] > p11:
                        p11 = p1[1]
            else:
                p00 = -1
                p01 = -1
                p10 = -1
                p11 = -1
                for j in range(5, 9):
                    landmarks = LMS(lmk_path + str(i) + '-' + str(j) + '.txt')
                    p0, p1 = landmarks.get_window()
                    if p00 == -1 or p0[0] < p00:
                        p00 = p0[0]
                    if p01 == -1 or p0[1] < p01:
                        p01 = p0[1]
                    if p10 == -1 or p1[0] > p10:
                        p10 = p1[0]
                    if p11 == -1 or p1[1] > p11:
                        p11 = p1[1]

            iS = str(i)
            if i < 10:
                iS = '0' + iS
            img = cv2.imread(img_path + iS + '.tif')
            img = IPP.enhance2(img)
            img_w = img[p01:p11, p00:p10]
            self.w += p10-p00
            self.h += p11-p01
            img_w = cv2.resize(img_w, (500, 500), interpolation=cv2.INTER_NEAREST)
            X[i-1] = (img_w.reshape((1, 250000)))
        self.w = self.w/nr
        self.h = self.h/nr
        return X

    def compute_model(self, X, doPlot=False):
        mu = np.average(X, axis=0)
        X = np.subtract(X , mu.transpose())

        eigenvalues, eigenvectors = np.linalg.eig(np.dot(X, np.transpose(X)))
        eigenvectors = np.dot(np.transpose(X), eigenvectors)

        eig = zip(eigenvalues, np.transpose(eigenvectors))
        eig = map(lambda x: (x[0] * np.linalg.norm(x[1]),
                             x[1] / np.linalg.norm(x[1])), eig)

        eig = sorted(eig, reverse=True, key=lambda x: abs(x[0]))
        eig = eig[:5]

        eigenvalues, eigenvectors = map(np.array, zip(*eig))

        self.mu = mu
        self.pcm = eigenvectors.transpose()

    def sample_vector(self, X):
        Y = np.subtract(X, self.mu)
        Y = np.dot(Y, self.pcm)
        Xo = np.dot(Y, np.transpose(self.pcm))
        Xo = Xo + self.mu
        error = la.norm(Xo-X)
        return error

    def sample_window(self, img_w):
        img_w = cv2.resize(img_w, (500, 500), interpolation=cv2.INTER_NEAREST)
        X = img_w.reshape(1, 250000).squeeze()
        return self.sample_vector(X)

    def find_best_match(self, img, pnt):
        b_error = float('inf')

        for Ty in range(0,16,5):
            for Sv in np.arange(0.8,1.2,0.1):
                for Sh in np.arange(0.8,1.2,0.1):
                    if self.isUp:
                        P00 = int(pnt[0] - int(Sh*self.w / 2))
                        P01 = int(pnt[0] + int(Sh*self.w / 2) + 1)
                        P10 = int(pnt[1] - int(Sv*self.h) - Ty)
                        P11 = int(pnt[1] - Ty)
                    else:
                        P00 = int(pnt[0] - int(Sh * self.w / 2))
                        P01 = int(pnt[0] + int(Sh * self.w / 2) + 1)
                        P10 = int(pnt[1] + Ty)
                        P11 = int(pnt[1] + int(Sv * self.h) + Ty)
                    img_w = img[P10:P11, P00:P01]
                    error = self.sample_window(img_w)
                    if error < b_error:
                        b_error = error
                        b_pnts = ((P00,P10),(P01,P11))
        return b_error, b_pnts


if __name__ == '__main__':
    img_path = '_Data/Radiographs/'
    lmk_path = '_Data/Landmarks/original/landmarks'
    tModel = InsrModel(img_path,lmk_path,False)

    from ImgSgm import find_jawline
    from ImgSgm import find_POI
    import ImgPP as iPP

    for id in range(1,9):
        img = cv2.imread('_Data/Radiographs/0' + str(id) + '.tif')
        img2 = img.copy()
        G_IMG = iPP.enhance2(img)
        best_path = find_jawline(img)
        start = best_path[0]
        window = best_path[1]
        n = len(best_path[2])
        b_error = float('inf')
        b_points = -1
        for i in range(int(0.4*n),int(0.6*n)):
            x = (start+i) * window
            y = best_path[2][i]
            error, pnts = tModel.find_best_match(G_IMG,(x,y))
            if error < b_error:
                b_error = error
                b_points = pnts
        find_POI(img, b_points, False)