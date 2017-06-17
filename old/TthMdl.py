from Landmarks import Landmarks as LMS
import ImgPP as IPP
import cv2
import numpy as np
import MnlSlctn
from ASM import ASM as ASM
from numpy import linalg as la

class ToothModel:
    def __init__(self, img_path, lmk_path, tooth_nr, nr=14):
        self.X = []
        self.mu = []
        self.pcm = []
        self.w = 0
        self.h = 0
        for i in range(2,nr+1):
            landmarks = LMS(lmk_path + str(i) + '-' + str(tooth_nr) + '.txt')
            iS = str(i)
            if i < 10:
                iS = '0' + iS
            img = cv2.imread(img_path + iS + '.tif')
            img = IPP.enhance2(img)
            self.add2model(img, landmarks)
        self.h = int(self.h/nr)
        self.w = int(self.w/nr)
        self.compute_model()

    def add2model(self, img, lmks):
        P0, P1 = lmks.get_window()
        w, h = lmks.get_dimensions()
        self.w += w
        self.h += h
        img_w = img[P0[1]:P1[1],P0[0]:P1[0]]
        img_w = cv2.resize(img_w,(100,250))
        img_w = img_w.reshape(1,25000).squeeze()
        self.X.append(img_w)

    def compute_model(self):
        mu = np.average(self.X, axis=0)
        X = np.subtract(self.X , mu.transpose())

        eigenvalues, eigenvectors = np.linalg.eig(np.dot(X, np.transpose(X)))
        eigenvectors = np.dot(np.transpose(X), eigenvectors)

        eig = zip(eigenvalues, np.transpose(eigenvectors))
        eig = map(lambda x: (x[0] * np.linalg.norm(x[1]),
                             x[1] / np.linalg.norm(x[1])), eig)

        eig = sorted(eig, reverse=True, key=lambda x: abs(x[0]))
        eig = eig[:8]

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
        img_w = cv2.resize(img_w, (100, 250))
        X = img_w.reshape(1, 25000).squeeze()
        return self.sample_vector(X)

    def find_best_match(self, img, pnts):
        b_error = float('inf')
        b_point = -1
        for pnt in pnts:
            P00 = pnt[0] - int(self.w / 2)
            P01 = pnt[0] + int(self.w / 2) + 1
            P10 = pnt[1] - int(self.h / 2)
            P11 = pnt[1] + int(self.h / 2) + 1
            for x in range(-25,26,5):
                for y in range(-25,26,5):
                    img_w = img[P10+y:P11+y, P00+x:P01+x]
                    error = self.sample_window(img_w)
                    if error < b_error:
                        b_error = error
                        b_point = (pnt[0]+x,pnt[1]+y)

        img3 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('img', 1500, 1000)
        cv2.rectangle(img3, b_point, (b_point[0]+5,b_point[1]+5),(0,256,256),thickness=-1)
        cv2.imshow('img', img3)
        cv2.waitKey(0)


if __name__ == '__main__':
    img_path = '_Data/Radiographs/'
    lmk_path = '_Data/Landmarks/original/landmarks'
    tModel = ToothModel(img_path,lmk_path,1)