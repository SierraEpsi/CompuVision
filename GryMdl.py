from Landmarks import Landmarks as LMS
import ImgPP as IPP
import cv2
import numpy as np
import MnlSlctn
from ASM import ASM as ASM
from numpy import linalg as la

class GreyModel:
    def __init__(self, img_path, lmk_path, tooth_nr, nr=14):
        self.X = []
        self.mu = []
        self.pcm = []
        for i in range(1,nr+1):
            landmarks = LMS(lmk_path + str(i) + '-' + str(tooth_nr) + '.txt')
            iS = str(i)
            if i < 10:
                iS = '0' + iS
            img = cv2.imread(img_path + iS + '.tif')
            img = IPP.enhance2(img)
            self.add2model(img, landmarks)
        self.compute_model()

    def add2model(self, img, lmks, k=3):
        landmarks = lmks.as_matrix()
        for i in range(0,len(landmarks)):
            landmark = landmarks[i]
            lX = int(landmark[0])
            lY = int(landmark[1])
            img_neigh = img[lY-k:lY+k+1,lX-k:lX+k+1]
            vector = img_neigh.reshape((1,(2*k+1)**2)).squeeze()
            if len(self.X) == i:
                self.X.append([vector])
            else:
                self.X[i].append(vector)

    def compute_model(self):
        for tooth_vectors in self.X:
            mu = np.mean(tooth_vectors, axis=0)
            tooth_vectors = np.subtract(tooth_vectors, mu)
            T = np.transpose(tooth_vectors)
            tt = np.dot(np.transpose(T), T)
            tt = np.divide(tt, tt.shape[0])

            eW, eU = la.eig(tt)
            eV = np.dot(T, eU)
            idx = eW.argsort()[::-1]
            eW = eW[idx]
            eV = eV[:, idx]
            norms = la.norm(eV, axis=0)
            eV = eV / norms
            eW = eW / norms

            pcm = []
            for i in range(0, 4):
                pcm.append(eW[i] * eV[:, i])
            pcm = np.array(pcm).squeeze().transpose()
            self.mu.append(mu)
            self.pcm.append(pcm)
        self.X = []

    def sample_point(self, X, i):
        Y = np.subtract(X, self.mu[i])
        Y = np.dot(Y, self.pcm[i])
        Xo = np.dot(Y, np.transpose(self.pcm[i]))
        Xo = Xo + self.mu
        error = np.sum(np.power(X - Xo,2))
        return np.sqrt(error)

    def find_points(self, img, pts, k=5):
        n_pts = np.zeros_like(pts)
        t_error = 0
        for i in range(0,len(pts)):
            point = pts[i]
            point, error = self.find_point(img, point, i, k)
            n_pts[i,:] = point
            t_error += error
        return n_pts, t_error

    def find_point(self, img, point, i, k, k2=3):
        b_error = float('inf')
        b_point = -1
        for x in range(-k,k+1):
            for y in range(-k,k+1):
                pX = point[0] + x
                pY = point[1] + y
                img_window = img[pY-k2:pY+k2+1,pX-k2:pX+k2+1]
                vector = img_window.reshape((1,(2*k2+1)**2)).squeeze()
                error = self.sample_point(vector, i)
                if error < b_error:
                    b_error = error
                    b_point = (pX,pY)
        return b_point, b_error

    def find_best_around(self, img, pts):
        landmarks = LMS(pts)
        b_pts = -1
        b_error = float('inf')
        for x in range(-25,26,5):
            for y in range(-25,26,5):
                t_pts = landmarks.translate([x,y]).as_matrix().astype('int32')
                n_pts, error = gModel.find_points(G_img,t_pts,5)
                if error < b_error:
                    b_pts = n_pts
                    b_error = error
        return b_pts


if __name__ == '__main__':
    img = cv2.imread('_Data/Radiographs/02.tif')
    img2 = img.copy()
    G_img = IPP.enhance2(img)

    img_path = '_Data/Radiographs/'
    lmk_path = '_Data/Landmarks/original/landmarks'
    gModel = GreyModel(img_path,lmk_path,1)
    print 'done 1'

    folder = '_Data/landmarks/original/'
    nbImgs = 14
    nbDims = 40
    tooth = 1
    asm = ASM(folder, nbImgs, nbDims, 1)
    print 'done 2'

    pts = asm.mu
    landmarks = LMS(pts)
    landmarks = landmarks.scale_to_window(asm.mW)
    pts = landmarks.as_matrix().astype('int32')
    pimg = landmarks.translate(MnlSlctn.init(pts, img)).as_matrix().astype('int32')
    pimg = gModel.find_best_around(G_img,pimg)
    pimg.reshape(-1, 1, 2)
    img3 = cv2.cvtColor(G_img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img3, [pimg], True, (0, 0, 256), thickness=5)
    cv2.namedWindow('choose', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('choose', 1200, 800)
    cv2.imshow('choose', img3)
    cv2.waitKey(0)