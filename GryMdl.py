from Landmarks import Landmarks as LMS
import ImgPP as IPP
import cv2
import numpy as np
import scipy.signal as sg
import MnlSlctn
from ASM import ASM as ASM
from numpy import linalg as la
import Util as ut

class GreyModel:
    def __init__(self, img_path, lmk_path, tooth_nr, nr=14,k=3, useGrad = True):

        self.mu = []
        self.pcm = []
        self.k = k
        self.useGrad = useGrad
        X = []
        for i in range(1,nr+1):
            landmarks = LMS(lmk_path + str(i) + '-' + str(tooth_nr) + '.txt')
            iS = str(i)
            if i < 10:
                iS = '0' + iS
            img = cv2.imread(img_path + iS + '.tif')
            if useGrad:
                img = IPP.enhance2(img)
            else:
                img = IPP.GRimg(img)
            X = self.add2model(X,img, landmarks)
        self.compute_model(X)

    def add2model(self,X, img, lmks):
        k = self.k
        landmarks = lmks.as_matrix()
        for i in range(0,len(landmarks)):
            landmark = landmarks[i]
            lX = int(landmark[0])
            lY = int(landmark[1])
            img_neigh = img[lY-k:lY+k+1,lX-k:lX+k+1]
            vector = img_neigh.reshape((1,(2*k+1)**2)).squeeze().astype('int32')
            if len(X) == i:
                X.append([vector])
            else:
                X[i].append(vector)
        return X

    def compute_model(self,X):
        for tooth_vectors in X:
            mu,pcm = ut.pca(tooth_vectors,4)
            self.mu.append(mu)
            self.pcm.append(pcm)

    def sample_point(self, X, i):
        Y = ut.project(X,self.mu[i],self.pcm[i])
        Xo = ut.reconstruct(Y,self.mu[i],self.pcm[i],2)
        error = np.linalg.norm(Xo-X)
        return error

    def find_points(self, img, pts, steps):

        n_pts = np.zeros_like(pts)
        t_error = 0
        for i in range(0,len(pts)):
            pt0 = pts[i - 1]
            pt1 = pts[i]
            if i==len(pts)-1:
                pt2=pts[0]
            else:
                pt2 = pts[i + 1]

            pt0, error = self.find_point(img, pt0,pt1,pt2,i,steps)
            n_pts[i,:] = pt0
            t_error += error**2
        n_pts = self.apply_medfilt(n_pts,5)

        return n_pts, np.sqrt(t_error)

    def find_point(self, img, p0,p1,p2, l_nr, steps):
        k = self.k
        b_error = float('inf')
        b_point = -1
        xt = p2[0] - p0[0]
        yt = p2[1] - p0[1]
        norm = np.sqrt(xt**2+yt**2)
        xt = xt/norm
        yt = yt/norm
        xn = yt
        yn = -xt

        if abs(xn) > abs(yn):
            yn = yn/xn
            xn = 1
        else:
            xn = xn/yn
            yn = 1

        for i in range(-steps,steps+1):
            pX = p1[0] + int(i*xn)
            pY = p1[1] + int(i*yn)
            img_window = img[pY-k:pY+k+1,pX-k:pX+k+1]
            vector = img_window.reshape((1,(2*k+1)**2)).squeeze().astype('int32')
            error = self.sample_point(vector, l_nr)
            if error < b_error:
                b_error = error
                b_point = [pX,pY]

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

    def apply_medfilt(self,pts,w=5):
        pts = pts.T
        offset = (w-1)/2
        x = np.hstack((pts[0][-offset:],pts[0],pts[0][0:offset]))
        y = np.hstack((pts[1][-offset:],pts[1],pts[1][0:offset]))
        x = sg.medfilt(x, w)[offset:-offset]
        y = sg.medfilt(y, w)[offset:-offset]

        return np.transpose([x,y]).astype('int32')

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
    while True:
        pimg,_ = gModel.find_points(G_img,pimg,10)
        pimg.reshape(-1, 1, 2)




        img3 = cv2.cvtColor(G_img, cv2.COLOR_GRAY2BGR)
        cv2.polylines(img3, [pimg], True, (0, 0, 256), thickness=5)
        cv2.namedWindow('choose', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('choose', 1200, 800)
        cv2.imshow('choose', img3)
        cv2.waitKey(0)

