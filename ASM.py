import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import cv2
import cv

global fig
global p1
global p2
global p3
global p4

fig = plt.figure()
p1 = fig.add_subplot(131)
p2 = fig.add_subplot(132)
p3 = fig.add_subplot(133)

class ASM:
    def __init__(self, folder_path, nbImgs, nbDims, tooth):
        self.mu = None
        self.eV = None

        self.eVr = None
        self.sF = None
        self.computeModel(folder_path,tooth,nbImgs,nbDims)

    def load_landmarks(self, path):
        f = open(path, 'r')
        landmarks = np.loadtxt(f)
        landmarks = np.reshape(landmarks, (landmarks.size / 2,2))  # shape (40,2)
        return landmarks

    def center_landmarks(self, landmarks):
        x = landmarks[:,0]
        y = landmarks[:,1]
        x = np.subtract(x,np.mean(x))
        y = np.subtract(y,np.mean(y))
        p1.plot(x, y, '*')
        p1.plot(x, y)

        array = np.zeros(landmarks.shape)
        array[:,0] = x
        array[:,1] = y
        return array

    def rotate_shape(self, landmarks, eigV1=None):
        cov = np.dot(landmarks.T, landmarks)
        n = landmarks.shape[0]
        cov = np.divide(cov, n - 1)
        eigW, eigV = la.eig(cov)

        if eigV1 == None:
            x = landmarks[:, 0]
            y = landmarks[:, 1]
            rotated = landmarks
            eigV1 = eigV
        else:
            rotated = np.dot(landmarks,eigV)
            rotated = np.dot(rotated, eigV1.T)
            x = rotated[:,0]
            y = rotated[:,1]
        p2.plot(x, y, '*')
        p2.plot(x, y)

        return rotated, eigV1

    def scale_estimate(self, landmarks):
        x = landmarks[:,0]
        y = landmarks[:,1]
        scale_factor = np.sqrt(np.sum(np.power(x,2)) + np.sum(np.power(y,2)))
        x = np.divide(x,scale_factor)
        y = np.divide(y,scale_factor)

        array = np.zeros(landmarks.shape)
        array[:,0] = x
        array[:,1] = y
        return array, scale_factor

    def rescale(self, landmarks, meanSF):
        x = landmarks[0]
        y = landmarks[1]
        x = x*meanSF
        y = y*meanSF
        p3.plot(x, y, '*')
        p3.plot(x, y)

        return x,y

    def pca(self, teeth, nbImgs):
        mu = np.mean(teeth, axis=0)
        teeth = np.subtract(teeth, mu)
        T = np.transpose(teeth)
        tt = np.dot(np.transpose(T), T)
        eW, eU = la.eig(tt)
        eV = np.dot(T, eU)
        idx = eW.argsort()[::-1]
        eW = eW[idx]
        eV = eV[:, idx]
        norms = la.norm(eV, axis=0)
        eV = eV / norms
        return eW[:nbImgs], eV[:,:nbImgs], mu

    def model(self, path):
        points = self.load_landmarks(path)

        x = points[:, 0]
        y = points[:, 1]
        x = np.subtract(x, np.mean(x))
        y = np.subtract(y, np.mean(y))
        points = np.zeros(points.shape)
        points[:, 0] = x
        points[:, 1] = y

        cov = np.dot(points.T, points)
        n = points.shape[0]
        cov = np.divide(cov, n - 1)
        eigW, eigV = la.eig(cov)
        rotated = np.dot(points, eigV)
        rotated = np.dot(rotated, self.eVr.T)
        x = rotated[:, 0]
        y = rotated[:, 1]

        scale_factor = np.sqrt(np.sum(np.power(x, 2)) + np.sum(np.power(y, 2)))
        x = np.divide(x, scale_factor)
        y = np.divide(y, scale_factor)

        x = x*self.sF
        y = y*self.sF

        w = points.shape[0]
        h = points.shape[1]
        X = [x,y]
        X = np.reshape(X,(w*h,1))[:,0]
        Y = self.project(X)
        X = self.reconstruct(Y)

        return [X[:w],X[w:]]

    def project(self, X):
        Y = np.subtract(X ,self.mu)
        Y = np.dot(Y,self.eV)
        return Y

    def reconstruct(self, Y):
        X = np.dot(Y, np.transpose(self.eV))
        X = X + self.mu
        return X

    def computeModel(self, folderWithLandmarks,toothNbr,nbImgs,nbDims):
        X = np.zeros((nbImgs, 2 * nbDims))
        folder = folderWithLandmarks
        SF = []
        eigV = None

        for i in range(1, nbImgs + 1):
            path = folder + 'landmarks' + str(i) + '-' + str(toothNbr) + '.txt'
            landmarks = self.load_landmarks(path)
            landmarks = self.center_landmarks(landmarks)
            landmarks, eigV = self.rotate_shape(landmarks,eigV) # uses first set as reference
            landmarks, sf = self.scale_estimate(landmarks)
            x = landmarks[:,0]
            y = landmarks[:,1]
            X[i - 1, 0:nbDims] = x
            X[i - 1, nbDims:] = y
            SF += [sf]

        meanSF = np.mean(SF)
        self.eVr = eigV
        self.sF = meanSF

        for i in range(0, nbImgs):
            landmarks = self.rescale([X[i, 0:nbDims],X[i, nbDims:]],meanSF)
            x = landmarks[0]
            y = landmarks[1]
            X[i, 0:nbDims] = x
            X[i, nbDims:] = y

        plt.show()

        eW, eV, mu = self.pca(X, nbImgs)
        self.eV = eV
        self.mu = mu

# test it
if __name__ == '__main__':
    # load all set of landmarks for first tooth
    folder = '_Data/landmarks/original/'
    nbImgs = 14
    nbDims = 40
    tooth = 1
    active_shape_model = ASM(folder, nbImgs, nbDims, tooth)
    X = active_shape_model.model('_Data/landmarks/original/landmarks2-1.txt')
    plt.clf()
    plt.plot(X[0],X[1], '*')
    plt.show()








