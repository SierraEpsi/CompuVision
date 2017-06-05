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

def load_landmarks(path):
    f = open(path, 'r')
    landmarks = np.loadtxt(f)
    landmarks = np.reshape(landmarks, (landmarks.size / 2, 2))  # shape (40,2)
    return landmarks

def center_landmarks(landmarks):
    x = landmarks[:, 0]
    y = landmarks[:, 1]
    x = np.subtract(x,np.mean(x))
    y = np.subtract(y,np.mean(y))
    p1.plot(x, y, '*')
    p1.plot(x, y)
    return x, y

def rotate_shape(landmarks):
    cov = np.dot(landmarks,np.transpose(landmarks))
    n = landmarks[0].size
    cov = np.divide(cov,n-1)
    eigW, eigV = la.eig(cov)
    eigV = eigV/la.norm(eigV)
    landmarks = np.dot(eigV,landmarks)
    x = landmarks[0,:]
    y = landmarks[1,:]
    p2.plot(x, y, '*')
    p2.plot(x, y)
    return landmarks

def scale_estimate(landmarks):
    x = landmarks[0]
    y = landmarks[1]
    scale_factor = np.sqrt(np.sum(np.power(x,2)) + np.sum(np.power(y,2)))
    x = np.divide(x,scale_factor)
    y = np.divide(y,scale_factor)
    p3.plot(x, y, '*')
    p3.plot(x, y)
    return x, y

def trans_rot(landmarks):
    landmarks = center_landmarks(landmarks)

    landmarks = rotate_shape(landmarks)

    landmarks = scale_estimate(landmarks)

    return landmarks

def scale_estimate_Math(listOfLandmarks):
    listOfAreas = []

    for i in xrange(len(listOfLandmarks)):
        landmarks = listOfLandmarks[i]
        landmarks = np.reshape(landmarks,(1,landmarks.size))
        listOfAreas += cv2.contourArea(landmarks)

    meanArea = np.mean(listOfAreas)

    for i in xrange(len(listOfLandmarks)):
        landmarks = listOfLandmarks[i]
        scale_factor = listOfAreas/meanArea
        x = landmarks[0]
        y = landmarks[1]
        x = x*scale_factor
        y = y*scale_factor
        listOfLandmarks[i][0] = x
        listOfLandmarks[i][1] = y

        p3.plot(x, y, '*')
        p3.plot(x, y)


    return listOfLandmarks

def pca(teeth, nbImgs):
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

def project(X, eigV, mu):
    Y = X - mu
    Y = np.dot(Y,eigV)
    return Y

def reconstruct(Y, eigV, mu):
    X = np.dot(Y, np.transpose(eigV))
    X = X + mu
    return X

def computeModel(folderWithLandmarks,toothNbr,nbImgs,nbDims):



    X = np.zeros((nbImgs, 2 * nbDims))
    folder = folderWithLandmarks
    #listOfLandmarks = []

    for i in range(1, nbImgs + 1):
        path = folder + 'landmarks' + str(i) + '-' + str(toothNbr) + '.txt'
        landmarks = load_landmarks(path)
        landmarks = trans_rot(landmarks)
        x = landmarks[0]
        y = landmarks[1]
        X[i - 1, 0:nbDims] = x
        X[i - 1, nbDims:] = y
        #listOfLandmarks +=  [landmarks]

   # listOfLandsmarks  = scale_estimate_Math(listOfLandmarks)

    plt.show()

    eW, eV, mu = pca(X, nbImgs)
    plt.clf()
    plt.plot(mu[0:nbDims], mu[nbDims:])
    plt.axis('off')
    plt.savefig('modelTooth'+ str(toothNbr) + '.png')
    plt.clf()

    return eV, eW, mu

if __name__ == '__main__':



    #load all set of landmarks for first tooth
    folder = '_Data/landmarks/original/'
    nbImgs = 14
    nbDims = 40
    tooth = 1

    eV,eW,mu = computeModel(folder,tooth,nbImgs,nbDims)

    # Wrong example??
    example = np.reshape(load_landmarks('_Data/landmarks/original/landmarks1-1.txt'),(1,nbDims*2))[0]
    example = project(example,eV,mu)
    example = reconstruct(example,eV,mu)
    plt.plot(example[0:nbDims],example[nbDims:])
    plt.show()

    print(example)








