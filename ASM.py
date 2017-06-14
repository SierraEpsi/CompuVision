import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from Landmarks import Landmarks
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
    def __init__(self, folder_path, nbImgs, nbDims, tooth, doPlot = False):
        self.mu = None
        self.eV = None
        self.eW = None

        self.eVr = None
        self.sF = None
        self.computeModel2(folder_path,tooth,nbImgs,nbDims, doPlot)

    def computeModel(self, folder_path, toothNbr, nbImgs, nbDims, doPlot = False):
        X = np.zeros((nbImgs, 2 * nbDims))
        SF = []
        eigV = None

        all_landmarks = []
        for i in range(1, nbImgs + 1):
            path = folder_path + 'landmarks' + str(i) + '-' + str(toothNbr) + '.txt'
            landmark = Landmarks(path)
            all_landmarks.append(landmark.translate_to_origin())

        if doPlot:
            plt.clf()
            for i in range(0,len(all_landmarks)):
                pts = all_landmarks[i].as_matrix()
                plt.plot(pts[:,0],pts[:,1])
            plt.show()

        for i in range(0, len(all_landmarks)):
            landmarks, eigV = self.rotate_shape(all_landmarks[i], eigV)  # uses first set as reference
            landmarks, sf = self.scale_estimate(landmarks)
            landmarks = landmarks.as_matrix()
            x = landmarks[:, 0]
            y = landmarks[:, 1]
            X[i - 1, 0:nbDims] = x
            X[i - 1, nbDims:] = y
            SF += [sf]

        meanSF = np.mean(SF)
        self.eVr = eigV
        self.sF = meanSF

        for i in range(0, nbImgs):
            landmarks = Landmarks(X[i]).scale(meanSF).as_matrix()
            x = landmarks[:,0]
            y = landmarks[:,1]
            X[i, 0:nbDims] = x
            X[i, nbDims:] = y

        if doPlot:
            plt.clf()
            for i in range(0,len(X)):
                pts = X[i]
                plt.plot(pts[:nbDims],pts[nbDims:])
            plt.show()

        _,eV = self.pca(X)
        self.mu = np.mean(X, axis=0)
        self.eV = eV

    def computeModel2(self, folder_path, toothNbr, nbImgs, nbDims, doPlot=False):
        all_landmarks = []
        for i in range(1, nbImgs + 1):
            path = folder_path + 'landmarks' + str(i) + '-' + str(toothNbr) + '.txt'
            landmark = Landmarks(path)
            all_landmarks.append(landmark.translate_to_origin())

        mu = all_landmarks[0].scale_to_unit()
        x0 = mu
        while True:
            n_mu = np.zeros_like(all_landmarks[0])
            for i in range(0,len(all_landmarks)):
                all_landmarks[i] = self.align_shape(mu,all_landmarks[i]).translate_to_origin()
                n_mu = n_mu + all_landmarks[i].as_matrix()
            n_mu = Landmarks(np.divide(n_mu,len(all_landmarks)))
            n_mu = self.align_shape(x0,n_mu)
            n_mu = n_mu.translate_to_origin()
            n_mu = n_mu.scale_to_unit()
            mu = n_mu
            if(abs(mu.as_vector() - n_mu.as_vector()) < 1e-15).all():
                break

        if doPlot:
            plt.clf()
            for i in range(0,len(all_landmarks)):
                pts = all_landmarks[i].as_matrix()
                plt.plot(pts[:,0],pts[:,1])
            plt.show()

        X = []
        for i in range(0, len(all_landmarks)):
            landmarks = all_landmarks[i]
            X.append(landmarks.as_vector())
        X = np.array(X)

        _,pcm = self.pca(X)
        self.mu = mu.as_vector()
        self.eV = pcm

    def align_shape(self, x1,x2):
        x1V = x1.as_vector()
        x2V = x2.as_vector()
        n = len(x1V)/2

        nX = (la.norm(x2V)**2)
        a = np.dot(x1V, x2V) / nX
        b = (np.dot(x2V[:n], x1V[n:]) - np.dot(x2V[n:], x1V[:n])) / nX

        sf = np.sqrt(a**2 + b**2)
        angle = np.arctan(b/a)

        x2 = x2.rotate(angle)
        x2 = x2.scale(sf)

        x2V = x2.as_vector()
        xx = np.dot(x1V, x2V)

        return Landmarks(x2V*(1.0/xx))

    def rotate_shape(self, landmarks, eigV1):
        landmarks = landmarks.as_matrix()
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
        p2.plot(x, y)

        return Landmarks(rotated), eigV1

    def scale_estimate(self, landmarks):
        landmarks = landmarks.as_matrix()
        x = landmarks[:,0]
        y = landmarks[:,1]
        scale_factor = np.sqrt(np.sum(np.power(x,2)) + np.sum(np.power(y,2)))
        x = np.divide(x,scale_factor)
        y = np.divide(y,scale_factor)

        array = np.zeros(landmarks.shape)
        array[:,0] = x
        array[:,1] = y
        return Landmarks(array), scale_factor

    def pca(self, teeth):
        mu = np.mean(teeth, axis=0)
        teeth = np.subtract(teeth, mu)
        T = np.transpose(teeth)
        tt = np.dot(np.transpose(T), T)
        tt = np.divide(tt,tt.shape[0])

        eW, eU = la.eig(tt)
        eV = np.dot(T, eU)
        idx = eW.argsort()[::-1]
        eW = eW[idx]
        eV = eV[:, idx]
        norms = la.norm(eV, axis=0)
        eV = eV / norms
        eW = eW / norms

        tSum = np.sum(abs(eW))
        pSum = np.cumsum(abs(eW))
        choice = -1
        for i in range(0,len(pSum)):
            if pSum[i]/tSum > 0.99:
                choice = i
                break
        pcm = []
        for i in range(0, 8):
           pcm.append(eW[i] * eV[:,i])
        pcm = np.array(pcm).squeeze().transpose()
        return mu, pcm

    def pca2(self, teeth):
        S = np.cov(teeth, rowvar=0)
        eW, eV = la.eigh(S)

        # only keep best  n eigW
        idx = np.argsort(-eW)  # reverse sort
        eW = eW[idx]
        eV = eV[:,idx]

        tSum = np.sum(eW)
        pSum = np.cumsum(eW)
        choice = -1
        for i in range(0,len(pSum)):
            if pSum[i]/tSum > 0.99:
                choice = i
                break
        pcm = []
        for i in range(0, choice+1):
           pcm.append(np.sqrt(eW[i]) * eV[:,i])
        pcm = np.array(pcm).squeeze().transpose()
        return pcm

    def explain_pca(self, eWs, eVs):
        tEW = np.sum(eWs)
        eWs, eVs = zip(*sorted(zip(eWs, eVs),reverse=True))
        thr = 0.05 * eWs[0]
        sEW = 0
        i = 0
        choice = -1
        for eW in eWs:
            i += 1
            sEW += eW
            if sEW/tEW > 0.99 and choice == -1:
                choice = i
            print i, 'eigenwaarden slagen erin om', sEW/tEW, '% van de data uit te leggen.'
        print 'Ik het slim algoritme dat snel in elkaar is gestoken geweest stel dus voor om', choice, 'eigenwaarden te behouden.'
        print 'lijkt u dat ook een goed idee?'
        print 'Nu volgen er enkele plots die deze voorstellen.'
        dummy = [0]*choice
        for i in range(-30,40,10):
            n_dummy = dummy
            n_dummy[0] = i
            X = self.reconstruct(n_dummy,choice)
            X = np.reshape(X,(2,len(X)/2))
            plt.plot(X[0],X[1])
        plt.show()

    def model(self, X, doPlot=False):
        X = X.translate_to_origin()
        X = self.align_shape(Landmarks(self.mu), X)
        Xo = X.as_matrix()

        Y = self.project(X.as_vector())
        X = self.reconstruct(Y)
        X = Landmarks(X).as_matrix()

        if doPlot == True:
            plt.clf()
            plt.plot(Xo[:,0], Xo[:,1], '*')
            plt.plot(Xo[:,0], Xo[:,1])
            plt.plot(X[:,0], X[:,1], '*')
            plt.plot(X[:,0], X[:,1])
            plt.show()

        # Root mean square
        error = np.sum(np.power((np.asarray(Xo)-np.asarray(X)),2),1)/len(X)
        # Total length
        error = np.sqrt(np.power(error[0],2) + np.power(error[1],2))

        return X, error

    def project(self, X):
        Y = np.subtract(X ,self.mu)
        Y = np.dot(Y,self.eV)
        # clip Y
        return Y

    def reconstruct(self, Y, n=None):
        if n == None: n = len(self.eV)
        X = np.dot(Y, np.transpose(self.eV[:,:n]))
        X = X + self.mu
        return X

    def get_search_box(self, point):
        n = len(self.mu)/2
        x1 = np.min(self.mu[:n])
        x2 = np.max(self.mu[:n])
        y1 = np.min(self.mu[n:])
        y2 = np.max(self.mu[n:])
        off = np.max((int((x2 - x1)/2),int((y2 - y1)/2)))
        return (point[0] - off, point[1] - off),(point[0] + off, point[1] + off)

# test it
if __name__ == '__main__':
    # load all set of landmarks for first tooth
    folder = '_Data/landmarks/original/'
    nbImgs = 14
    nbDims = 40
    tooth = 1

    active_shape_model = ASM(folder, nbImgs, nbDims, tooth, True)

    plt.clf()
    fig = plt.figure()
    weights = [-9,-4,-1,0,1,4,9]
    for j in range(0,8):
        p = fig.add_subplot(int('24' + str(j+1)))
        for i in range(0,7):
            pts = [0,0,0,0,0,0,0,0]
            pts[j] = weights[i]
            pts = active_shape_model.reconstruct(pts)
            p.plot(pts[:40],pts[40:])
    plt.show()

    tooth1 = Landmarks('_Data/landmarks/original/landmarks2-1.txt')
    tooth2 = Landmarks('_Data/landmarks/original/landmarks2-2.txt')
    X, error = active_shape_model.model(tooth1,True)
    print "The error of a matching tooth: ", error
    X, error = active_shape_model.model(tooth2,True)
    print "The error of a non-matching tooth: ", error






