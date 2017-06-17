import cv2
import numpy as np

def show_image(img):
        cv2.namedWindow('img',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('img',1500,1000)
        cv2.imshow('img',img)
        cv2.waitKey()
        
def plot_polyLines(img,points):
        img = img.copy()
        points = np.reshape(points,(-1,1,2))
        cv2.polylines(img,points,True,(0,0,255),thickness=5)


def pca(X):
        mu = np.mean(X, axis=0)
        X = np.subtract(X, mu)
        T = np.transpose(X)
        tt = np.dot(np.transpose(T), T)
        tt = np.divide(tt, tt.shape[0]-1)

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
        for i in range(0, len(pSum)):
                if pSum[i] / tSum > 0.99:
                        choice = i
                        break
        pcm = []
        for i in range(0, 8):
                pcm.append(eW[i] * eV[:, i])
        pcm = np.array(pcm).squeeze().transpose()
        return mu, pcm