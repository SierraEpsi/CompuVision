import cv2
import numpy as np
from numpy import linalg as la

def show_image(img,str,path=None):
        if path!= None:
                cv2.imwrite(path, img)
        cv2.namedWindow(str,cv2.WINDOW_NORMAL)
        cv2.resizeWindow(str,1500,1000)
        cv2.imshow(str,img)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        
def plot_polyLines(img,points,name, path=None):
        img = img.copy()
        points = np.reshape(points,(-1,1,2))
        cv2.polylines(img,points,True,(0,0,255),thickness=5)
        show_image(img,name,path)



def pca(X,n=-1):
        # based on reduced pca in an intro to ASM..
        mu = np.mean(X, axis=0)
        X = np.subtract(X, mu)
        T = np.transpose(X)
        tt = np.dot(X,T)
        tt = np.divide(tt, tt.shape[0])

        eW, eU = la.eig(tt)
        eV = np.dot(T, eU)
        idx = eW.argsort()[::-1]
        eW = eW[idx]
        eV = eV[:, idx]
        norms = la.norm(eV, axis=0)
        eV = eV / norms

        tSum = np.sum(abs(eW))
        pSum = np.cumsum(abs(eW))

        if n==-1:
                for i in range(0, len(pSum)):
                        if pSum[i] / tSum > 0.99:
                                n = i + 1
                                break

        pcm = np.array(eV[:,:n])
        eW = np.sqrt(eW[:n])
        pcm = np.array(pcm).squeeze()

        return mu, pcm, eW


def project(X,mu,pcm):
        Y = np.subtract(X, mu)
        Y = np.dot(Y, pcm)
        return Y


def reconstruct( Y, mu, pcm,doClip = -1):
        if doClip != -1:
                Y = np.clip(Y,-doClip,doClip)
        X = np.dot(Y, pcm.T)
        X = X + mu

        return X