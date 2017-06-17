import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from Landmarks import Landmarks
import Util as ut
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
        self.pcm = None
        self.eW = None
        self.mW = [0,0]

        self.computeModel(folder_path,tooth,nbImgs,nbDims, doPlot)

    def computeModel(self, folder_path, toothNbr, nbImgs, nbDims, doPlot=False):
        all_landmarks = []
        for i in range(1, nbImgs + 1):
            path = folder_path + 'landmarks' + str(i) + '-' + str(toothNbr) + '.txt'
            landmark = Landmarks(path)
            w, h = landmark.get_dimensions()
            self.mW[0] += w
            self.mW[1] += h
            all_landmarks.append(landmark.translate_to_origin())
        self.mW = np.divide(self.mW ,len(all_landmarks))

        mu = all_landmarks[0].scale_to_unit()
        x0 = mu
        while True:
            n_mu = np.zeros_like(all_landmarks[0])
            for i in range(0,len(all_landmarks)):
                all_landmarks[i] = self.allign_shape(mu,all_landmarks[i])
                n_mu = n_mu + all_landmarks[i].as_matrix()
            n_mu = Landmarks(np.divide(n_mu,len(all_landmarks)))
            n_mu = self.allign_shape(x0,n_mu)
            n_mu = n_mu.scale_to_unit()

            if(abs(mu.as_vector() - n_mu.as_vector()) < 1e-15).all():
                break
            mu = n_mu

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

        _, self.pcm, self.eW = ut.pca(X)
        self.mu = mu.as_vector()

    def allign_shape(self, x1, x2):
        
        t,sf,angle = self.allign_param(x1,x2)
        
        x1V = x1.as_vector()
        x2 = x2.rotate(angle)
        x2 = x2.scale(sf)
        x2V = x2.as_vector()
        xx = np.dot(x2V, x1V)

        return Landmarks(x2V / xx)

    def allign_param(self, x1,x2):
        # translation & center to or
        t = x1.get_centroid() - x2.get_centroid()
        x1 = x1.translate_to_origin()
        x2 = x2.translate_to_origin()

        x1V = x1.as_vector()
        x2V = x2.as_vector()

        n = len(x1V) / 2

        nX = (la.norm(x2V) ** 2)
        a = np.dot(x1V, x2V) / nX
        b = (np.dot(x2V[:n], x1V[n:]) - np.dot(x2V[n:], x1V[:n])) / nX

        sf = np.sqrt(a ** 2 + b ** 2)
        angle = np.arctan(b / a)
            
        return t,sf,angle
    
    def model(self, X, doPlot=False):
        X = X.translate_to_origin()
        X = self.allign_shape(Landmarks(self.mu), X)
        Xo = X.as_matrix()

        Y = ut.project(X.as_vector(),self.mu,self.pcm)
        X = ut.reconstruct(Y,self.mu,self.pcm)
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

    def estimate_trans(self, pts):
        y = Landmarks(pts)
        b = np.zeros(self.pcm.shape[1])
        diff = 1
        Tx = 0
        Ty = 0
        tsf = 1.0
        tAngle = 0
        maxit = 100
        it =0
        while diff > 1e-18 and it<maxit:
            x = Landmarks(ut.reconstruct(b,self.mu,self.pcm))
            t,sf,angle = self.allign_param(y,x)

            Tx += t[0]
            Ty += t[1]
            tsf *= sf
            tAngle += angle
            tAngle = np.clip(tAngle,-0.55,0.55)

            y = y.invT(t,sf,angle)
            # plt.plot(x.as_matrix()[:,0],x.as_matrix()[:,1],'*')
            # plt.plot(y.as_matrix()[:,0], y.as_matrix()[:,1],'*')
            # plt.show()

            xV = x.as_vector()
            yV = y.as_vector()
            xx = np.dot(xV, yV)
            yT = Landmarks(yV * (1.0 / xx))
            nb = ut.project(yT.as_vector(),self.mu,self.pcm)
            for i in range(0,len(b)):
                nb[i] = np.clip(nb[i],-self.eW[i],self.eW[i])
            diff = np.sum(abs(b-nb))
            print tsf
            b = nb
            print b
            it+=1
            # plt.plot(x.as_matrix()[:,0],x.as_matrix()[:,1])
            # plt.plot(yT.as_matrix()[:,0],yT.as_matrix()[:,1])
            # plt.show()



        Y = Landmarks(x).T([Tx,Ty],tsf,tAngle)
        return Y.as_matrix().astype('int32')

# Gives the 7 best pca modes
if __name__ == '__main__':
    # load all set of landmarks for first tooth
    folder = '_Data/landmarks/original/'
    nbImgs = 14
    nbDims = 40
    tooth = 1

    asm = ASM(folder, nbImgs, nbDims, tooth, True)

    # plt.clf()
    fig = plt.figure()
    weights = [-3,-2,-1,0,1,2,3]
    for j in range(0,7):
        p = fig.add_subplot(int('24' + str(j+1)))
        for i in range(0,7):
            pts = [0,0,0,0,0,0,0]
            pts[j] = weights[i]
            pts = np.transpose(pts)
            pts = ut.reconstruct(pts,asm.mu,asm.pcm)
            p.plot(pts[:40],pts[40:])
    plt.show()


    pts = [3, 2, 1, 0, 0, 0, 0]
    pts = np.transpose(pts)
    pts = ut.reconstruct(pts, asm.mu, asm.pcm)
    X = Landmarks(pts).as_vector()
    print ut.project(X, asm.mu, asm.pcm)
    print np.dot(asm.pcm.T,asm.pcm)


    tooth1 = Landmarks('_Data/landmarks/original/landmarks2-5.txt')
    tooth2 = Landmarks('_Data/landmarks/original/landmarks2-2.txt')
    X, error = asm.model(tooth1,True)
    X2  = asm.estimate_trans(tooth1.as_matrix())

    print "The error of a matching tooth: ", error
    X, error = asm.model(tooth2,True)
    print "The error of a non-matching tooth: ", error






