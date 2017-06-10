import numpy as np

import cv2
import cv
import math
import ImgPP
from ASM import ASM as ASM
import matplotlib.pyplot as plt

refPt = []

def getMouseCoord(event,x,y,flags,params):
        global mouseX,mouseY, refPt
        if event == cv2.EVENT_LBUTTONDBLCLK:
                mouseX, mouseY = x,y
        elif event == cv2.EVENT_LBUTTONDOWN:
                refPt.append((x,y))

def findClosestPoints(model,pnts):
        pntsC = ASM.center_landmarks(pnts)
        model = ASM.center_landmarks(model)
        target = np.zeros(model.shape)

        for m in xrange(len(model)):
                dif = pntsC - model[m]
                px = np.power(dif[:,0],2)
                py = np.power(dif[:,1],2)
                d = np.sqrt(px + py)

                target [m] = pnts[np.argmin(d).astype(np.int32)]


        plt.plot(target[:,0],target[:,1],'r.')
        plt.plot(model[:,0],model[:,1],'b.')
        plt.show()

        return target

def translate(model,target):
        ctx = np.mean(target[:,0])
        cty = np.mean(target[:,1])
        cmx = np.mean(model[:,0])
        cmy = np.mean(model[:,1])

        plt.plot(model[:,0],model[:,1])
        plt.show()
        model[:,0] = model[:,0] - cmx
        model[:,1] = model[:,1] - cmy

        plt.plot(model[:, 0], model[:, 1])

        model[:,0] = model[:,0] + ctx
        model[:,1] = model[:,1] + cty
        plt.plot(model[:, 0], model[:, 1])

        return model

def rotate(model,target):
        cov = np.dot(target.T, target)
        n = target.shape[0]
        cov = np.divide(cov, n - 1)
        eigW, eigVt = np.linalg.eig(cov)

        cov = np.dot(model.T, model)
        n = model.shape[0]
        cov = np.divide(cov, n - 1)
        eigW, eigVm = np.linalg.eig(cov)

        plt.plot(model[:,0],model[:,1])
        plt.show()

        rotated = np.dot(model, eigVm)
        plt.plot(rotated[:,0],rotated[:,1])
        model = np.dot(rotated, eigVt.T)
        plt.plot(model[:,0],model[:,1])

        return model

if __name__ == '__main__':


        # Img.shape = y x c
        img = cv2.imread('_Data/Radiographs/01.tif')
       # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        img = ImgPP.PPimg(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        #path = '_Data/landmarks/original/landmarks1-1.txt'
        #landmarks = ASM.load_landmarks(path)
        #model = ASM.center_landmarks(landmarks)
        #model = model.astype(np.int32)

        folder = '_Data/landmarks/original/'
        tooth = 1
        nbImgs = 14
        nbDims = 40
        ASM = ASM(folder,nbImgs,nbDims,tooth)

        model = np.zeros((nbDims,2))
        model[:,0] = ASM.mu[:nbDims]
        model[:,1] = ASM.mu[nbDims:]



        cv2.namedWindow('img1',cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('img1', getMouseCoord)
        mouseX = -1
        mouseY = -1

        # Double click places center of model at mouse coords => used model = landmarks
        while (1):
                cv2.imshow('img1', img)
                k = cv2.waitKey(20) & 0xFF
                print refPt
                if k == 27:
                        break

                if mouseX != -1 and mouseY != -1:
                        img1 = np.copy(img)
                        model_translated = np.copy(model)
                        model_translated = np.rint(model_translated).astype(int)
                        model_translated[:,0] = model_translated[:,0] + mouseX
                        model_translated[:,1] = model_translated[:,1] + mouseY
                        for i in xrange(len(model_translated) - 1):
                                cv2.line(img1, (model_translated[i, 0], model_translated[i, 1]), (model_translated[i + 1, 0], model_translated[i + 1, 1]), [0, 0, 255],2)
                                cv2.imshow('img1', img1)
                                k = cv2.waitKey(20) & 0xFF
                        cv2.line(img1, (model_translated[-1, 0], model_translated[-1, 1]), (model_translated[0, 0], model_translated[0, 1]), [0, 0, 255],2)
                        cv2.imshow('img1', img1)
                        k = cv2.waitKey(2000) & 0xFF
                        mouseX = -1
                        mouseY = -1
                        refPt = []
                if len(refPt) >= 2 :
                        x1 = np.min((refPt[0][0],refPt[1][0]))
                        x2 = np.max((refPt[0][0],refPt[1][0]))
                        y1 = np.min((refPt[0][1],refPt[1][1]))
                        y2 = np.max((refPt[0][1], refPt[1][1]))
                        cx = x1 + np.rint((x2-x1)/2)
                        cx = cx.astype(int)
                        cy = y1 + np.rint((y2-y1)/2)
                        cy = cy.astype(int)
                        roi = img[y1:y2,x1:x2,:] # img y,x,c
                        if roi.shape[0]>0 and roi.shape[1]>0 and roi.shape[2]>0:
                                roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
                                cv2.imshow('roi', roi)
                                k = cv2.waitKey(20) & 0xFF
                                pntsYX = np.where(roi>0) #contains all possible landmarks at estimated location
                                pntsYX = np.asarray(pntsYX).T
                                pnts = np.zeros(pntsYX.shape,int)
                                pnts[:,0] = pntsYX[:,1]
                                pnts[:,1] = pntsYX[:,0]

                                plt.plot(pnts[:,0],pnts[:,1],'r.')
                                #plt.gca().invert_xaxis()
                                plt.gca().invert_yaxis()
                                plt.show()
                        refPt =[]

        for i in xrange(10):
                target = findClosestPoints(model,pnts)
                target = target.astype(int)
                
                target[:,0] = target[:,0] + x1
                target[:,1] = target[:,1] + y1
                for i in xrange(len(target) - 1):
                        cv2.line(img, (target[i, 0], target[i, 1]),
                                 (target[i + 1, 0], target[i + 1, 1]), [0, 0, 255], 2)
                        cv2.imshow('img1', img)
                        k = cv2.waitKey(20) & 0xFF
                cv2.line(img, (target[-1, 0], target[-1, 1]),
                         (target[0, 0], target[0, 1]), [0, 0, 255], 2)
                cv2.imshow('img1', img)
                k = cv2.waitKey(20) & 0xFF

                model = translate(model,target)
                model = rotate(model,target)
                model = model.astype(int)
                for i in xrange(len(model) - 1):
                        cv2.line(img, (model[i, 0], model[i, 1]),
                                 (model[i + 1, 0], model[i + 1, 1]), [0, 255, 0], 2)
                        cv2.imshow('img1', img)
                        k = cv2.waitKey(20) & 0xFF
                cv2.line(img, (model[-1, 0], model[-1, 1]),
                         (model[0, 0], model[0, 1]), [0, 255,0], 2)
                cv2.imshow('img1', img)
                k = cv2.waitKey(20) & 0xFF

        print(model)