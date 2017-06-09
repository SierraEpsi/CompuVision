import numpy as np

import cv2
import cv
import math
import image_loader
import ASM
import matplotlib.pyplot as plt

refPt = []

def getMouseCoord(event,x,y,flags,params):
        global mouseX,mouseY, refPt
        if event == cv2.EVENT_LBUTTONDBLCLK:
                mouseX, mouseY = x,y
        elif event == cv2.EVENT_LBUTTONDOWN:
                refPt.append((x,y))



if __name__ == '__main__':

        img = cv2.imread('_Data/Radiographs/01.tif')
       # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        img = image_loader.filterImage(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        #path = '_Data/landmarks/original/landmarks1-1.txt'
        #landmarks = ASM.load_landmarks(path)
        #model = ASM.center_landmarks(landmarks)
        #model = model.astype(np.int32)

        folder = '_Data/landmarks/original/'
        tooth = 2
        nbImgs = 14
        nbDims = 40
        Ev, Ew, mu = ASM.computeModel(folder,tooth,nbImgs,nbDims)
        model = np.zeros((nbDims,2))
        model[:,0] = mu[:nbDims]
        model[:,1] = mu[nbDims:]
        model = np.rint(model).astype(int)


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
                elif k == ord('a'):
                        print mouseX, mouseY
                if mouseX != -1 and mouseY != -1:
                        model_translated = np.copy(model)
                        model_translated[:,0] = model_translated[:,0] + mouseX
                        model_translated[:,1] = model_translated[:,1] + mouseY
                        for i in xrange(len(model_translated) - 1):
                                cv2.line(img, (model_translated[i, 0], model_translated[i, 1]), (model_translated[i + 1, 0], model_translated[i + 1, 1]), [0, 0, 255],2)
                                cv2.imshow('img1', img)
                                k = cv2.waitKey(20) & 0xFF
                        cv2.line(img, (model_translated[-1, 0], model_translated[-1, 1]), (model_translated[0, 0], model_translated[0, 1]), [0, 0, 255],2)
                        cv2.imshow('img1', img)
                        k = cv2.waitKey(20) & 0xFF
                        mouseX = -1
                        mouseY = -1
                if len(refPt) >= 2 :
                        #cv2.rectangle(img,refPt[0],refPt[1],[0,0,255],2)
                        x1 = np.min((refPt[0][1],refPt[1][1]))
                        x2 = np.max((refPt[0][1],refPt[1][1]))
                        y1 = np.min((refPt[0][0],refPt[1][0]))
                        y2 = np.max((refPt[0][0], refPt[1][0]))
                        roi = img[x1:x2,y1:y2,:]
                        if roi.shape[0]>0 and roi.shape[1]>0 and roi.shape[2]>0:
                                roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
                                cv2.imshow('roi', roi)
                                k = cv2.waitKey(20) & 0xFF
                                pnts = np.where(roi>0) #contains all possible landmarks at estimated location
                                plt.plot(pnts[1],pnts[0],'r.')
                                plt.gca().invert_xaxis()
                                plt.gca().invert_yaxis()
                                plt.show()
                        refPt =[]



        print(model)