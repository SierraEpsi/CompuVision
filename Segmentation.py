import numpy as np
import cv
import cv2
import math
import image_loader
import ASM
import matplotlib.pyplot as plt

if __name__ == '__main__':

        img = cv2.imread('_Data/Radiographs/01.tif')

        edges = image_loader.filterImage(img)

        # load all set of landmarks for first tooth
        folder = '_Data/landmarks/original/'
        nbImgs = 14
        nbDims = 40
        tooth = 1

        # load landmarks of tooth 1, case 1
        landmarks = ASM.load_landmarks(folder + 'landmarks1-1.txt') #removed trans/rot from load: computed during computeModel
        landmarks = np.reshape(landmarks,(1,nbDims*2))[0] # Waarom????


        eV,eW,mu = ASM.computeModel(folder,tooth,nbImgs,nbDims)

        landmarksProjected = ASM.project(landmarks,eV,mu) # scale first???
        landmarksReconstruct = ASM.reconstruct(landmarksProjected,eV,mu) # rescale to Original... or rescale model first???

        model = landmarksReconstruct
        plt.plot(model[0:nbDims], model[nbDims:])
        plt.show()

        img = np.copy(edges)
        img = cv2.polylines(img,model,True,(0,0,255))


        cv2.namedWindow('img1',cv2.WINDOW_NORMAL)
        cv2.imshow('img1',img)
        cv2.waitKey(0)


        print(model)