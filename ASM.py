import numpy as np
import matplotlib.pyplot as plt
import cv2
import cv

if __name__ == '__main__':

    #load 1 set of landmarks
    f = open('_Data/landmarks/original/landmarks1-1.txt','r')
    landmarks = np.loadtxt(f)
    landmarks = np.reshape(landmarks,(landmarks.size/2,2)) #shape (40,2)

    #2D plot.
    x = landmarks[:, 0]
    y = landmarks[:, 1]
    plt.plot(x,y,'*')
    cv2.waitKey(0)