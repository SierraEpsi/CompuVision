import numpy as np
import matplotlib.pyplot as plt
import cv2
import cv

global fig
global p1
global p2
global p3
global p4

def load_landmarks(path):
    f = open(path, 'r')
    landmarks = np.loadtxt(f)
    landmarks = np.reshape(landmarks, (landmarks.size / 2, 2))  # shape (40,2)
    landmarks = trans_rot(landmarks)
    return landmarks

def center_landmarks(landmarks):
    x = landmarks[:, 0]
    y = landmarks[:, 1]
    x = np.subtract(x,np.mean(x))
    y = np.subtract(y,np.mean(y))
    p1.plot(x, y, '*')
    p1.plot(x, y)
    return x, y

def scale_estimate(landmarks):
    x = landmarks[0]
    y = landmarks[1]
    scale_factor = np.sqrt(np.sum(np.power(x,2)) + np.sum(np.power(y,2)))
    x = np.divide(x,scale_factor)
    y = np.divide(y,scale_factor)
    p2.plot(x, y, '*')
    p2.plot(x, y)
    return x, y

def rotate_shape(landmarks):
    cov = np.dot(landmarks,np.transpose(landmarks))
    n = landmarks[0].size
    cov = np.divide(cov,n-1)
    eigW, eigV = np.linalg.eig(cov)
    eigV = eigV/np.linalg.norm(eigV)
    landmarks = np.dot(eigV,landmarks)
    return landmarks

def trans_rot(landmarks):
    landmarks = center_landmarks(landmarks)
    landmarks = scale_estimate(landmarks)
    landmarks = rotate_shape(landmarks)
    return landmarks

if __name__ == '__main__':

    fig = plt.figure()
    p1 = fig.add_subplot(131)
    p2 = fig.add_subplot(132)
    p3 = fig.add_subplot(133)
    #load all set of landmarks for first tooth
    folder = '_Data/landmarks/original/'
    for i in range(1,15):
        path = folder + 'landmarks' + str(i) + '-1.txt'
        landmarks = load_landmarks(path)
        x = landmarks[0]
        y = landmarks[1]
        p3.plot(x, y, '*')
        p3.plot(x, y)
    plt.show()
    cv2.waitKey(0)

