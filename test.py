import ASM
import cv
import cv2
import matplotlib.pyplot as plt
import numpy as np


print "Hello Mat"
print "test1"
print "gecloned"
print "test 2"

path = '_Data/landmarks/original/landmarks1-1.txt'
path2 = '_Data/landmarks/original/landmarks2-1.txt'


landmarks = ASM.load_landmarks(path)
translated = ASM.center_landmarks(landmarks)
rotated = ASM.rotate_shape(translated)
#scaled = ASM.scale_estimate(rotated)
x = translated[:, 0]
y = translated[:, 1]
plt.plot(x, y, '*')
plt.plot(x, y)

landmarks2 = ASM.load_landmarks(path2)
translated2 = ASM.center_landmarks(landmarks2)



cov = np.dot(translated.T,translated)
n = landmarks.shape[0]
cov = np.divide(cov,n-1)
eigW, eigV = np.linalg.eig(cov)
eigV = eigV/np.linalg.norm(eigV)
rotated = np.dot(translated,eigV)
x = rotated[:, 0]
y = rotated[:, 1]
plt.plot(x, y, '*')
plt.plot(x, y)


rotated2 = np.dot(translated2,eigV)
x = rotated2[:, 0]
y = rotated2[:, 1]
plt.plot(x, y, '*')
plt.plot(x, y)

plt.show()


image = np.zeros((20,20))
rect = [(1,1),(1,5),(5,5),(5,1)]
for i in xrange(len(rect)-1):
        cv2.line(image,rect[i],rect[i+1],255)
cv2.namedWindow('img2', cv2.WINDOW_NORMAL)
cv2.imshow('img2', image)
cv2.waitKey(0)
