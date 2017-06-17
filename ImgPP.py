
import cv2
from scipy.ndimage import morphology

def enhance(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img,7)
    img = cv2.bilateralFilter(img, 9, 500, 500)
    return img;

def enhance2(img):
    img = enhance(img)
    imgW = morphology.white_tophat(img, size=500)
    imgB = morphology.black_tophat(img, size=100)
    imgH = cv2.subtract(imgW, imgB)
    img = cv2.add(img, imgH)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16))
    img = clahe.apply(img)
    return img

def jaw_enhance(img):
    img = enhance(img)
    img = morphology.white_tophat(img, size=500)
    img = 255 - img
    img = cv2.GaussianBlur(img, (111, 25), 5)
    return img

def GRimg(img):
    img = enhance2(img)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(sobelx)
    abs_grad_y = cv2.convertScaleAbs(sobely)
    img = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(25, 25))
    img = clahe.apply(img)
    img = cv2.threshold(img, 75, 256, cv2.THRESH_TOZERO)[1]
    return img

if __name__ == '__main__':
    # read an image
    img = cv2.imread('_Data/Radiographs/05.tif')
    img = GRimg(img)

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img', 1500, 1000)
    cv2.imshow('img', img)
    cv2.waitKey(0)
