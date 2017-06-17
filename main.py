from GryMdl import GreyModel
from ASM import ASM
from ACM import ACM
import numpy as np
import cv2
import ImgPP as IPP
from Landmarks import Landmarks as LMS
import MnlSlctn
import matplotlib.pyplot as plt
import Util as ut
import ImgSgm as ISgm
from InsrMdl import InsrModel as IModel


if __name__ == '__main__':

        img = cv2.imread('_Data/Radiographs/01.tif')
        img2 = img.copy()
        G_img = IPP.enhance2(img)
        G_img2 = IPP.GRimg(img)

        img_path = '_Data/Radiographs/'
        lmk_path = '_Data/Landmarks/original/landmarks'
        gModel = GreyModel(img_path, lmk_path, 1, k = 5)
        print 'done 1'

        folder = '_Data/landmarks/original/'
        nbImgs = 14
        nbDims = 40
        tooth = 2
        asm = ASM(folder, nbImgs, nbDims, 1)
        print 'done 2'

        pts = asm.mu
        landmarks = LMS(pts)
        landmarks = landmarks.scale_to_window(asm.mW)
        # pts = landmarks.as_matrix().astype('int32')
        # pts = landmarks.translate(MnlSlctn.init(pts, img)).as_matrix().astype('int32')

        # AUTO
        best_path = ISgm.find_jawline(img)
        iModel = IModel(img_path, lmk_path, True)
        window = iModel.find_window(best_path, G_img)
        poi_u = ISgm.find_POI(G_img, window, True)
        pts = landmarks.translate(poi_u[1]).as_matrix().astype('int32')

        i=0
        maxit = 50
        b_error = float('inf')
        errors = []
        b_pts = -1
        while i < maxit:

                pts, error = gModel.find_points(G_img, pts, 25)
                acm = ACM(-0.01, -0.1, 25, G_img2, pts)
                for i in range(0,10):
                        acm.greedy_step()
                pts = acm.pts
                errors.append(error)

                pimg = pts.reshape(-1, 1, 2)
                img3 = cv2.cvtColor(G_img, cv2.COLOR_GRAY2BGR)
                cv2.polylines(img3, [pimg], True, (0, 0, 256), thickness=5)
                cv2.namedWindow('choose', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('choose', 1200, 800)
                cv2.imshow('choose', img3)
                cv2.waitKey(0)

                pts = asm.estimate_trans(pts)

                if error<b_error:
                        b_error = error
                        b_pts = pts

                pimg = np.reshape(pts,(-1, 1, 2))
                cv2.polylines(img3, [pimg], True, (0,256,0), thickness=5)
                cv2.imshow('choose', img3)
                cv2.waitKey(0)

                i+=1

        plt.plot(range(1,maxit+1),errors)
        plt.show()

        ut.plot_polyLines(img,b_pts)
        ut.plot_polyLines(img, pts)

        print 'done'