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

        nbImgs = 13
        nbDims = 40
        tr_img_path = '_Data/Radiographs/'
        tr_lmk_path = '_Data/Landmarks/original/'
        tst_seg_path = '_Data/Segmentations/'
        tst_img_path = '_Data/Radiographs/'


        for leaveOneOut in range(1,nbImgs+2):
##################### BUILD THE MODELS ##################        

                iModelU = IModel(tr_img_path, tr_lmk_path, True, nr=nbImgs, leaveOneOut=leaveOneOut)

                iModelL = IModel(tr_img_path, tr_lmk_path, False, nr=nbImgs, leaveOneOut=leaveOneOut)
                print 'incisor models done'
                # PreProc
                iS = str(leaveOneOut)
                if leaveOneOut < 10:
                        iS = '0' + iS
                img = cv2.imread(tst_img_path + iS + '.tif')
                img2 = img.copy()
                G_img = IPP.enhance2(img)
                G_img2 = IPP.GRimg(img)

                # AUTO: Find jawLine
                best_path = ISgm.find_jawline(img)

                if True:  # doPlot
                        img3 = G_img.copy()
                        pts = []
                        for i in range(0, len(best_path[2])):
                                point = (int((best_path[0] + i) * best_path[1]), int(best_path[2][i]))
                                pts.append(point)

                        img3 = cv2.cvtColor(G_img, cv2.COLOR_GRAY2BGR)
                        path = '_Data/jaw/LOO' + str(leaveOneOut) + '.tif'
                        ut.plot_polyLines(img3, pts, 'jaw', path)


                windowU = iModelU.find_window(best_path, G_img)
                poi_u = ISgm.find_POI(G_img2, windowU, True)

                windowL = iModelL.find_window(best_path, G_img)
                poi_l = ISgm.find_POI(G_img2, windowL, False)


################### BUILD TOOTH MODEL
                for tooth in range(1, 9):

                        gModel = GreyModel(tr_img_path, tr_lmk_path, tooth, k=12, nr=nbImgs, leaveOneOut=leaveOneOut)
                        print 'Grey Models done '

                        asm = ASM(tr_lmk_path, nbImgs, nbDims, tooth, leaveOneOut= leaveOneOut)
                        print 'ASM models done'

                        pts = asm.mu
                        landmarks = LMS(pts)
                        landmarks = landmarks.scale_to_window(asm.mW)
                        # pts = landmarks.as_matrix().astype('int32')
                        # pts = landmarks.translate(MnlSlctn.init(pts, img)).as_matrix().astype('int32')


################# FIT ############################

                        if tooth <= 4:
                                poi = landmarks.translate(poi_u[tooth - 1]).as_matrix().astype('int32')
                        else:
                                poi = landmarks.translate(poi_l[tooth - 5]).as_matrix().astype('int32')

                        i=0
                        maxit = 50
                        b_error = float('inf')
                        errors = []
                        b_pts = -1
                        pts = poi
                        while i < maxit:

                                pts, error = gModel.find_points(G_img, pts, 25)
                                # acm = ACM(-0.01, -0.1, 25, G_img2, pts)
                                # for i in range(0,10):
                                #         acm.greedy_step()
                                # pts = acm.pts
                                errors.append(error)

                                # pimg = pts.reshape(-1, 1, 2)
                                # img3 = cv2.cvtColor(G_img, cv2.COLOR_GRAY2BGR)
                                # cv2.polylines(img3, [pimg], True, (0, 0, 256), thickness=5)
                                # cv2.namedWindow('choose', cv2.WINDOW_NORMAL)
                                # cv2.resizeWindow('choose', 1200, 800)
                                # cv2.imshow('choose', img3)
                                # cv2.waitKey(0)

                                pts = asm.estimate_trans(pts)

                                if error<b_error:
                                        b_error = error
                                        b_pts = pts

                                # pimg = np.reshape(pts,(-1, 1, 2))
                                # cv2.polylines(img3, [pimg], True, (0,256,0), thickness=5)
                                # cv2.imshow('choose', img3)
                                # cv2.waitKey(0)

                                i+=1



                        path = '_Data/errors/LOO' + str(leaveOneOut) + '-' + str(tooth) + '.png'
                        fig1 = plt.figure()
                        plt.plot(range(1,maxit+1),errors)
                        fig1.savefig(path)
                        plt.close()


                        path = '_Data/fit/bestLOO' + str(leaveOneOut) + '-' + str(tooth) + '.tif'
                        ut.plot_polyLines(img,b_pts,'best' , path)
                        path = '_Data/fit/lastLOO' + str(leaveOneOut) + '-' + str(tooth) + '.tif'
                        ut.plot_polyLines(img, pts,'last',path)



                        # compare to GT:
                        gt_img = cv2.imread(tst_seg_path + iS + '-' + str(tooth-1) + '.png')
                        seg_img = np.zeros(gt_img.shape,dtype='uint8')


                        path = '_Data/Predictions/GTLOO' + str(leaveOneOut) + '-' + str(tooth) + '.png'
                        ut.plot_polyLines(gt_img,b_pts,'best',path)

                        # gt_img = cv2.cvtColor(gt_img,cv2.COLOR_BGR2GRAY)
                        # _,gt_img = cv2.threshold(gt_img,1,100,cv2.THRESH_BINARY)

                        filename = '_Data/Prediction/segLOO' + str(leaveOneOut) + '-' + str(tooth) + '.png'
                        b_pts = np.reshape(b_pts, (-1, 1, 2))
                        cv2.fillPoly(seg_img,[b_pts],(255,255,255),8)
                        ut.show_image(seg_img,'seg',filename)
                        seg_img = cv2.cvtColor(seg_img,cv2.COLOR_BGR2GRAY)
                        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
                        _, gt_img = cv2.threshold(gt_img, 1, 1, cv2.THRESH_BINARY)
                        _, seg_img = cv2.threshold(seg_img, 1, 1, cv2.THRESH_BINARY)

                        # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
                        TP = np.sum(np.logical_and(seg_img == 1, gt_img == 1)).astype(np.float32)
                        # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
                        TN = np.sum(np.logical_and(seg_img == 0, gt_img == 0)).astype(np.float32)
                        # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
                        FP = np.sum(np.logical_and(seg_img == 1, gt_img == 0)).astype(np.float32)
                        # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
                        FN = np.sum(np.logical_and(seg_img == 0, gt_img == 1)).astype(np.float32)

                        Dice = 2*TP/(2*TP + FN + FP)

                        file = open('_Data/ValidationResults.txt','w')
                        file.write('Validation using leave one out:\n')
                        file.write('leave one out = img' + str(iS)+ '\n')
                        file.write('tooth nbr =' + str(tooth)+ '\n')
                        file.write('the amount of True positives:' + str(TP)+ '\n')
                        file.write('the amount of True negatives:' + str(TN)+ '\n')
                        file.write('the amount of False positives:' + str(FP)+ '\n')
                        file.write('the amount of False negatives:' + str(FN)+ '\n')
                        file.write('the Dice score:' + str(Dice)+ '\n')

##################### BUILD THE MODELS ##################
        leaveOneOut =-1
        tst_img_path = '_Data/Radiographs/extra'

        iModelU = IModel(tr_img_path, tr_lmk_path, True, nr=nbImgs, leaveOneOut=leaveOneOut)

        iModelL = IModel(tr_img_path, tr_lmk_path, False, nr=nbImgs, leaveOneOut=leaveOneOut)
        print 'incisor models done'

        for tst_img in range(15, 31):
                # PreProc
                iS = str(tst_img)
                if tst_img < 10:
                        iS = '0' + iS
                img = cv2.imread(tst_img_path + iS + '.tif')
                img2 = img.copy()
                G_img = IPP.enhance2(img)
                G_img2 = IPP.GRimg(img)

                # AUTO: Find jawLine
                best_path = ISgm.find_jawline(img)

                if True:  # doPlot
                        img3 = G_img.copy()
                        pts = []
                        for i in range(0, len(best_path[2])):
                                point = (int((best_path[0] + i) * best_path[1]), int(best_path[2][i]))
                                pts.append(point)

                        img3 = cv2.cvtColor(G_img, cv2.COLOR_GRAY2BGR)
                        path = '_Data/jaw/LOO' + str(tst_img) + '.tif'
                        ut.plot_polyLines(img3, pts, 'jaw', path)

                windowU = iModelU.find_window(best_path, G_img)
                poi_u = ISgm.find_POI(G_img2, windowU, True)

                windowL = iModelL.find_window(best_path, G_img)
                poi_l = ISgm.find_POI(G_img2, windowL, False)
                ################### BUILD TOOTH MODEL
                for tooth in range(1, 9):
                        gModel = GreyModel(tr_img_path, tr_lmk_path, tooth, k=12, nr=nbImgs,
                                           tst_img=tst_img)
                        print 'Grey Models done '

                        asm = ASM(tr_lmk_path, nbImgs, nbDims, tooth, tst_img=tst_img)
                        print 'ASM models done'

                        pts = asm.mu
                        landmarks = LMS(pts)
                        landmarks = landmarks.scale_to_window(asm.mW)
                        # pts = landmarks.as_matrix().astype('int32')
                        # pts = landmarks.translate(MnlSlctn.init(pts, img)).as_matrix().astype('int32')





                        ################# FIT ############################

                        if tooth <= 4:
                                poi = landmarks.translate(poi_u[tooth - 1]).as_matrix().astype('int32')
                        else:
                                poi = landmarks.translate(poi_l[tooth - 5]).as_matrix().astype('int32')

                        i = 0
                        maxit = 50
                        b_error = float('inf')
                        errors = []
                        b_pts = -1
                        pts = poi
                        while i < maxit:

                                pts, error = gModel.find_points(G_img, pts, 25)
                                # acm = ACM(-0.01, -0.1, 25, G_img2, pts)
                                # for i in range(0,10):
                                #         acm.greedy_step()
                                # pts = acm.pts
                                errors.append(error)

                                # pimg = pts.reshape(-1, 1, 2)
                                # img3 = cv2.cvtColor(G_img, cv2.COLOR_GRAY2BGR)
                                # cv2.polylines(img3, [pimg], True, (0, 0, 256), thickness=5)
                                # cv2.namedWindow('choose', cv2.WINDOW_NORMAL)
                                # cv2.resizeWindow('choose', 1200, 800)
                                # cv2.imshow('choose', img3)
                                # cv2.waitKey(0)

                                pts = asm.estimate_trans(pts)

                                if error < b_error:
                                        b_error = error
                                        b_pts = pts

                                # pimg = np.reshape(pts,(-1, 1, 2))
                                # cv2.polylines(img3, [pimg], True, (0,256,0), thickness=5)
                                # cv2.imshow('choose', img3)
                                # cv2.waitKey(0)

                                i += 1

                        path = '_Data/errors/LOO' + str(tst_img) + '-' + str(tooth) + '.png'
                        fig1 = plt.figure()
                        plt.plot(range(1, maxit + 1), errors)
                        fig1.savefig(path)
                        plt.close()

                        path = '_Data/fit/bestLOO' + str(tst_img) + '-' + str(tooth) + '.tif'
                        ut.plot_polyLines(img, b_pts, 'best', path)
                        path = '_Data/fit/lastLOO' + str(tst_img) + '-' + str(tooth) + '.tif'
                        ut.plot_polyLines(img, pts, 'last', path)

        file.write('done', + '\n')