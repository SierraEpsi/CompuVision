import numpy as np
import random

class ACM:
    def __init__(self, alpha, beta, gamma, img, pts):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.img = img.copy()
        self.pts = pts
        self.avg_dis = -1
        self.compute_avg_d()
        self.Et = self.compute_energy()

    def compute_energy(self, pts=None):
        if pts == None:
            pts = self.pts
        energy = self.gamma * self.compute_out(pts)
        energy += self.alpha * self.compute_elasticity(pts)
        energy += self.alpha * self.compute_curvature(pts)
        return energy

    def compute_out(self, pts):
        Eo = 0
        for point in pts:
            Eo -= self.img[point[1], point[0]]
        print Eo
        return int(Eo)

    def compute_elasticity(self, pts):
        Ee = self.c_elas(pts[-1],pts[0])
        for i in range(0, len(pts) - 1):
            Ee += self.c_elas(pts[i],pts[i+1])
        Ee = np.sqrt(Ee)
        return int(Ee)

    def c_elas(self, pt0, pt1):
        temp = np.subtract(pt1, pt0)
        temp = np.power(temp,2)
        temp = np.sum(temp)
        temp = np.subtract(temp, self.avg_dis)
        temp = np.power(temp,2)
        return temp

    def compute_curvature(self, pts):
        Ec = self.c_curv(pts[-2],pts[-1],pts[0])
        Ec += self.c_curv(pts[-1],pts[0],pts[1])
        for i in range(1,len(pts)-1):
            Ec += self.c_curv(pts[i-1],pts[i],pts[i+1])
        Ec = np.sqrt(Ec)
        return int(Ec)

    def c_curv(self, pt0, pt1, pt2):
        temp = np.add(pt0,pt2)
        temp = np.add(temp,np.multiply(pt1,-2))
        temp = np.power(temp,2)
        temp = np.sum(temp)
        return temp

    def compute_avg_d(self):
        pts = self.pts
        dis = self.distance(pts[-1],pts[0])
        for i in range(0,len(pts)-1):
            dis += self.distance(pts[i], pts[i+1])
        dis = dis/len(pts)
        self.avg_dis = dis

    def distance(self, pt0, pt1):
        temp = np.subtract(pt0,pt1)
        temp = np.power(temp,2)
        temp = np.sum(temp)
        temp = np.sqrt(temp)
        return temp

    def compare_diff(self, i, np):
        pts = self.pts

        n_pts = pts
        n_pts[i] = np
        nEt = self.compute_energy(n_pts)
        return nEt - self.Et

    def greedy_search(self, i, dis=2):
        best_gain = 0
        best_point = -1
        point = self.pts[i]
        for x in range(point[0]-dis,point[0]+dis+1):
            for y in range(point[1]-dis,point[1]+dis+1):
                if x != point[0] and y != point[0]:
                    np = (x,y)
                    gain = self.compare_diff(i, np)
                    if gain < best_gain:
                        best_gain = gain
                        best_point = np
        if best_point != -1:
            self.pts[i] = best_point
            self.Et -= gain
            self.compute_avg_d()

    def greedy_step(self):
        ri = random.randint(0,len(self.pts)-1)
        self.greedy_search(ri,4)