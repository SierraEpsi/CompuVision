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
        energy += self.beta * self.compute_curvature(pts)
        return energy

    def compute_out(self, pts):
        Eo = 0
        for point in pts:
            Eo -= self.c_out(point)
        return int(Eo)

    def c_out(self,pt):
        g_w = [[1,4,1],
               [4,7,4],
               [1,4,1]]
        img_window = self.img[pt[1]-1:pt[1]+2,pt[0]-1:pt[0]+2].copy()
        temp = np.multiply(img_window,g_w)
        temp = np.sum(temp)
        temp = temp/27.0
        temp = np.power(temp,2)
        return temp


    def compute_elasticity(self, pts):
        Ee = self.c_elas(pts[-1],pts[0])
        for i in range(0, len(pts) - 1):
            Ee += self.c_elas(pts[i],pts[i+1])
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

    def compare_diff(self, i, nP):
        if i == len(self.pts)-1:
            i = -1
        elif i == len(self.pts)-2:
            i = -2

        Eo = self.c_out(self.pts[i])
        Eo -= self.c_out(nP)

        Ee = self.c_elas(self.pts[i-1],self.pts[i])
        Ee += self.c_elas(self.pts[i],self.pts[i+1])
        Ee -= self.c_elas(self.pts[i-1],nP)
        Ee -= self.c_elas(nP,self.pts[i+1])

        Ec = self.c_curv(self.pts[i-2],self.pts[i-1],self.pts[i])
        Ec += self.c_curv(self.pts[i-1],self.pts[i],self.pts[i+1])
        Ec += self.c_curv(self.pts[i],self.pts[i+1],self.pts[i+2])
        Ec -= self.c_curv(self.pts[i-2],self.pts[i-1],nP)
        Ec -= self.c_curv(self.pts[i-1],nP,self.pts[i+1])
        Ec -= self.c_curv(nP,self.pts[i+1],self.pts[i+2])
        return self.gamma*Eo - self.alpha*Ee - self.beta*Ec

    def greedy_search(self, i, dis=2):
        best_gain = 0
        best_point = -1
        point = self.pts[i]
        for x in range(point[0]-dis,point[0]+dis+1):
            for y in range(point[1]-dis,point[1]+dis+1):
                if x != point[0] and y != point[0]:
                    np = (x,y)
                    gain = -self.compare_diff(i, np)
                    if gain < best_gain:
                        best_gain = gain
                        best_point = np
        if best_point != -1:
            self.pts[i] = best_point
            self.Et += gain
            self.compute_avg_d()
        return best_gain

    def greedy_step(self, dis=2):
        order = random.sample(range(0,len(self.pts)), len(self.pts))
        gain = 0
        for i in range(0,len(self.pts)):
            gain += self.greedy_search(order[i],dis)
        return gain
