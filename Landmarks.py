import os
import re
import fnmatch
import numpy as np

class Landmarks(object):
    def __init__(self, source):
        self.pts = -1
        if isinstance(source, str):
            self.load_landmarks(source)
        elif isinstance(source, np.ndarray) and len(source.shape) == 2:
            self.pts = source
        else:
            self.pts = np.array((source[:len(source) / 2], source[len(source) / 2:])).T
            
    def load_landmarks(self, path):
        f = open(path, 'r')
        landmarks = np.loadtxt(f)
        landmarks = np.reshape(landmarks, (landmarks.size / 2, 2))  # shape (40,2)
        self.pts = landmarks

    def as_vector(self):
        return np.hstack((self.pts[:, 0], self.pts[:, 1]))

    def as_matrix(self):
        return self.pts

    def get_centroid(self):
        return np.mean(self.pts, axis=0)

    def get_center(self):
        ma = np.max(self.pts,0)
        mi = np.max(self.pts,0)
        center = np.add(ma,mi)
        center = np.divide(center,2)
        return center

    def translate_to_origin(self):
        center = self.get_centroid()
        pts = self.pts - center
        return Landmarks(pts)

    def translate_center_to_origin(self):
        center = self.get_center()
        pts = self.pts - center
        return Landmarks(pts)

    def scale_to_unit(self):
        centroid = self.get_centroid()
        scale_factor = np.sqrt(np.power(self.pts - centroid, 2).sum())
        pts = self.pts.dot(1. / scale_factor)
        return Landmarks(pts)

    def translate(self, vec):
        pts = self.pts + vec
        return Landmarks(pts)

    def scale(self, factor):
        centroid = self.get_centroid()
        pts = (self.pts - centroid).dot(factor) + centroid
        return Landmarks(pts)

    def rotate(self, angle):
        rotmat = np.array([[np.cos(angle), np.sin(angle)],
                           [-np.sin(angle), np.cos(angle)]])

        pts = np.zeros_like(self.pts)
        centroid = self.get_centroid()
        tmp_pts = self.pts - centroid
        for ind in range(len(tmp_pts)):
            pts[ind, :] = tmp_pts[ind, :].dot(rotmat)
        pts = pts + centroid

        return Landmarks(pts)

    def T(self, t, s, theta):
        return self.rotate(theta).scale(s).translate(t)

    def invT(self, t, s, theta):
        t = np.multiply(t,-1)
        return self.translate(t).scale(1/s).rotate(-theta)

    def get_dimensions(self):
        h = self.pts[:, 1].max() - self.pts[:, 1].min()
        w = self.pts[:, 0].max() - self.pts[:, 0].min()
        return (w,h)

    def scale_to_window(self, window):
        h = window[1]
        sf = h / (self.pts[:, 1].max() - self.pts[:, 1].min())
        return self.scale(sf)