import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt


###################################################### Checker Class ######################################################
class Checker:
    def __init__(self, resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size

    def draw(self):
        (Img_H, Img_W) = self.resolution
        (tile_H, tile_W) = self.tile_size
        if Img_H % (2*tile_H) == 0 and Img_W % (2*tile_W) == 0:
            tb = np.zeros((tile_H, tile_W))
            tw = np.ones((tile_H, tile_W))
            chess = np.tile(np.concatenate((np.concatenate((tb, tw), axis=1), np.concatenate((tw, tb), axis=1)) ,axis=0),(Img_H//(2*tile_H),Img_W//(2*tile_W))).astype('float64')
            return chess
        else:
            return print("Error: Resolution must be evenly dividable by 2*tile size in each dimention")

    def show(self):
        plt.imshow(self.draw(), cmap=plt.cm.gray)
        plt.show()

        

###################################################### Circle Class ######################################################
class Circle:
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position

    def draw(self):
        (spec_H, spec_W) = self.resolution
        r = self.radius
        (cx, cy) = self.position
        X = np.tile(np.arange(0,spec_W,1), (spec_H,1))
        Y = np.tile(np.arange(0,spec_H,1), (spec_W,1)).T
        C = lambda x, y: (x-cx)**2 + (y-cy)**2 - r**2
        s = np.sign(C(X,Y))
        s[s == -1] = 0
        Circle = np.invert(s)+2
              
        return Circle

    def show(self):
        plt.imshow(self.draw(), cmap=plt.cm.gray)
        plt.show()
