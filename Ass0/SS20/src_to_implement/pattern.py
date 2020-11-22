import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt


###################################################### Checkers Class ######################################################
class Checker:
    def __init__(self, resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = self.draw()

    def draw(self):
        Img_H = self.resolution
        tile_H = self.tile_size
        if Img_H % (2*tile_H) == 0:
            tb = np.zeros((tile_H, tile_H))
            tw = np.ones((tile_H, tile_H))
            self.output = np.tile(np.concatenate((np.concatenate((tb, tw), axis=1), np.concatenate((tw, tb), axis=1)) ,axis=0),(Img_H//(2*tile_H),Img_H//(2*tile_H))).astype('float64')
            return np.copy(self.output)
        else:
            return print("Error: Resolution must be divisible by by 2*tile_size")

    def show(self):
        if self.resolution % (2*self.tile_size) == 0:
            plt.imshow(self.output, cmap=plt.cm.gray)
            plt.show()
        else:
            return print("Error: Resolution must be divisible by by 2*tile_size")
        

###################################################### Circle Class ######################################################
class Circle:
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = self.draw()

    def draw(self):
        spec_H = self.resolution
        r = self.radius
        (cx, cy) = self.position
        X = np.tile(np.arange(0,spec_H,1), (spec_H,1))
        Y = np.tile(np.arange(0,spec_H,1), (spec_H,1)).T
        C = lambda x, y: (x-cx)**2 + (y-cy)**2 - r**2
        s = np.sign(C(X,Y))
        s[s == -1] = 0
        self.output = np.invert(s)+2
              
        return np.copy(self.output)

    def show(self):
        plt.imshow(self.output, cmap=plt.cm.gray)
        plt.show()
        
        
        
        
###################################################### Spectrum Class ######################################################

class Spectrum:
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = self.draw()

    def draw(self):
        spec_H = self.resolution
        rgbArray = np.zeros((spec_H,spec_H,3))
        rgbArray[:,:,0] = np.tile(np.linspace(0, 1, spec_H), (spec_H,1))
        rgbArray[:,:,1] = np.tile(np.linspace(0, 1, spec_H), (spec_H,1)).T
        rgbArray[:,:,2] = np.flip(rgbArray[:,:,0])
        self.output = rgbArray
              
        return np.copy(self.output)

    def show(self):
        plt.imshow(self.output)
        plt.show()
