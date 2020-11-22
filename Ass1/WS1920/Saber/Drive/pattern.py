import numpy as np
import matplotlib.pyplot as plt


class Checker:
    def __init__(self, tile_size, resolution):

        self.tile_size = tile_size
        self.resolution = resolution

    def draw(self):
        if self.tile_size % self.resolution == 0:
            size = self.tile_size
            res = self.resolution
            zeros = np.zeros([res, res], dtype=int)
            ones = np.ones([res, res], dtype=int)
            seq1 = np.concatenate((zeros, ones), axis=0)
            seq2 = np.concatenate((ones, zeros), axis=0)
            unit = np.concatenate((seq1, seq2), axis=1)
            width = size // res
            shape = np.tile(unit, (width // 2, width // 2))

            self.output = shape


            print(unit)
            print(shape)
            plt.imshow(shape)
            # plt.show()
            return shape.copy()

        else:
            print("please enter sensible numbers")


class Circle:
    def __init__(self, reso, radius, center):
        self.reso = reso
        self.center = center
        self.radius = radius

    def draw(self):
        reso = self.reso
        center = self.center
        radius = self.radius

        image = np.zeros([reso, reso], dtype=int)


        y, x = np.ogrid [-radius: radius, -radius: radius]
        index = x ** 2 + y ** 2 <= radius ** 2
        image[center[1] - radius : center[1] + radius, center[0] - radius : center[0] + radius][index] = 1

        self.output = image

        print(image)
        plt.imshow(image, cmap='gray')
        # plt.show()
        return image.copy()


class Spectrum:
    def __init__(self, resolution):
        self.resolution = resolution

    def draw(self):
        res = self.resolution

        g1 = np.arange(0, res)
        g2 = np.ones((1, res))
        g1_nor = g1 / max(g1)
        g2_nor = g2 / max(g2)
        g_channel = np.outer(g1_nor, g2_nor[::-1])


        b1 = np.arange(0, res)
        b2 = np.ones((1, res))
        b1_nor = b1 / max(b1)
        b2_nor = b2 / max(b2)
        b_channel1 = np.outer(b1_nor, b2_nor)
        b_channel = list(zip(*b_channel1[::-1]))

        r1 = np.arange(0, res)
        r2 = np.ones((1, res))
        r1_nor = r1 / max(r1)
        r2_nor = r2 / max(r2)
        r_channel1 = np.outer(r1_nor[::-1], r2_nor)
        r_channel = list(zip(*r_channel1[::-1]))

        test = np.zeros((res, res, 3))

        test[:, :, 0] = r_channel
        test[:, :, 1] = g_channel
        test[:, :, 2] = b_channel
        self.output=test

        print(test)
        plt.imshow(test, 'brg')
        # plt.show()
        return test.copy()
#
# def main():
#
#     test = Checker(8,2)
#     test.draw()
#     test1 = Spectrum(1000)
#     test1.draw()
#
# main()