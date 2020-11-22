from pattern import Checker, Circle, Spectrum
from generator import ImageGenerator

#####################################################################################
l = 100
t = 10
checker_test = Checker(l,t)

checker_test.draw()

checker_test.show()



res = 200
r = 70
center = (150, 125)
C = Circle(res, r, center)

C.draw()

C.show()



CS = Spectrum(200)

CS.draw()

CS.show()




img_gen = ImageGenerator(file_path = 'C:/Users/utg_1/OneDrive/Documents/DL EX/Ex0 data/exercise_data/',
                         label_path = 'src_to_implement/Labels.json',
                         batch_size = 20,
                         image_size = [32, 32], #h, W,
                         rotation = True,
                         mirroring = True,
                         shuffle = False)

img_gen.show()