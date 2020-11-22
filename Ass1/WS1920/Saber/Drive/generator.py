import os.path
import json
import scipy
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image


class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):


        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}

        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle

        self.q = 0

        self.labels = json.loads(open(self.label_path).read())
        self.dic_len = len(self.labels.keys())
        files_list = os.listdir(self.file_path)
        self.data_set = []


        if self.shuffle:
            np.random.shuffle(files_list)
            name = self.load_files(files_list)
        else:
            name = self.load_files(files_list)
        self.name = name

    def load_files(self, files_list):
        name = []
        for index in files_list:
            self.data_set.append(np.load("/Users/Hagag/PycharmProjects/dl/exercise_data" + "/" + index))
            name.append(str(index[0:-4]))

        return name



    def next(self):

        labels = self.labels
        dic_len = self.dic_len
        data_set = self.data_set
        batch_s = self.batch_size
        name = self.name

        images = []
        labels1 = []

        for _ in range(batch_s):

            if self.q >= dic_len:
                self.q = 0
            current_img = data_set[self.q]
            img = Image.fromarray(current_img)
            resized = img.resize((self.image_size[0], self.image_size[1]),Image.BICUBIC)
            imgarr = np.array(resized)
            images.append(self.augment(imgarr))
            labels1.append(labels.get(name[self.q]))
            self.q += 1
        return (np.array(images), labels1)



        # This function creates a batch of images and corresponding labels and returns it.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        # TODO: implement next method

    def augment(self, img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image

        if self.mirroring:
            img = np.fliplr(img)

        if self.rotation :
            a = np.random.choice([90, 180, 270])
            if a == 90:
                img = np.rot90(img,1)
            elif a == 180:
                img = np.rot90(img,2)
            else:
                img = np.rot90(img,3)

        return img



        # TODO: implement augmentation function

    def class_name(self, x):
        labels = self.class_dict
        name = labels[x]
        return name


    def show(self):
        size = self.batch_size
        images , labels = self.next()

        fig, axs = plt.subplots(size, 1)

        for i in range(size):

            label = labels[i]
            c_name = self.class_name(label)
            img = images[i]

            axs[i].imshow(img)
            axs[i].set_title(c_name)

        plt.show()

        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        # TODO: implement show method



def main():

    imgs_path = "/Users/Hagag/PycharmProjects/dl/exercise_data"
    labels_path = "/Users/Hagag/PycharmProjects/dl/Labels.json"

    test = ImageGenerator(imgs_path, labels_path, 20, [50, 50],False,False,True)
    images, labels = test.next()
    test.class_name(labels[0])
    test.show()

    # test2 = ImageGenerator(imgs_path, labels_path, 4, [20, 20])
    # images, labels = test2.next()
    # test2.class_name(labels[0])
    # test2.show()



main()