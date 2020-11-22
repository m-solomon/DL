import os.path
import json
import scipy.misc
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
from random import randrange

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.
        #TODO: implement constructor
        
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        self.counter = 0

    def next(self):
        # This function creates a batch of images and corresponding labels and returns it.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        #TODO: implement next method
        
        """
        if shuffle is false:
            the first time next() is called: show the first batch
            the second time next() is called:: show second batch
            ...
        else if shuffle is true:
            shuffle the original dictionary
            the first time next() is called: show the first batch of the shuffled list
            the second time next() is called: show the second batch of the same shuffled list
        
        1) get the json dictionary (self.label_path)
        2) check for shuffling (self.shuffle)
        3) creat the batches (self.batch_size) and add to the last batch if it was smaller than the requested size (creat an array of arrays of image_names)
        3*) creat the a crresponding list of arrays of image lables
        4) load images for a certain batch based on the results from (3) and (self.file_path) and on the rank of the call
        4*) and check for size then resize(self.image_size)
        5) check for mirroring (self.mirroring)
        6) check for rotation (self.rotation)
        7) save images in array and save it in the output variable (batch)
        8) save corresponding lables in output variable (lables)
        9) increment counter
        
        
        """
        
        #step 1 and 2: get the json dictionary (self.label_path) &&& check for shuffling (self.shuffle)
        with open(self.label_path) as labeled:
            if self.shuffle == False:
                all_labeled_image_names = json.load(labeled)

            else:
                shuffled_list = list(json.load(labeled).items())
                random.shuffle(shuffled_list)
                all_labeled_image_names = dict( shuffled_list )

        
        #step 3: creat the batches & creat the a crresponding lables list
        all_image_names_array = np.asarray(list(all_labeled_image_names.keys()))
        all_image_labels_array = np.asarray(list(all_labeled_image_names.values()))
        remainder = len(all_image_names_array) % self.batch_size
        if remainder != 0:
            updated_image_names_array = np.concatenate((all_image_names_array, all_image_names_array[0:self.batch_size-remainder]), axis=None)
            updated_image_labels_array = np.concatenate((all_image_labels_array, all_image_labels_array[0:self.batch_size-remainder]), axis=None)
        else:
            updated_image_names_array = all_image_names_array
            updated_image_labels_array = all_image_labels_array

        batched_image_names_matrix = np.reshape(updated_image_names_array, (-1, self.batch_size))
        batched_image_labels_matrix = np.reshape(updated_image_labels_array, (-1, self.batch_size))
        
        
        
        #step 4: load images for a single batch based on the results from (3) and (self.file_path) and on the rank of the call (creat a counter variable)
        #step 4*: and check for size then resize  (self.image_size)


        #preparing files addresses
        files_path = np.asarray([self.file_path]*self.batch_size)
        current_batch = batched_image_names_matrix[self.counter]
        file_format = np.asarray(['.npy']*self.batch_size)
        filelist = tuple(np.array(files_path,dtype=object)+np.array(current_batch,dtype=object)+np.array(file_format,dtype=object))

        #loading files to a list
        single_batch_of_images_list = [np.load(image) for image in tuple(filelist)]

        #check for size and resize, then put images in 4d array
        single_batch_of_resized_images_4Darray = np.zeros((self.batch_size, self.image_size[0], self.image_size[1], 3)) #creat empty array for resized images
        if len(np.unique(np.asarray([imag.shape for imag in single_batch_of_images_list]), axis=0)) == 1: #if all images have the same size
            ListTo4DArray = np.asarray(single_batch_of_images_list)                                       #then put them all in a 4d array
            single_batch_of_resized_images_4Darray = scipy.ndimage.zoom(ListTo4DArray, zoom = (1,self.image_size[0]/ListTo4DArray.shape[1],self.image_size[1]/ListTo4DArray.shape[1],1), order=0)                 #then resize them

        else:                                                                                             # if not
            for i in range(batch_size):                                                                   # then resize them one by one
                factor_h = self.image_size[0]/single_batch_of_images_list[i].shape[0]
                factor_w = self.image_size[1]/single_batch_of_images_list[i].shape[1]
                single_batch_of_resized_images_4Darray[i] = scipy.ndimage.zoom(single_batch_of_images_list[i], zoom = (factor_h,factor_w,1), order=0)                                                                                             #then fill the 4d array image by image

        
        
        
        #step 5&6: check mirror & rotate
        if ((self.rotation == True) and (self.image_size[0] == self.image_size[1])) or (self.mirroring == True):    #if images ae not squared, then don't rotate
            image_indx_to_transform = random.sample(range(self.batch_size), k=randrange(self.batch_size))           #if any transformation is requested, then select a random sample of a random size from the batch
            for i in image_indx_to_transform:                                                                   #then pass the images in the sample one by one to the transformation function
                single_batch_of_resized_images_4Darray[i] = self.augment(single_batch_of_resized_images_4Darray[i])
                
                
           
        
        #step 7: save images in array and save it in the output variable (batch)
        batch = single_batch_of_resized_images_4Darray
        
        
        #step 8: save corresponding lables in output variable (lables)
        labels = batched_image_labels_matrix[self.counter]
        

        #step 10: increment counter
        if self.counter < (len(batched_image_names_matrix)-1):
            self.counter = self.counter+1
        else:
            self.counter = 0
            
        
        
        

        return batch, labels

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function

        if ((self.rotation == True) and (self.image_size[0] == self.image_size[1])) and (self.mirroring == False):
            #img = np.rot90(img, randrange(4)) #in this implementation, every image has 25% chance of not being rotated
            return np.rot90(img, random.choice([1,2,3])) #in this implementation, every image has is rotated in one way or another

        elif (self.rotation == False) and (self.mirroring == True):
            return np.fliplr(img)

        elif ((self.rotation == True) and (self.image_size[0] == self.image_size[1])) and (self.mirroring == True):
            return np.rot90(np.fliplr(img), randrange(4)) #in this implementation, every image has 25% chance of not being rotated

        else:
            return img
        
        

    
    
    def class_name(self, x):
        # This function returns the class name for a specific input
        #TODO: implement class name function
               
        return self.class_dict[x]
    
    
    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method
        images, labels = self.next()
        fig = plt.figure(figsize=(15, 15))
        cols = int(round(np.sqrt(self.batch_size)))
        rows = np.ceil(self.batch_size/cols)
        
        img_grid = []
        
        for i in range(len(images)):
            img = images[i]
            img_grid.append( fig.add_subplot(rows, cols, i+1) )
            img_grid[-1].set_title(self.class_name(labels[i]))
            plt.axis('off')
            plt.imshow(img)
            
        plt.show()
        
        
        
        
        

