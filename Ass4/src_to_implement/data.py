from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
import os

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]

class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description
    
    def __init__(self, data , mode):
        self.data = data
        self.mode = mode                                                                            #when do we use the mode
        self._transform = tv.transforms.Compose([tv.transforms.ToPILImage(),                        #why the "_"?
                                                 tv.transforms.ToTensor(),                          #should we creat two members?
                                                 tv.transforms.Normalize(train_mean, train_std)])
        self.classes = ['crack','inactive']
        #self.train_transform = tv.transforms.Compose([tv.transforms.ToPILImage(),                    
        #                                         tv.transforms.ToTensor(),                       
        #                                         tv.transforms.Normalize(train_mean, train_std)])
        
        #self.val_transform = tv.transforms.Compose([tv.transforms.ToPILImage(),                    
        #                                         tv.transforms.ToTensor(),                       
        #                                         tv.transforms.Normalize(train_mean, train_std)])
        
        
    
    def __len__(self):
        return len(self.data)
    
    
    
    def __getitem__(self, index):
        
        if torch.is_tensor(index):
            index = index.tolist()

        img_name = os.path.join("C:\\Users\\utg_1\\OneDrive\\Documents\\Mohamed's\\Studies\\FAU\\SS20\\DL\\Ass4\\src_to_implement\\", self.data.iloc[index, 0])
        image = imread(img_name)
        rgb_img = gray2rgb(image)
        lbl = self.data.iloc[index,1::]
        #bl = np.array([lbl])
        
        if self.mode == "train":
            rgb_img = self._transform(rgb_img)
            
        else:
            rgb_img = self._transform(rgb_img)
            
        

        
            
        # convert to torch tensor and return
        images = torch.tensor(rgb_img)
        labels = torch.tensor(lbl[self.classes]).float()

        return images, labels
        
        
        
        
        
        
        
        
        
   