from torch.utils.data import Dataset
import torch
import pandas as pd
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
from sklearn.model_selection import train_test_split
import torchvision as tv
import os

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]

class ChallengeDataset(Dataset):
    #TODO implement the Dataset class according to the description

    def __init__(self, mode_flag, CSV_path, train_val_split, transform = tv.transforms.Compose([tv.transforms.ToTensor()])):
        # 'C:/Users/ahmed/PycharmProjects/dl/src_to_implement/train.csv'
        self.data = pd.read_csv(CSV_path)
        self.d_train, self.d_val,  self.train_labels, self.val_labels = train_test_split(self.data.iloc[1:,0],self.data.iloc[1:,2:],test_size = train_val_split, random_state= 42)

        self.mode_flag = mode_flag
        # self.CSV_path = CSV_path
        self.train_val_split = train_val_split
        self.transform = transform

        self.i = None
        self.l = None
        self.length = 0

    def __len__(self):
        if self.mode_flag == "train":
            return self.d_train.shape[0]
        else:
            return self.d_val.shape[0]

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.mode_flag == "train":
            img_name = self.d_train.iloc[idx] # I think we should read the image name ohne the numbers [-6]\
            label = self.train_labels.iloc[idx] #debug
        else:
            img_name = self.d_val.iloc[idx]
            label = self.val_labels.iloc[idx] #debug

        img_path = os.path.join("", img_name)
        label = label.to_numpy()
        image = imread(img_path)
        rgb_img = gray2rgb(image)

        # apply transformation if exists
        if self.transform:
            rgb_img = self.transform(rgb_img)

        # convert to torch tensor and return

        self.i = torch.tensor(rgb_img)
        self.l = torch.tensor(label, dtype=torch.float32) # cpu
        # self.l = torch.tensor(label, dtype=torch.cuda.float32)  # gpu
        return (self.i, self.l)

        # return (rgb_img.clone().detach(), label.clone().detach())


    def pos_weight(self):


        l = self.train_labels
        leng = l.shape[0]
        ones = [1,1]
        w = [0,0]
        neg = [0,0]

        for i in range(leng):
            neg += ones - l.iloc[i]
            w += l.iloc[i]

        w = neg / w
        return torch.tensor(w)


def get_train_dataset():
    #TODO
    transform_train = tv.transforms.Compose([tv.transforms.ToPILImage(),

                                               tv.transforms.ToTensor(),
                                               tv.transforms.Normalize(train_mean, train_std)])
    # instantiate
    train_data = ChallengeDataset("train", "train.csv",  0.33,  transform = transform_train)

    return train_data


# this needs to return a dataset *without* data augmentation!
def get_validation_dataset():

    transform_validation = tv.transforms.Compose([tv.transforms.ToPILImage(),
                                                  tv.transforms.ToTensor(),
                                                  tv.transforms.Normalize(train_mean, train_std)])
    validation_data = ChallengeDataset("val", "train.csv",  0.33,  transform = transform_validation)

    return validation_data