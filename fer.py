''' Fer2013 Dataset class'''

from __future__ import print_function
from PIL import Image
import numpy as np
import h5py
import torch.utils.data as data
import pandas as pd


class FER2013(data.Dataset):
    """`FER2013 Dataset.

    Args:
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self, split='Training', transform=None):
        self.transform = transform
        self.split = split  # training set or test set
        # self.data = h5py.File('./data/fer2013.h5', 'r', driver='core')
        self.data = pd.read_csv("./data/fer2013.csv")
        # now load the picked numpy arrays
        if self.split == 'Training':
            # self.train_data = self.data['pixels']
            self.train_data = self.data.loc[self.data['Usage'] == 'Training', 'pixels']
            self.train_labels = self.data.loc[self.data['Usage'] == 'Training', 'emotion']
            self.train_data = np.asarray(self.train_data)
            # self.train_data = self.train_data.reshape((28709, 48, 48))  #temporarily disabled by YH.
        elif self.split == 'PublicTest':
            # self.PublicTest_data = self.data['PublicTest_pixel']
            self.PublicTest_data = self.data.loc[self.data['Usage'] == 'PublicTest', 'pixels']
            ############revise by YH. change index from 0
            # 创建新的从 0 开始的索引
            new_index = np.arange(len(self.PublicTest_data))
            # 将新索引应用到数组
            self.PublicTest_data = pd.Series(self.PublicTest_data.values, index=new_index)
            ##################
            # self.PublicTest_labels = self.data['PublicTest_label']
            self.PublicTest_labels = self.data.loc[self.data['Usage'] == 'PublicTest', 'emotion']
            ############revise by YH. change index from 0
            # 创建新的从 0 开始的索引
            #new_index = np.arange(len(self.PublicTest_labels))
            # 将新索引应用到数组
            self.PublicTest_labels = pd.Series(self.PublicTest_labels.values, index=new_index)
            ##################
            self.PublicTest_data = np.asarray(self.PublicTest_data)
            # self.PublicTest_data = self.PublicTest_data.reshape((3589, 48, 48))

        else:
            # self.PrivateTest_data = self.data['PrivateTest_pixel']
            self.PrivateTest_data = self.data.loc[self.data['Usage'] == 'PrivateTest', 'pixels']
            ###########
            new_index = np.arange(len(self.PrivateTest_data))
            self.PrivateTest_data = pd.Series(self.PrivateTest_data.values, index=new_index)
            #############
            # self.PrivateTest_labels = self.data['PrivateTest_label']
            self.PrivateTest_labels = self.data.loc[self.data['Usage'] == 'PrivateTest', 'emotion']
            #############
            self.PrivateTest_labels = pd.Series(self.PrivateTest_labels.values, index=new_index)
            ##############
            self.PrivateTest_data = np.asarray(self.PrivateTest_data)
            # self.PrivateTest_data = self.PrivateTest_data.reshape((3589, 48, 48))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.split == 'Training':
            img, target = self.train_data[index], self.train_labels[index]
        elif self.split == 'PublicTest':
            img, target = self.PublicTest_data[index], self.PublicTest_labels[index]
        else:
            img, target = self.PrivateTest_data[index], self.PrivateTest_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        string_list = img.split()
        img = [int(num) for num in string_list]
        img = np.array(img)
        # img = img[:, :, np.newaxis]
        img = img.reshape(48, 48)
        img = img[:, :, np.newaxis]

        img = np.concatenate((img, img, img), axis=2)
        img = img.astype(np.uint8)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        if self.split == 'Training':
            return len(self.train_data)
        elif self.split == 'PublicTest':
            return len(self.PublicTest_data)
        else:
            return len(self.PrivateTest_data)
