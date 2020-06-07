import csv
import os
import os.path
import tarfile
from urllib.parse import urlparse

import numpy as np
import scipy.io
import torch
import torch.utils.data as data
from PIL import Image
import pickle
import util
from util import *

images_folder_path = os.path.join('ESP-ImageSet', 'images')
datamap_path = os.path.join('misc', 'esp_data.mat')
mat = scipy.io.loadmat(datamap_path)

def get_classes():
    classes = []
    keywords = mat['dict']
    for keyword in keywords:
        keyword = keyword[0][0]
        classes.append(keyword)
    return classes

def get_labelled_data(setname):
    images = []
    dataset = mat[dataset][0]
    num_img = len(mat['data'][0])
    num_classes = len(mat['dict'])
    # annot = np.zeros(shape = (num_classes, num_img), dtype=np.int64)
    data = mat['data'][0]
    for val in dataset:
        img = data[val-1]
        keywords = img['keywords'][0]
        name = img['file'][0]
        labels = np.zeros(shape = (num_classes, ), dtype=np.int64)
        for keyword in keywords:
            labels[keyword-1] = 1
        labels = torch.from_numpy(labels)
        item = (name, labels)
        images.append(item)
    return images

class ESPGAME(data.Dataset):
    def __init__(self, root, set, transform=None, target_transform=None, inp_name=None, adj=None):
        self.root = root
        self.path_images = os.path.join(root, images_folder_path)
        self.set = set
        self.transform = transform
        self.target_transform = target_transform
        self.classes = get_classes()

        if self.set == 'trainval':
            setname = 'train'
        else:
            setname = 'test'

        self.images = get_labelled_data(setname)

        with open(inp_name, 'rb') as f:
            self.inp = pickle.load(f)
        self.inp_name = inp_name

    def __getitem__(self, index):
        path, target = self.images[index]
        img = Image.open(os.path.join(self.path_images, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (img, path, self.inp), target

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return len(self.classes)