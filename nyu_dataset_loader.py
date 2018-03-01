import os
import os.path
from scipy.ndimage import imread
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py

class ListDataset(object):
    def __init__(self,data_dir,listing,input_transform=None,target_depth_transform=None,
                target_labels_transform=None,co_transform=None,random_scale = None):

        self.data_dir = data_dir
        self.listing = listing
        #self.depth_imgs = depth_imgs
        self.input_transform = input_transform
        self.target_depth_transform = target_depth_transform
        self.target_labels_transform = target_labels_transform
        self.co_transform = co_transform
        self.f = h5py.File(data_dir)

    def __getitem__(self, index):
        img_idx = self.listing[index]

        input_im = self.f['images'][img_idx].transpose(2, 1, 0)
        target_depth_im = self.f['depths'][img_idx].transpose(1, 0)

        if self.co_transform is not None:
            input_im, target_depth_im = self.co_transform(input_im,target_depth_im)

        if self.input_transform is not None:
            input_im = self.input_transform(input_im)

        if self.target_depth_transform is not None :
            target_depth_im = self.target_depth_transform(target_depth_im)

        return input_im,target_depth_im

    def __len__(self):
        return len(self.listing)
