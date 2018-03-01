import numpy as np
import h5py
from PIL import Image
import flow_transforms
import matplotlib.pyplot as plt
from nyu_dataset_loader import *

IMAGE_NUM = 0

def load_data(is_training = True):
    global IMAGE_NUM
    train_idx_path = "data/trainNdxs.txt"
    test_idx_path = "data/testNdxs.txt"
    input_rgb_images_dir = 'data/nyu_datasets_changed/input/'
    target_depth_images_dir = 'data/nyu_datasets_changed/target_depths/'
    target_labels_images_dir = 'data/nyu_datasets_changed/labels_38/'

    data_path = "data/nyu_depth_v2_labeled.mat"
    train_idx = np.loadtxt(train_idx_path, dtype = 'int')
    test_idx = np.loadtxt(test_idx_path, dtype = 'int')

    input_transform = flow_transforms.Compose([flow_transforms.Scale(120)])
    target_depth_transform = flow_transforms.Compose([flow_transforms.Scale_Single(60)])
    target_labels_transform = flow_transforms.Compose([])

    co_transform=flow_transforms.Compose([
            flow_transforms.RandomRotate(4),
            flow_transforms.RandomCrop((480,640)),
            flow_transforms.RandomVerticalFlip()
        ])

    data = []
    if is_training:
        data = ListDataset(data_path,train_idx,input_transform,target_depth_transform, target_labels_transform, co_transform)
    else:
        data = ListDataset(data_path, test_idx, input_transform, target_depth_transform, target_labels_transform)
    IMAGE_NUM = len(data)
    return data

"""
def load_data(path):
    data = h5py.File(path, 'r')
    IMAGE_NUM = data['images'].shape[0]
    print("load data successfully")
    return data
"""

beg = 0
def get_batch(data, batch_size):
    global beg
    if (beg >= IMAGE_NUM):
        beg = 0
    end = beg+batch_size

    imgs = []
    deps = []
    for idx in range(beg, end):
        rgb, depth = data[idx % IMAGE_NUM]
        imgs.append(np.array(rgb[0])/255.0)
        deps.append(depth[0])

    beg += batch_size
    return imgs, deps

