import numpy as np
import h5py


def create_dataset(origin, idx, dataset_path):
    datatset = h5py.File(dataset_path, "w")
    size = len(idx)
    datatset.create_dataset("images", shape=[size, 120, 160, 3], dtype='float32', maxshape=(None, 120, 160, 3), chunks=True)
    datatset.create_dataset("depths", shape=[size, 60, 80], dtype='float32', maxshape=(None, 60, 80), chunks=True)
    count = 0
    for x in idx:
        x = x-1
        datatset['images'].resize(count + 1, axis=0)
        datatset['depths'].resize(count + 1, axis=0)
        datatset['images'][count:] = origin['images'][x]
        datatset['depths'][count:] = origin['depths'][x]
        count += 1



origin_data_path = "E:/NYU_Depth/labeled/nyu_depth_v2_trans.mat"
train_data_path = "E:/NYU_Depth/labeled/nyu_depth_v2_training.mat"
test_data_path = "E:/NYU_Depth/labeled/nyu_depth_v2_test.mat"

train_idx_path = "E:/NYU_Depth/labeled/trainNdxs.txt"
test_idx_path = "E:/NYU_Depth/labeled/testNdxs.txt"

train_idx = np.loadtxt(train_idx_path)
test_idx = np.loadtxt(test_idx_path)

origin = h5py.File(origin_data_path, "r")
print("load data successfully", origin['images'].dtype)


create_dataset(origin, train_idx, train_data_path)
create_dataset(origin, test_idx, test_data_path)


print("create dataset successgfully")