import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import h5py
import time
import skimage
import skimage.io
import skimage.transform


def reshape_image(image):
     # print(image.shape)


     img = np.empty([480, 640, 3])
     img[:, :, 0] = image[0, :, :].T
     img[:, :, 1] = image[1, :, :].T
     img[:, :, 2] = image[2, :, :].T
     img = skimage.transform.resize(img,(120,160))
     img = img.astype(np.float32) / 255.0
     return img


def reshape_depth(depth):
     dep = np.empty([480, 640], np.float32)
     dep = depth.T
     dep = np.array(Image.fromarray(dep, "F").resize((80, 60)))
     return dep


orign_data_path = "E:/NYU_Depth/labeled/nyu_depth_v2_labeled.mat"
final_data_path = "E:/NYU_Depth/labeled/nyu_depth_v2_trans.mat"

orign = h5py.File(orign_data_path, "r")
print("load data successfully", orign['images'].dtype)

trans = h5py.File(final_data_path, "w")

trans.create_dataset("images", shape=[1449, 120, 160, 3], dtype='float32', maxshape=(None, 120, 160, 3), chunks=True)
trans.create_dataset("depths", shape=[1449, 60, 80], dtype='float32', maxshape=(None, 60, 80), chunks=True)
print("create dataset successgfully")

img = np.empty([1449, 120, 160, 3], dtype=np.float32)
dep = np.empty([1449, 60, 80], dtype=np.float32)
begin = time.time()
count = 0

for i in range(orign['images'].shape[0]):
     img[i] = reshape_image(orign['images'][i])
     dep[i] = reshape_depth(orign['depths'][i])
     print(i)
     # print(img.shape)
     # print(dep.shape)
     '''
     plt.figure("image and depth")
     plt.subplot(121)
     plt.imshow(img[i])

     plt.subplot(122)
     plt.imshow(dep[i])

     plt.show()
     
     '''

trans['images'].resize(count + 1449, axis=0)
trans['depths'].resize(count + 1449, axis=0)
trans['images'][count:] = img
trans['depths'][count:] = dep

print("Used time: %d s" % (time.time() - begin))

