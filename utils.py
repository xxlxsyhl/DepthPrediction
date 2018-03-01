import skimage
import skimage.io
import skimage.transform
import numpy as np

# returns image of shape [224, 224, 3]
def load_image(path):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    return resized_img


def imshow(image):
    image = image*255
    image = image.astype(np.ubyte)
    skimage.io.imshow(image)
    skimage.io.show()


def test():
    image = load_image("tiger.jpeg")
    imshow(image)
