import glob, PIL
import numpy as np
from PIL import Image

def load_data(path):
    filenames = glob.glob(path + '/*.png')
    imgs = []
    for i,filename in enumerate(filenames):
        img = Image.open(filename)
        array = np.asarray(img)
        imgs.append(array)
    imgs = np.array(imgs)
    print("imgs.shape = {}".format(imgs.shape))
    return imgs

train_images = load_data("../preprocessed")
train_images = train_images.reshape(train_images.shape[0], 96, 96, 3).astype('float32')

print(train_images.shape)