import glob, PIL
import numpy as np

def load_data(path):
    filenames = glob.glob(path + '/*.png')
    imgs = []
    for i,filename in enumerate(filenames):
        img = Image.open(filename)
        imgs.append(img)
    array = np.asarray(imgs)
    print("array.shape = {}".format(array.shape))

train_images = load_data("../preprocessed")
train_images = train_images.reshape(train_images.shape[0], 96, 96, 3).astype('float32')

print(train_images.shape)