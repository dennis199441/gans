from PIL import Image
import glob, os
import numpy as np

output_folder = "../preprocessed"
if not os.path.exists(output_folder):
	os.mkdir(output_folder)

folder_name = "../pokemon"
filenames = glob.glob(folder_name + '/*.png')

for i,filename in enumerate(filenames):
    img = Image.open(filename)
    new_filename = filename.replace(folder_name, output_folder)
    out = img.convert('RGB')
    out = out.resize((96,96))
    out.save(new_filename, "png")

print("process done!\n\n")

print("original data")
for i,filename in enumerate(filenames):
    img = Image.open(filename)
    array = np.asarray(img)
    print("filename = {}, shape = {}, bands = {}".format(filename, array.shape, img.getbands()))

print("preprocessed data")
filenames = glob.glob(output_folder + '/*.png')
for i,filename in enumerate(filenames):
    img = Image.open(filename)
    array = np.asarray(img)
    print("filename = {}, shape = {}, bands = {}".format(filename, array.shape, img.getbands()))