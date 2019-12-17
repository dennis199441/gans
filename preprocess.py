from PIL import Image
import glob, os

if not os.path.exists('../preprocessed'):
	os.mkdir("../preprocessed/")

folder_name = "../pokemon"
filenames = glob.glob(folder_name + '/*.png')

for i,filename in enumerate(filenames):
    img = Image.open(filename)
    new_filename = filename.replace(folder_name, "../preprocessed/")
    resized_img = img.resize((96,96))
    resized_img.save(new_filename, "png")
    print("new_filename = {}, shape = {}".format(new_filename, resized_img.shape))