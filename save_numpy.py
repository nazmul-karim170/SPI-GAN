import cv2
import glob
import numpy as np
from PIL import Image 
from os import listdir
from os.path import join

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

####Train data
train = []
test = []

# files = glob.glob("./data/STL_10_train/CS_Input_Images_64/*.png") # Your image path
# dataset_dir_original = '/home/ryota/anaconda3/Computational Imaging /Main Methods/SPIRIT-master/data/CS_methods_reconstruction/DCT/ratio_5'
# dataset_dir_original ='/home/ryota/anaconda3/Project_2021/Computational_Imaging /Main Methods/SPIRIT-master/data/STL10_reconstruction/CS_Images_unlabeled_0.40'
dataset_dir_original ='/home/ryota/anaconda3/Project_2021/Computational_Imaging /Main Methods/SPIRIT-master/SunHays80_SR/Reconstructed/ratio_0.40/'

files  = [join(dataset_dir_original, x) for x in listdir(dataset_dir_original) if is_image_file(x)]
num = 0

## For Different Ratio
# for myFile in files:
#     print(myFile)
#     # ext = myFile.split('_')[7].split('.')[0]
#     # if int(ext)<=45000:
#     if num>5000:
#         image = np.array(Image.open(myFile))
#         train.append(image)
#     else:
#       image = np.array(Image.open(myFile))
#       test.append(image)
#       num +=1

#     if num ==45000:
#     	break
#     print(num)
#     # print(np.shape(train))

# ## For new dataset
for myFile in files:

    image = np.array(Image.open(myFile))
    train.append(image)

    num +=1
    print(num)

train = np.array(train,dtype='float32')                 # As mnist
# test  = np.array(test ,dtype='float32')   

## Convert (number of images x height x width x number of channels) to (number of images x (height * width *3)) 
## for example (120 * 40 * 40 * 3)-> (120 * 4800)
## train = np.reshape(train,[train.shape[0],train.shape[1]*train.shape[2]*train.shape[3]])

##### Save numpy array as .npy formats
# np.save('STL_train_0.40.npy',train)
# np.save('STL_test_0.40.npy',test)
np.save('SUNHAYS_recons_40.npy',train)
# np.save('DCT_test.npy',test) 			# saves test.npy


