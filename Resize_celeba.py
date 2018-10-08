import os
from PIL import Image
from scipy.misc import imresize

# root path depends on your computer
root = 'G:\\Celeba_Crop_128\\'
save_root = 'G:\\Celeba_Resize_64\\'
resize_size = 64

if not os.path.isdir(save_root):
    os.mkdir(save_root)
if not os.path.isdir(save_root+'train\\'):
    os.mkdir(save_root+'train\\')
if not os.path.isdir(save_root+'valid\\'):
    os.mkdir(save_root+'valid\\')
if not os.path.isdir(save_root+'test\\'):
    os.mkdir(save_root+'test\\')

img_list = os.listdir(root)
NUM_EXAMPLES = 202599
TRAIN_STOP = 162770
VALID_STOP = 182637
if len(img_list)==NUM_EXAMPLES:
    for i in range(0, TRAIN_STOP):
        img = Image.open(root + img_list[i])
        img = imresize(img, (resize_size, resize_size))
        img = Image.fromarray(img)
        img.save(save_root +'train\\'+ img_list[i])
        if (i % 1000) == 0:
            print('%d images complete' % i)
    for i in range(TRAIN_STOP, VALID_STOP):
        img = Image.open(root + img_list[i])
        img = imresize(img, (resize_size, resize_size))
        img = Image.fromarray(img)
        img.save(save_root +'valid\\'+ img_list[i])
        if (i % 1000) == 0:
            print('%d images complete' % i)
    for i in range(VALID_STOP, NUM_EXAMPLES):
        img = Image.open(root + img_list[i])
        img = imresize(img, (resize_size, resize_size))
        img = Image.fromarray(img)
        img.save(save_root +'test\\'+ img_list[i])
        if (i % 1000) == 0:
            print('%d images complete' % i)
