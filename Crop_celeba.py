from PIL import Image
import os
# root path depends on your computer
root = 'D:\\Celeba\\'
save_root = 'G:\\Celeba_Crop_128\\'

if not os.path.isdir(save_root):
    os.mkdir(save_root)
img_list = os.listdir(root)

for i in range(len(img_list)):
    img = Image.open(root + img_list[i])
    img = img.crop([25,50,25+128,50+128])
    img.save(save_root + img_list[i])
    if (i % 1000) == 0:
        print('%d images complete' % i)