import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

image_dir = '/media/yotamg/Yotam/Stereo/from_ids/'
image_dir = '/home/yotamg/Documents/ZED/'
l_name = 'image3L.bmp'
l_name = 'imgL.png'
r_name = 'image3R.bmp'
r_name = 'imgR.png'

imgL = cv2.imread(os.path.join(image_dir, l_name),0)
imgR = cv2.imread(os.path.join(image_dir, r_name),0)

# stereo = cv2.createStereoBM(numDisparities=16, blockSize=15)
stereo = cv2.StereoBM_create(numDisparities=192, blockSize=15)
# stereo = cv2.StereoBM(disparities=16, SADWindowSize=15)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity)
plt.show()