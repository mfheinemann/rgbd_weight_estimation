import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np


# IMAGE_PATH = '/media/michel_ma/NVMe2/MA_Heinemann_Dataset/000_All_data/002_Real/002_Mug_Large_real/Mug_Large_real000189.depth.png'
# IMAGE_PATH = '/media/michel_ma/NVMe2/MA_Heinemann_Dataset/000_All_data/000_Synthetic_simple/0001_Adjustable_Wrench/Adjustable_Wrench000000.depth.png'

IMAGE_PATH = '/home/michel_ma/MA_Heinemann/catkin_ws/src/rgbd_weight_estimation/record_rs_frames/Banana_2_000000.depth.png'

img = cv.imread(IMAGE_PATH, cv.IMREAD_ANYDEPTH)

# img[img==255] = 0.0
# img = img.astype(np.int16)
# img = img*100
img = img*10
print(img)
print(np.max(img))
print(np.min(img))
cv.imwrite(IMAGE_PATH, img)
# plt.show()


