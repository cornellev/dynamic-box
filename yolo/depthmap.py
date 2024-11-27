import cv2
import numpy as np
import os

from matplotlib import pyplot as plt

# Load stereo images
lefti = "left0.png"
righti = "right0.png"
imgL = cv2.imread(lefti, cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread(righti, cv2.IMREAD_GRAYSCALE)

# Initialize the stereo block matcher
stereo = cv2.StereoBM_create(numDisparities=16*20, blockSize=85)

# Compute disparity map

disparity = stereo.compute(imgL, imgR)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()
f = 1398.82
b = 119.905
depth = (f*b)/(disparity + 1e-5)
print(depth)

# Normalize the disparity for better visualization
disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
disparity = np.uint8(disparity)

# Save the disparity map
cv2.imwrite(os.path.join("data", 'disparity_map.jpg'), disparity)



