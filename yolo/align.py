"""
Both a subscriber and a publisher:
Subscribes to "yolo": takes in the left, right annotated images and:
1) Pushes the image pair of (boxes, confs, class_ids) into an [ROIalign] stack
   that stores pairs in FIFO order and only processes an image pair if either the
   popped pair has been published or deleted.
2) Pops (removed from [ROIalign]) the first pair from [ROIalign] and does ROI Align 
   and calculates SSIM similarity of the two images:
   Compare similarity between bounding box locations of the two images along with 
   if annotated objects have the same label in the same location.
   If SSIM similarity is within threshold, publish the pair to "aligned" subscribers.
   Delete the pair completely if not aligned.
"""

import cv2
import os
import numpy as np
import math
import threading
import matplotlib.pyplot as plt
from scipy.linalg import svd
import YOLO
import asyncio

np.set_printoptions(suppress=True)

# Given the sample images from ZED are 1280 x 720, most likely using HD resolution.
# [LEFT_CAM_HD]
fx_l=699.41
fy_l=699.365
cx_l=652.8
cy_l=358.133
k1_l=-0.170704
k2_l=0.0249542
p1_l=9.55189e-05
p2_l=-0.000109509
k3_l=-9.96888e-11

# [RIGHT_CAM_HD]
fx_r=697.635
fy_r=697.63
cx_r=671.665
cy_r=354.611
k1_r=-0.171533
k2_r=0.0258402
p1_r=7.86599e-05
p2_r=-0.000136126
k3_r=-2.87251e-10

Baseline=119.905
TY=0.0458908
TZ=-0.467919
CV_2K=0.00697667
RX_2K=0.00239722
RZ_2K=-0.0021326

# Canera matrices.
K_L = np.array([[fx_l, 0, cx_l], [0, fy_l, cy_l], [0, 0, 1]])
K_R = np.array([[fx_r, 0, cx_r], [0, fy_r, cy_r], [0, 0, 1]])
T = np.array([Baseline, TY, TZ])
# R_Z, _ = cv2.Rodrigues(np.array([0, 0, RZ_2K]))
# R_Y, _ = cv2.Rodrigues(np.array([0, CV_2K, 0]))
# R_X, _ = cv2.Rodrigues(np.array([RX_2K, 0, 0]))
R, _ = cv2.Rodrigues(np.array([RX_2K, CV_2K, RZ_2K]))
T_x = np.array([[0, -TZ, TY], [TZ, 0, -Baseline], [-TY, Baseline, 0]])
E = T_x @ R
W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

lefti = "left0.png"
righti = "right0.png"

# Use this to get points to draw epipolar lines. Not for 8-point algorithm.
# Make YOLO async.
non_dups_left, non_dups_right = YOLO.get_pair([lefti, righti])

# Given the bounding boxes for each image: do 8 point algorithm for corresponding images in each bounding box.
def eightPoint ():
   global lefti, righti
   sift = cv2.SIFT_create()
   l_img = cv2.imread(lefti)
   r_img = cv2.imread(righti)
   kp1, des1 = sift.detectAndCompute(l_img, None)
   kp2, des2 = sift.detectAndCompute(r_img, None)

   bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
   matches = bf.match(des1, des2)

   matches = sorted(matches, key=lambda x: x.distance)
   good_matches = matches[:8]
   matched_img = cv2.drawMatches(l_img, kp1, r_img, kp2, good_matches, None, flags=2)

   left_P = np.float32([kp1[m.queryIdx].pt for m in good_matches])
   right_P = np.float32([kp2[m.trainIdx].pt for m in good_matches])

   return left_P, right_P

# Normalize left, right corresponding points.
def normalizePoints(points):
   x_, y_ = np.mean(points, axis=0)
   d = np.mean(np.linalg.norm(points-np.array([x_, y_]), axis = 1))
   s = np.sqrt(2)/d
   s1 = 1/np.sqrt((np.sum((points[:,0]-x_)**2)/points.shape[0]))
   s2 = 1/np.sqrt((np.sum((points[:,1]-y_)**2)/points.shape[0]))
   T = np.array([[s1, 0, -s1 * x_],
                 [0, s2, -s2 * y_],
                 [0, 0, 1]])
   P = (T @ np.hstack([points, np.ones((points.shape[0], 1))]).T).T
   return P[:, :2], T

def fundamentalMatrix(left_P, right_P):
   left, T1 = normalizePoints(left_P)
   right, T2 = normalizePoints(right_P)
   A = np.zeros((left.shape[0],9))
   for i in range(left.shape[0]):
      x1, y1 = left[i]
      x2, y2 = right[i]
      A[i] = [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1]
   _, _, V = svd(A)
   F = V[-1].reshape(3, 3)
   U, S, V = svd(F)
   S[2] = 0
   F = U @ np.diag(S) @ V
   return T2.T @ F @ T1

def getDepthMap (ptl, ptr):
   # Depth map: 2D u_r = M_intR @ x_r (3D), u_l = P_L @ x_r
   # Need corresponding points left ptl and right ptr.
   M_intL = np.append(K_L, np.array([[0], [0], [0]]), axis = 1)
   M_intR = np.append(K_R, np.array([[0], [0], [0]]), axis = 1)
   R_T = np.append(np.append(R, np.array([[Baseline], [TY], [TZ]]), axis = 1), [np.array([0, 0, 0, 1])], axis = 0)
   P_L = M_intL @ R_T
   u_l, v_l, _ = ptl
   u_r, v_r, _ = ptr
   # A @ x_r = b
   A = np.array([[u_r*M_intR[2][0]-M_intR[0][0], u_r*M_intR[2][1]-M_intR[0][1], u_r*M_intR[2][2]-M_intR[0][2]],
                 [v_r*M_intR[2][0]-M_intR[1][0], v_r*M_intR[2][1]-M_intR[1][1], v_r*M_intR[2][2]-M_intR[1][2]],
                 [u_l*P_L[2][0]-P_L[0][0], u_l*P_L[2][1]-P_L[0][1], u_l*P_L[2][2]-P_L[0][2]],
                 [v_l*P_L[2][0]-P_L[1][0], v_l*P_L[2][1]-P_L[1][1], v_l*P_L[2][2]-P_L[1][2]]])
   b = np.array([[M_intR[0][3]-M_intR[2][3]], 
                 [M_intR[1][3]-M_intR[2][3]], 
                 [P_L[0][3]-P_L[2][3]], 
                 [P_L[1][3]-P_L[2][3]]])
   # 3D coordinate in the right image x_r = np.linalg.inv(A.T @ A) @ A.T @ B
   return np.linalg.inv(A.T @ A) @ A.T @ b

# Find point on right image [u_r, v_r, 1] such that a*u_r + b*v_r + c = 0.
def getEpipolarLines (img1, img2, pt, F, data):
      pt, dir = pt
      # tmg1 = np.pad(img1, pad_width=120, mode = "constant", constant_values = 0)
      # tmg2 = np.pad(img2, pad_width=120, mode = "constant", constant_values = 0)
      # Given: a point on the right image, fundamental matrix F
      # Return: equation of epipolar line in the left image.
      if (dir == "RIGHT"): 
         a = (F[0][0] * pt[0] + F[1][0] * pt[1] + F[2][0])
         b = (F[0][1] * pt[0] + F[1][1] * pt[1] + F[2][1])
         c = (F[0][2] * pt[0] + F[1][2] * pt[1] + F[2][2])
      # Given: a point on the left image, fundamental matrix F
      # Return: equation of epipolar line in the right image.
      elif (dir == "LEFT"):
         a = (F[0][0] * pt[0] + F[0][1] * pt[1] + F[0][2])
         b = (F[1][0] * pt[0] + F[1][1] * pt[1] + F[1][2])
         c = (F[2][0] * pt[0] + F[2][1] * pt[1] + F[2][2])

      y = lambda x : -(a/b)*x - c/b

      # Do block matching on left point and right epipolar line within the range of the left point:
      ptr = np.array([0, 0, 1])
      # print(pt[0])
      # print((tmg1[pt[1]+Baseline, pt[0]+Baseline]))
      block = lambda im, pt : im[int(max(0.,pt[1]-50.)):int(min(float(img1.shape[0]), pt[1]+50.)), 
                                 int(max(0.,pt[0]-50.)):int(min(float(img1.shape[1]), pt[0]+50.)),
                                 :] 
      blockl = block (img1, pt)
      ptr = np.array([int(max(0.,pt[0]-50.)), 
                              y(int(max(0.,pt[0]-50.)))])
      diff = np.absolute(blockl - 
                              block (img2, ptr))
      weight = 1000 * np.absolute(img1[int(pt[1])][int(pt[0])] - img2[int(y(ptr[0]))][int(ptr[0])])
      min_SAD = np.sum(np.sum(diff, axis=(0,1)))
      # I HAVE A LINE: 
      for x in range(int(max(0.,pt[0]-Baseline)), int(min(float(img1.shape[1]), pt[0]))):
         blockr = block (img2, np.array([x, y(x)]))
         abs_diff = np.absolute(blockl - blockr)
         weight = 1000 * np.sum(np.sum(np.absolute(img1[int(pt[1])][int(pt[0])] - img2[int(y(x))][x])))
         sum = np.sum(np.sum(abs_diff, axis=(0,1))) + np.sum(weight)
         if sum < min_SAD:
            min_SAD = sum
            ptr = np.array([x, y(x)])
      
      # im2 = cv2.line(img2, (0, int(y(0))), (img1.shape[1], int(y(img1.shape[1]))), (0, 0, 255), 1)
      # im2 = cv2.circle(img2, (int(max(0.,pt[0]-Baseline)), int(y(max(0.,pt[0]-Baseline)))), 5, (100, 20, 100), -1)
      # im2 = cv2.circle(img2, (int(min(float(im1.shape[1]), pt[0])), int(y(min(float(im1.shape[1]), pt[0])))), 5, (255, 255, 0), -1)
      z = (fx_l * Baseline) / (pt[0] - ptr[0])
      x = pt[0] - cx_l * z/fx_l
      y = pt[1] - cy_l * z/fy_l
      # print(np.array([x,y,z])*0.001)
      data = np.concatenate((data, [[x*0.001], [y*0.001], [z*0.001]]), axis=1)
      # print(getDepthMap(pt, np.append(ptr, np.array([1.])))  * 0.001)
      # if (z*0.001 > 1.0 and z*0.001 < 2.4):
      im1 = cv2.circle(img1, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
      im2 = cv2.putText(img2, str(round(z*0.001,3)), (int(ptr[0]), int(ptr[1])-10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
      # im1 = cv2.putText(img1, str(round(z*0.001,3)), (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
      z = getDepthMap(pt, np.append(ptr, np.array([1.])))[-1]  * 0.001
      # im2 = cv2.putText(img2, str(round(z[0],3)), (int(ptr[0]), int(ptr[1])+30), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 2)

      im2 = cv2.circle(img2, (int(ptr[0]), int(ptr[1])), 5, (0, 0, 255), -1)
   
      cv2.imwrite("im.png", block (im2, ptr) )
      return img1, img2, data

def drawMinMaxBox (data):
   means = np.mean(data, axis = 1)
   cov = np.cov(data)
   eigval, eigvec = np.linalg.eig(cov)
   centered_data = data - means[:,np.newaxis]

# def drawOrientedBox

# left_P, right_P = eightPoint()

# left_ = np.hstack((left_P, np.ones((left_P.shape[0], 1))))
# right_ = np.hstack((right_P, np.ones((right_P.shape[0], 1))))
# F = fundamentalMatrix(left_P, right_P)
# F, mask = cv2.findFundamentalMat(left_P, right_P, method=cv2.RANSAC)
F = np.linalg.inv(K_L).T @ E @ np.linalg.inv(K_R)
E = K_R.T @ F @ K_L
U, S, V = np.linalg.svd(E)

# img1, img2 = getEpipolarLines(cv2.imread(lefti), cv2.imread(righti), (left_[0], "LEFT"), F)
# img1, img2 = getEpipolarLines(img1, img2, (np.array([800., 100., 1.]), "LEFT"), F)
# img1, img2 = getEpipolarLines(img1, img2, (left_[1], "LEFT"), F)
# img1, img2 = getEpipolarLines(img1, img2, (left_[4], "LEFT"), F)
# img1, img2 = getEpipolarLines(img1, img2, (left_[6], "LEFT"), F)
# img1, img2 = getEpipolarLines(img1, img2, (left_[7], "LEFT"), F)
# img1, img2 = getEpipolarLines(img1, img2, (np.array([250., 300., 1.]), "LEFT"), F)
# img1, img2 = getEpipolarLines(img1, img2, (np.array([200., 500., 1.]), "LEFT"), F)
# cv2.imwrite(os.path.join("data", "leftepip.png"), img1)
# cv2.imwrite(os.path.join("data", "rightepip.png"), img2)


img1, img2 = cv2.imread(lefti), cv2.imread(righti)
# for left in non_dups_left:
left = non_dups_left[2]
data = np.array([[0.], [0.], [0.]])
for x in range(left[0], left[0]+left[2], 20):
   for y in range(left[1], left[1]+left[3], 20):
      pt = np.array([x, y, 1])
      img1, img2, data = getEpipolarLines(img1, img2, (pt, "LEFT"), F, data)
      # left_P = np.array([left[0],left[1]])
      # left_P = np.vstack((left_P, np.array([left[0]+left[2],left[1]])))
      # left_P = np.vstack((left_P, np.array([left[0]+left[2],left[1]+left[3]])))
      # left_P = np.vstack((left_P, np.array([left[0],left[1]+left[3]])))
drawMinMaxBox(data)

# left_P = np.array([200,200])
# for left in non_dups_left:
#    left_P = np.vstack((left_P, np.array([left[1],left[0]])))
#    left_P = np.vstack((left_P, np.array([left[1],left[0]+left[2]])))
#    left_P = np.vstack((left_P, np.array([left[1],left[0]])))
#    left_P = np.vstack((left_P, np.array([left[1]+left[3],left[0]])))

# left_ = np.hstack((left_P, np.ones((left_P.shape[0], 1))))
# img1, img2 = cv2.imread(lefti), cv2.imread(righti)

# for pt in left_:
#    img1, img2 = getEpipolarLines(img1, img2, (pt, "LEFT"), F)

cv2.imwrite(os.path.join("data", "leftepip.png"), img1)
cv2.imwrite(os.path.join("data", "rightepip.png"), img2)
# for right in non_dups_right:
#    right_P.append([right[0],right[1]])
   # right_P.append([right[0]+right[2],right[1]])
   # right_P.append([right[0],right[1]])
   # right_P.append([right[0],right[1]+right[3]])


