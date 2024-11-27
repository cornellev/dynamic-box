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
import scipy as sp
from scipy.linalg import svd
import YOLO
from mpl_toolkits.mplot3d import Axes3D

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
      # if (z*0.001 > 1.0 and z*0.001 < 2.4):
      data = np.concatenate((data, [[pt[0]], [pt[1]], [z*0.001]]), axis=1)
      # im1 = cv2.circle(img1, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
      # im2 = cv2.putText(img2, str(round(z*0.001,3)), (int(ptr[0]), int(ptr[1])-10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
      # im1 = cv2.putText(img1, str(round(z*0.001,3)), (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
      # im2 = cv2.putText(img2, str(round(z[0],3)), (int(ptr[0]), int(ptr[1])+30), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 2)
      return img1, img2, data

def drawOrientedBox (data, img, box):
   # xmin, xmax, ymin, ymax = box[0], box[0]+box[2], box[1], box[1]+box[3]
   data = data[:, np.abs(data[-1, :] - np.median(data, axis = 1)[-1])/np.std(data[-1]) < 3]
   means = np.mean(data, axis = 1)
   data = data[:, (data[-1, :] > means[-1]-np.std(data[-1])) & ((data[-1, :] < means[-1]+np.std(data[-1])))]
   means = np.mean(data, axis = 1)
   median = np.median(data, axis = 1)

   front_data = data
   back_data = data[:, np.abs(data[-1, :] - np.median(data, axis = 1)[-1])/np.std(data[-1]) < 3]
   xmin_f, xmax_f, ymin_f, ymax_f = int(np.min(front_data[0, :])), int(np.max(front_data[0, :])), int(np.min(front_data[1, :])), int(np.max(front_data[1, :]))
   xmin_b, xmax_b, ymin_b, ymax_b = int(np.min(back_data[0, :])), int(np.max(back_data[0, :])), int(np.min(back_data[1, :])), int(np.max(back_data[1, :]))
   cov = np.cov(back_data)
   eigval, eigvec = np.linalg.eig(cov)
   major_axis = eigvec[:, np.argmax(eigval)]
   yaw = np.arctan2(major_axis[1], major_axis[0])
   pitch = np.arctan2(-major_axis[1], np.sqrt(major_axis[1]**2 + major_axis[2]**2))
   roll = np.arctan2(eigvec[0][1], eigvec[0][2])

   # Positive yaw is counter-clockwise, get projection of x-axis onto yaw degree vector, assume origin is xmax:
   if yaw > 0:
      xmin_f, xmin_b = xmax_f - int((xmax_f-xmin_f)*math.cos(yaw%(math.pi/2))), xmin_b
      xmax_f, xmax_b = xmax_f, xmin_b + int((xmax_b-xmin_b)*math.cos(yaw%(math.pi/2)))

   elif yaw < 0:
      xmin_f, xmin_b = xmin_f, xmax_b - int((xmax_b-xmin_b)*math.cos(yaw%(math.pi/2)))
      xmax_f, xmax_b = xmin_f + int((xmax_f-xmin_f)*math.cos(yaw%(math.pi/2))), xmax_b
      
   if pitch > 0:
      # ymax is the bottom horizontal of the box.
      ymin_f, ymin_b = ymin_f, ymax_b - int((ymax_b-ymin_b)*math.cos(pitch))
      ymax_f, ymax_b = ymin_f + int((ymax_f-ymin_f)*math.cos(pitch)), ymax_b

   elif pitch < 0:
      ymin_f, ymin_b = ymax_f - int((ymax_f-ymin_f)*math.cos(pitch)), ymin_b
      ymax_f, ymax_b = ymax_f, ymin_b + int((ymax_b-ymin_b)*math.cos(pitch))

   img = cv2.rectangle(img, (xmin_f,ymin_f), (xmax_f,ymax_f), (0, 0, 255), 2)
   img = cv2.rectangle(img, (xmin_b,ymin_b), (xmax_b,ymax_b), (0, 0, 255), 2)
   img = cv2.line(img, (xmin_f, ymin_f), (xmin_b, ymin_b), (0, 0, 255), 2)
   img = cv2.line(img, (xmin_f, ymax_f), (xmin_b, ymax_b), (0, 0, 255), 2)
   img = cv2.line(img, (xmax_f, ymin_f), (xmax_b, ymin_b), (0, 0, 255), 2)
   img = cv2.line(img, (xmax_f, ymax_f), (xmax_b, ymax_b), (0, 0, 255), 2)
   return img

# left_P, right_P = eightPoint()
# left_ = np.hstack((left_P, np.ones((left_P.shape[0], 1))))
# right_ = np.hstack((right_P, np.ones((right_P.shape[0], 1))))
# F = fundamentalMatrix(left_P, right_P)
# F, mask = cv2.findFundamentalMat(left_P, right_P, method=cv2.RANSAC)
F = np.linalg.inv(K_L).T @ E @ np.linalg.inv(K_R)
E = K_R.T @ F @ K_L
U, S, V = np.linalg.svd(E)

img1, img2 = cv2.imread(lefti), cv2.imread(righti)
left = non_dups_left[0]
data = np.array([[], [], []])
for x in range(left[0], left[0]+left[2], 10):
   for y in range(left[1], left[1]+left[3], 10):
      pt = np.array([x, y, 1])
      img1, img2, data = getEpipolarLines(img1, img2, (pt, "LEFT"), F, data)

img = drawOrientedBox(data, img1, left)
cv2.imwrite(os.path.join("data", "3dbox.png"), img)


