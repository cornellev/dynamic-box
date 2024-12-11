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
import edge
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
      # z = getDepthMap(pt, np.hstack((ptr, np.array([1]))))[-1][0]
      x = pt[0] - cx_l * z/fx_l
      y = pt[1] - cy_l * z/fy_l
      # if (z*0.001 > 1.0 and z*0.001 < 2.4):
      data = np.concatenate((data, [[pt[0]], [pt[1]], [z*0.001]]), axis=1)
      # im1 = cv2.circle(img1, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
      # im2 = cv2.putText(img2, str(round(z*0.001,3)), (int(ptr[0]), int(ptr[1])-10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
      # im1 = cv2.putText(img1, str(round(z*0.001,3)), (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
      # im2 = cv2.putText(img2, str(round(z[0],3)), (int(ptr[0]), int(ptr[1])+30), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 2)
      return img1, img2, data

def drawMinMaxBox (data, img, box):
   dict_mask = edge.getEdgeMask(img, box)
   data = np.array([row for row in data.T if dict_mask.get((row[0], row[1]), 0) == 255]).T
   for pt in data.T:
      img = cv2.circle(img, (int(pt[0]), int(pt[1])), 5, (255, 255, 0), -1)
   cv2.imwrite("check.png", img)

   xmin, xmax, ymin, ymax = box[0], box[0]+box[2], box[1], box[1]+box[3]
   data = data[:, np.abs(data[-1, :] - np.median(data, axis = 1)[-1])/np.std(data[-1]) < 3]
   means = np.mean(data, axis = 1)
   data = data[:, (data[-1, :] > means[-1]-3*np.std(data[-1])) & ((data[-1, :] < means[-1]+3*np.std(data[-1])))]
   means = np.mean(data, axis = 1)
   median = np.median(data, axis = 1)

   front_data = data[:, data[-1, :] < median[-1]]
   back_data = data[:, data[-1, :] >= median[-1]]

   means = np.mean(data, axis = 1)
   cov = np.cov(data)
   eigval, eigvec = np.linalg.eig(cov)
   sorted_indices = np.argsort(eigval)[::-1]
   eigval = eigval[sorted_indices]
   eigvec = eigvec[:, sorted_indices]
   print(eigvec)
   print(eigval)
   # eigvec = np.array([eigval[0]*eigvec[0], eigval[1]*eigvec[1], eigval[2]*eigvec[2]])
   print(eigvec)
   centered_data = data - means[:,np.newaxis]
   xmin, xmax, ymin, ymax, zmin, zmax = np.min(centered_data[0, :]), np.max(centered_data[0, :]), np.min(centered_data[1, :]), np.max(centered_data[1, :]), np.min(centered_data[2, :]), np.max(centered_data[2, :])
   fig = plt.figure()
   ax = fig.add_subplot(111, projection='3d')
   ax.scatter(centered_data[0,:], centered_data[1,:], centered_data[2,:], label="original data")
   # ax.scatter(centered_data[0,:], centered_data[1,:], centered_data[2,:], label="centered data")
   ax.legend()
   # eigen basis
   aligned_coords = np.matmul(eigvec.T, centered_data)
   xmin, xmax, ymin, ymax, zmin, zmax = np.min(aligned_coords[0, :]), np.max(aligned_coords[0, :]), np.min(aligned_coords[1, :]), np.max(aligned_coords[1, :]), np.min(aligned_coords[2, :]), np.max(aligned_coords[2, :])

   fig = plt.figure()
   ax = fig.add_subplot(111, projection='3d')
   ax.scatter(aligned_coords[0,:], aligned_coords[1,:], aligned_coords[2,:], color='g', label="rotated/aligned data")
   ax.legend()
   ax.set_xlabel('x')
   ax.set_ylabel('y')
   ax.set_zlabel('z')
   # cartesian basis
   ax.plot([0, 1],  [0, 0], [0, 0], color='b', linewidth=4)
   ax.plot([0, 0],  [0, 1], [0, 0], color='b', linewidth=4)
   ax.plot([0, 0],  [0, 0], [0, 1], color='b', linewidth=4)
   # eigen basis
   ax.plot([0, eigvec[0, 0]],  [0, eigvec[1, 0]], [0, eigvec[2, 0]], color='r', linewidth=4)
   ax.plot([0, eigvec[0, 1]],  [0, eigvec[1, 1]], [0, eigvec[2, 1]], color='g', linewidth=4)
   ax.plot([0, eigvec[0, 2]],  [0, eigvec[1, 2]], [0, eigvec[2, 2]], color='k', linewidth=4)

   rectCoords = lambda x1, y1, z1, x2, y2, z2: np.array([[x1, x1, x2, x2, x1, x1, x2, x2],
                                                      [y1, y2, y2, y1, y1, y2, y2, y1],
                                                      [z1, z1, z1, z1, z2, z2, z2, z2]])

   realigned_coords = np.matmul(eigvec, aligned_coords)
   realigned_coords += means[:, np.newaxis]

   rrc = np.matmul(eigvec, rectCoords(xmin, ymin, zmin, xmax, ymax, zmax))
   rrc += means[:, np.newaxis] 
   fig = plt.figure()
   ax = fig.add_subplot(111, projection='3d')
   ax.scatter(realigned_coords[0,:], realigned_coords[1,:], realigned_coords[2,:], label="rotation and translation undone")
   ax.legend()

   # z1 plane boundary
   print(rrc[0, 0:2], rrc[1, 0:2], rrc[2, 0:2])
   print(rrc[0, 1:3], rrc[1, 1:3], rrc[2, 1:3])
   print(rrc[0, 2:4], rrc[1, 2:4], rrc[2, 2:4])
   print(rrc[0, [3,0]], rrc[1, [3,0]], rrc[2, [3,0]])
   ax.plot(rrc[0, 0:2], rrc[1, 0:2], rrc[2, 0:2], color='b')
   ax.plot(rrc[0, 1:3], rrc[1, 1:3], rrc[2, 1:3], color='b')
   ax.plot(rrc[0, 2:4], rrc[1, 2:4], rrc[2, 2:4], color='b')
   ax.plot(rrc[0, [3,0]], rrc[1, [3,0]], rrc[2, [3,0]], color='b')

   # # z2 plane boundary
   # ax.plot(rrc[0, 4:6], rrc[1, 4:6], rrc[2, 4:6], color='b')
   # ax.plot(rrc[0, 5:7], rrc[1, 5:7], rrc[2, 5:7], color='b')
   # ax.plot(rrc[0, 6:], rrc[1, 6:], rrc[2, 6:], color='b')
   # ax.plot(rrc[0, [7, 4]], rrc[1, [7, 4]], rrc[2, [7, 4]], color='b')

   # # z1 and z2 connecting boundaries
   ax.plot(rrc[0, [0, 4]], rrc[1, [0, 4]], rrc[2, [0, 4]], color='b')
   ax.plot(rrc[0, [1, 5]], rrc[1, [1, 5]], rrc[2, [1, 5]], color='b')
   ax.plot(rrc[0, [2, 6]], rrc[1, [2, 6]], rrc[2, [2, 6]], color='b')
   ax.plot(rrc[0, [3, 7]], rrc[1, [3, 7]], rrc[2, [3, 7]], color='b')
   print((int(rrc[0, 0:2][0]), int(rrc[1, 0:2][0])), (int(rrc[0, 0:2][1]), int(rrc[1, 0:2][1])))
   img = cv2.line(img, (int(rrc[0, 0:2][0]), int(rrc[1, 0:2][0])), (int(rrc[0, 0:2][1]), int(rrc[1, 0:2][1])), (0, 0, 255), 2)
   img = cv2.line(img, (int(rrc[0, 1:3][0]), int(rrc[1, 1:3][0])), (int(rrc[0, 1:3][1]), int(rrc[1, 1:3][1])), (0, 0, 255), 2)
   img = cv2.line(img, (int(rrc[0, 2:4][0]), int(rrc[1, 2:4][0])), (int(rrc[0, 2:4][1]), int(rrc[1, 2:4][1])), (0, 0, 255), 2)
   img = cv2.line(img, (int(rrc[0, [3,0]][0]), int(rrc[1, [3,0]][0])), (int(rrc[0, [3,0]][1]), int(rrc[1, [3,0]][1])), (0, 0, 255), 2)
   img = cv2.line(img, (int(rrc[0, 4:6][0]), int(rrc[1, 4:6][0])), (int(rrc[0, 4:6][1]), int(rrc[1, 4:6][1])), (0, 0, 255), 2)
   img = cv2.line(img, (int(rrc[0, 5:7][0]), int(rrc[1, 5:7][0])), (int(rrc[0, 5:7][1]), int(rrc[1, 5:7][1])), (0, 0, 255), 2)
   img = cv2.line(img, (int(rrc[0, 6:][0]), int(rrc[1, 6:][0])), (int(rrc[0, 6:][1]), int(rrc[1, 6:][1])), (0, 0, 255), 2)
   img = cv2.line(img, (int(rrc[0, [7, 4]][0]), int(rrc[1, [7, 4]][0])), (int(rrc[0, [7, 4]][1]), int(rrc[1, [7, 4]][1])), (0, 0, 255), 2)
   xmin, xmax, ymin, ymax = box[0], box[0]+box[2], box[1], box[1]+box[3]
   img = cv2.rectangle(img, (int(xmin),int(ymin)), (int(xmax),int(ymax)), (0, 0, 255), 2)
   img = cv2.circle(img, (int((xmin+xmax)/2), int((ymin+ymax)/2)), 5, (255, 255, 0), -1)

   plt.show()
   return img

def drawOrientedBox (data, img, box):
   dict_mask = edge.getEdgeMask(img, box)
   data = np.array([row for row in data.T if dict_mask.get((row[0], row[1]), 0) == 255]).T
   xmin, xmax, ymin, ymax = box[0], box[0]+box[2], box[1], box[1]+box[3]
   data = data[:, np.abs(data[-1, :] - np.median(data, axis = 1)[-1])/np.std(data[-1]) < 3]
   means = np.mean(data, axis = 1)
   data = data[:, (data[-1, :] > means[-1]-np.std(data[-1])) & ((data[-1, :] < means[-1]+np.std(data[-1])))]
   means = np.mean(data, axis = 1)
   median = np.median(data, axis = 1)

   front_data = data[:, data[-1, :] < means[-1]]
   back_data = data[:, data[-1, :] > means[-1]]
   xmin_f, xmax_f, ymin_f, ymax_f = int(np.min(front_data[0, :])), int(np.max(front_data[0, :])), int(np.min(front_data[1, :])), int(np.max(front_data[1, :]))
   xmin_b, xmax_b, ymin_b, ymax_b = int(np.min(back_data[0, :])), int(np.max(back_data[0, :])), int(np.min(back_data[1, :])), int(np.max(back_data[1, :]))
   cov = np.cov(back_data)
   eigval, eigvec = np.linalg.eig(cov)
   sorted_indices = np.argsort(eigval)[::-1]
   eigval, eigvec = eigval[sorted_indices], eigvec[:, sorted_indices]
   # img = cv2.line(img, (int((xmin+xmax)/2), int((ymin+ymax)/2)), (int(eigvec[0][0]*eigval[0])+int((xmin+xmax)/2), int(eigvec[0][1]*eigval[0])+int((ymin+ymax)/2)), (0, 0, 255), 2)
   # img = cv2.line(img, (int((xmin+xmax)/2), int((ymin+ymax)/2)), (int(eigvec[1][0]*eigval[1])+int((xmin+xmax)/2), int(eigvec[1][1]*eigval[1])+int((ymin+ymax)/2)), (0, 0, 255), 2)
   # img = cv2.line(img, (int((xmin+xmax)/2), int((ymin+ymax)/2)), (int(eigvec[2][0]*eigval[2])+int((xmin+xmax)/2), int(eigvec[2][1]*eigval[2])+int((ymin+ymax)/2)), (0, 0, 255), 2)
   # img = cv2.line(img, (xmin_f, ymax_f), (xmin_b, ymax_b), (0, 0, 255), 2)
   # img = cv2.line(img, (xmax_f, ymin_f), (xmax_b, ymin_b), (0, 0, 255), 2)
   # img = cv2.line(img, (xmax_f, ymax_f), (xmax_b, ymax_b), (0, 0, 255), 2)
   major_axis = eigvec[:, np.argmax(eigval)]
   yaw = np.arctan2(major_axis[1], major_axis[0])
   pitch = np.arctan2(-major_axis[1], np.sqrt(major_axis[1]**2 + major_axis[2]**2))
   roll = np.arctan2(eigvec[0][1], eigvec[0][2])
   print(np.degrees(yaw))
   print(np.degrees(pitch))
   print(np.degrees(roll))
   # print(xmin)
   # print(xmin, xmax, xmin_b-(np.abs(xmin-xmin_b))*math.sin(yaw%(math.pi/2)), xmax_b)
   # print(((xmax_b-xmin_b))*math.cos(yaw%(math.pi/2)))
   # Positive yaw is clockwise, get projection of x-axis onto yaw degree vector, assume origin is xmax:
   if yaw < 0:
      # Shifts center of back box to the left proportional to yaw, subtract by half of the original width of the back
      # box to get new [xmin_b].
      xmin_b = max(xmin, int((xmin+xmax)/2 - (xmax-(xmin+xmax)/2)*math.sin(-yaw) - (xmax_b-xmin_b)/2))
      # xmin_b = (int(xmin_b-(np.abs(xmax-xmin))*math.sin(yaw%(math.pi/2))))
      # Shifts center of front box to the right proportional to yaw, subtract by half of the original width of the front
      # box to get new [xmax_f].
      xmax_f = min(xmax, int((xmin+xmax)/2 + (xmax-(xmin+xmax)/2)*math.sin(-yaw) + (xmax_f-xmin_f)/2))
      xmax_b = (int(xmax_b-(np.abs(xmax-xmin))*math.sin(-yaw)))
      xmin_f = int(xmin_f+(np.abs(xmax-xmin))*math.sin(-yaw))

   elif yaw > 0:
      xmin_f, xmin_b = xmin_f, xmax_b - int((xmax_b-xmin_b)*math.cos(yaw%(math.pi/2)))
      xmax_f, xmax_b = xmin_f + int((xmax_f-xmin_f)*math.cos(yaw%(math.pi/2))), xmax_b
      
   # positive pitch points upwards
   if pitch > 0:
      # ymax is the bottom horizontal of the box.
      ymin_f, ymin_b = ymin_f, ymax_b - int((ymax_b-ymin_b)*math.cos(pitch))
      ymax_f, ymax_b = ymin_f + int((ymax_f-ymin_f)*math.cos(pitch)), ymax_b

   elif pitch < 0:
      # int((xmin+xmax)/2 + (xmax-(xmin+xmax)/2)*math.sin(yaw%(math.pi/2)) + (xmax_f-xmin_f)/2)
      ymin_f, ymin_b = ymin_f + int((ymax_f-ymin_f)*math.cos(pitch)), ymin_b
      ymax_f, ymax_b = ymax_f, ymax_b - int((ymax_b-ymin_b)*math.cos(pitch))

   img = cv2.rectangle(img, (xmin_f,ymin_f), (xmax_f,ymax_f), (0, 0, 255), 2)
   img = cv2.rectangle(img, (xmin_b,ymin_b), (xmax_b,ymax_b), (255, 255, 0), 2)
   img = cv2.line(img, (xmin_f, ymin_f), (xmin_b, ymin_b), (0, 0, 255), 2)
   img = cv2.line(img, (xmin_f, ymax_f), (xmin_b, ymax_b), (0, 0, 255), 2)
   img = cv2.line(img, (xmax_f, ymin_f), (xmax_b, ymin_b), (0, 0, 255), 2)
   img = cv2.line(img, (xmax_f, ymax_f), (xmax_b, ymax_b), (0, 0, 255), 2)
   return img
F = np.linalg.inv(K_L).T @ E @ np.linalg.inv(K_R)
E = K_R.T @ F @ K_L
U, S, V = np.linalg.svd(E)

print("RUN")
img1, img2 = cv2.imread(lefti), cv2.imread(righti)
left = non_dups_left[2]
data = np.array([[], [], []])
for x in range(left[0], left[0]+left[2], 10):
   for y in range(left[1], left[1]+left[3], 10):
      pt = np.array([x, y, 1])
      img1, img2, data = getEpipolarLines(img1, img2, (pt, "LEFT"), F, data)

cv2.imwrite("lil.png", img1)
# img = drawOrientedBox(data, img1, left)
img = drawMinMaxBox(data, img1, left)
cv2.imwrite(os.path.join("data", "3dbox.png"), img)


