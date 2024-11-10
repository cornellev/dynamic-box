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

# Given essential matrix E, get R and T using SVD, where R = U @ W.T @ V
# and W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

# lefti = "wagner_left.png"
# righti = "wagner_right.png"
lefti = "left0.png"
righti = "right0.png"

# Use this to get points to draw epipolar lines. Not for 8-point algorithm.
non_dups_left, non_dups_right = YOLO.get_pair([lefti, righti])

# Given the bounding boxes for each image: do 8 point algorithm for corresponding images in each bounding box.
def eight_point ():
   # Step 2: Initialize SIFT detector
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

   # for corr in (np.float32([kp1[m.queryIdx].pt for m in good_matches])):
   #    x, y = corr
   left_P = np.float32([kp1[m.queryIdx].pt for m in good_matches])
   right_P = np.float32([kp2[m.trainIdx].pt for m in good_matches])
   # left_P = np.concatenate((left_P, np.float32([kp1[m.queryIdx].pt for m in good_matches])))
   # right_P = np.concatenate((right_P, np.float32([kp2[m.trainIdx].pt for m in good_matches])))
   # return np.hstack((left_P, np.ones((points.shape[0], 1)))), np.hstack((right_P, np.ones((points.shape[0], 1))))
   return left_P, right_P

img1 = cv2.imread('left0.png', cv2.IMREAD_GRAYSCALE)  #queryimage # left image
img2 = cv2.imread('right0.png', cv2.IMREAD_GRAYSCALE) #trainimage # right image
 
# Normalize left, right corresponding points.
def normalize_points(points):
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
   left, T1 = normalize_points(left_P)
   right, T2 = normalize_points(right_P)
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

def draw_epipolar_lines(img1, img2, pts1, pts2, F):
   U, S, V = svd(F)
   # e is epipole.
   e = S
   # Convert the points to homogeneous coordinates
   pts1_homogeneous = np.hstack([pts1, np.ones((pts1.shape[0], 1))])
   pts2_homogeneous = np.hstack([pts2, np.ones((pts2.shape[0], 1))])
   # Compute epipolar lines in the second image for points in the first image
   lines2 = np.dot(F, pts1_homogeneous.T).T  # Lines in image 2 for points in image 1

   # Compute epipolar lines in the first image for points in the second image
   lines1 = np.dot(F.T, pts2_homogeneous.T).T  # Lines in image 1 for points in image 2

   # Draw line from epipole to point.

   # Draw the lines on the images
   l_count = 0
   for r, _, pt2 in zip(lines2, pts1, pts2):
      # Line in image 2
      x0, y0 = e[:2]
      x1, y1 = pt2
      # x0, y0 = map(int, [0, -r[2] / r[1]])
      # x1, y1 = map(int, [img2.shape[1], -(r[2] + r[0] * img2.shape[1]) / r[1]])
      img2 = cv2.line(img2, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 1)

      # Draw the corresponding points
      x, y = pt2
      cv2.putText(img2, str(l_count), (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1.5, 2)
      l_count = l_count + 1
      img2 = cv2.circle(img2, (int(x), int(y)), 5, (0, 0, 255), -1)

   r_count = 0
   for r, pt1, _ in zip(lines1, pts1, pts2):
      # Line in image 1
      x0, y0 = map(int, [0, -r[2] / r[1]])
      x1, y1 = map(int, [img1.shape[1], -(r[2] + r[0] * img1.shape[1]) / r[1]])
      img1 = cv2.line(img1, (x0, y0), (x1, y1), (0, 255, 0), 1)

      x, y = pt1
      # Draw the corresponding points
      cv2.putText(img1, str(r_count), (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1.5, 2)
      r_count = r_count + 1
      img1 = cv2.circle(img1, (int(x), int(y)), 5, (0, 0, 255), -1)
   
   for i in range(min(len(pts1), len(pts2))):
      depth = (Baseline/100 * fx_l/10)/(pts1[i][0]-pts2[i][0])
      x, y = pts1[i][0], pts1[i][1]
      cv2.putText(img1, str(depth), (x-10, y-50), cv2.FONT_HERSHEY_PLAIN, 1.5, 2)
   print("worked")
   return img1, img2

def drawLines(img, lines):
   _, c, _ = img.shape
   for r in lines:
      x0, y0 = map(int, [0, -r[2]/r[1]])
      x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
      cv2.line(img, (x0, y0), (x1, y1), (0, 0, 255), 1)
   return img

# Find point on right image [u_r, v_r, 1] such that a*u_r + b*v_r + c = 0.
def getEpipolarLines (img1, img2, pt, F):
   pt, dir = pt
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
   im1 = cv2.circle(img1, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
   im2 = cv2.line(img2, (0, int(y(0))), (img1.shape[1], int(y(img1.shape[1]))), (0, 0, 255), 1)
   return img1, img2


left_P, right_P = eight_point()

left_ = np.hstack((left_P, np.ones((left_P.shape[0], 1))))
right_ = np.hstack((right_P, np.ones((right_P.shape[0], 1))))
F = fundamentalMatrix(left_P, right_P)
F, mask = cv2.findFundamentalMat(left_P, right_P, method=cv2.RANSAC)
# F = np.linalg.inv(K_L).T @ E @ np.linalg.inv(K_R)
E = K_R.T @ F @ K_L
U, S, V = np.linalg.svd(E)

print(left_[0])
img1, img2 = getEpipolarLines(cv2.imread(lefti), cv2.imread(righti), (left_[0], "LEFT"), F)
cv2.imwrite(os.path.join("data", "leftepip.png"), img1)
cv2.imwrite(os.path.join("data", "rightepip.png"), img2)
# U @ W @ V to get the rotation matrix R.


# HL = cv2.findHomography(left_P, right_P)[0]
# HR = cv2.findHomography(right_P, left_P)[0]

# size = cv2.imread(lefti).shape[:2]

# left_rectified = cv2.warpPerspective(cv2.imread(lefti), HL, (size[1],size[0]))
# right_rectified = cv2.warpPerspective(cv2.imread(righti), HR, (size[1],size[0]))


# # # Display the rectified images
# cv2.imwrite(os.path.join("data", "leftrectified.png"), left_rectified)
# cv2.imwrite(os.path.join("data", "rightrectified.png"), right_rectified)

# left_P = []
# right_P = []
# for left in non_dups_left:
#    left_P.append([left[0],left[1]])
#    # left_P.append([left[0]+left[2],left[1]])
#    # left_P.append([left[0],left[1]])
#    # left_P.append([left[0],left[1]+left[3]])

# for right in non_dups_right:
#    right_P.append([right[0],right[1]])
#    # right_P.append([right[0]+right[2],right[1]])
#    # right_P.append([right[0],right[1]])
#    # right_P.append([right[0],right[1]+right[3]])

# left_P = np.array(left_P)
# right_P = np.array(right_P)
# print(left_P, right_P)

# epiline_l, epiline_r = draw_epipolar_lines(cv2.imread("boundedleft0.png"), cv2.imread("boundedright0.png"), left_P, right_P, F)
# cv2.imwrite(os.path.join("data", "boundepip_l.png"), epiline_l)
# cv2.imwrite(os.path.join("data", "boundepip_r.png"), epiline_r)

# epiline_l, epiline_r = draw_epipolar_lines(cv2.imread("boundedleft0.png"), cv2.imread("boundedright0.png"), left_P, right_P, F)

# # # Display the images with epipolar lines
# cv2.imwrite(os.path.join("data", "epip_l.png"), epiline_l)
# cv2.imwrite(os.path.join("data", "epip_r.png"), epiline_r)

# epilinesR = cv2.computeCorrespondEpilines(right_P.reshape(-1, 1, 2), 2, F)
# epilinesR = epilinesR.reshape(-1, 3)

# lefti = "boundedleft0.png"
# righti = "boundedright0.png"
# F = np.dot(np.dot(HR.T, F), HL)

# pts1 = np.hstack((left_P, np.ones((left_P.shape[0], 1))))
# pts2 = np.hstack((right_P, np.ones((right_P.shape[0], 1))))
# left_P = cv2.perspectiveTransform(pts1.reshape(-1, 1, 2), HL).reshape(-1, 2)
# right_P = cv2.perspectiveTransform(pts2.reshape(-1, 1, 2), HR).reshape(-1, 2)

# Verify F is calculated correctly -> recitfy the image -> generate a depth map / point cloud via stereo matching -> get 3D points
# left_P = []
# right_P = []

# for left in non_dups_left:
#    left_P.append([left[0],left[1]])
#    left_P.append([left[0]+left[2],left[1]])
#    left_P.append([left[0],left[1]])
#    left_P.append([left[0],left[1]+left[3]])

# for right in non_dups_right:
#    right_P.append([right[0],right[1]])
#    right_P.append([right[0]+right[2],right[1]])
#    right_P.append([right[0],right[1]])
#    right_P.append([right[0],right[1]+right[3]])

# left_P = np.array(left_P)
# right_P = np.array(right_P)


# epilinesR = cv2.computeCorrespondEpilines(right_P.reshape(-1, 1, 2), 2, F)
# epilinesR = epilinesR.reshape(-1, 3)
# lef = drawLines(cv2.imread(lefti), epilinesR)

# # find epilines corresponding to points in left image and draw them on the right image
# epilinesL = cv2.computeCorrespondEpilines(left_P.reshape(-1, 1, 2), 1, F)
# epilinesL = epilinesL.reshape(-1, 3)
# igh = drawLines(cv2.imread(righti), epilinesL)

# epiline_l, epiline_r = draw_epipolar_lines(cv2.imread("boundedleft0.png"), cv2.imread("boundedright0.png"), left_P, right_P, F)
# cv2.imwrite(os.path.join("data", "boundepip_l.png"), epiline_l)
# cv2.imwrite(os.path.join("data", "boundepip_r.png"), epiline_r)