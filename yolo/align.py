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

non_dups_left = []
non_dups_right = []

YOLO.get_pair(["left.png", "right.png"])

# Given the bounding boxes for each image: do 8 point algorithm for corresponding images in each bounding box.
def eight_point (left, right):
   # Step 2: Initialize SIFT detector
   global left_P, right_P
   sift = cv2.SIFT_create()

   # For finding 8 corresponding keypoints for each bounding box.
   # for i in range(len(left)):
   #    x_l, y_l, w_l, h_l = left[i]
   #    x_r, y_r, w_r, h_r = right[i]

   #    l = cv2.imread("left.png")
   #    l_img = l[min(0,y_l):max(l.shape[0],y_l+h_l), min(0,x_l):max(l.shape[1],x_l+w_l)]
   #    r = cv2.imread("right.png")
   #    r_img = r[min(0,y_l):max(r.shape[0],y_l+h_l), min(0,x_l):max(r.shape[1],x_l+w_l)]
   
   l_img = cv2.imread("left.png")
   r_img = cv2.imread("right.png")
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

# Normalize left, right corresponding points.
def normalize_points(points):
   """Normalize image points by translating and scaling."""
   x_, y_ = np.mean(points, axis=0)
   d = np.mean(np.linalg.norm(points-np.array([x_, y_]), axis = 1))
   s = np.sqrt(2)/d
   T = np.array([[s, 0, -s * x_],
                 [0, s, -s * y_],
                 [0, 0, 1]])
   P = ((points-np.array([x_, y_]))*s)[:, :2]
   return P, T

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

def depth(left, right):
   l = cv2.imread("left.png")
   r = cv2.imread("right.png")
   for i in range(len(left)):
      Z = (2.12 * 0.120) / (left[i][0] - right[i][0])
      X_L, Y_L = left[i][0]-(665.465*Z/700.819), left[i][1]-(371.953*Z/700.819) 
      X_H, Y_H = (left[i][0]+left[i][2])-(665.465*Z/700.819), (left[i][1]+left[i][3])-(371.953*Z/700.819) 
      cv2.putText(l, str(Z), (left[i][0], left[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 
               fontScale=0.5, color=(255, 25, 205), thickness=1)
      cv2.putText(r, str(Z), (right[i][0], right[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 
               fontScale=0.5, color=(255, 25, 205), thickness=1)

   cv2.imwrite(os.path.join("data", 'leftdepth.jpg'), l)
   cv2.imwrite(os.path.join("data", 'rightdepth.jpg'), r)

def draw_epipolar_lines(img1, img2, pts1, pts2, F):
    # Convert the points to homogeneous coordinates
    pts1_homogeneous = np.hstack([pts1, np.ones((pts1.shape[0], 1))])
    pts2_homogeneous = np.hstack([pts2, np.ones((pts2.shape[0], 1))])

    # Compute epipolar lines in the second image for points in the first image
    lines2 = np.dot(F, pts1_homogeneous.T).T  # Lines in image 2 for points in image 1

    # Compute epipolar lines in the first image for points in the second image
    lines1 = np.dot(F.T, pts2_homogeneous.T).T  # Lines in image 1 for points in image 2

    # Draw the lines on the images
    for r, pt1, pt2 in zip(lines2, pts1, pts2):
        # Line in image 2
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [img2.shape[1], -(r[2] + r[0] * img2.shape[1]) / r[1]])
        img2 = cv2.line(img2, (x0, y0), (x1, y1), (0, 255, 0), 1)

        # Draw the corresponding points
        x, y = pt2
        img2 = cv2.circle(img2, (int(x), int(y)), 5, (0, 0, 255), -1)

    for r, pt1, pt2 in zip(lines1, pts1, pts2):
        # Line in image 1
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [img1.shape[1], -(r[2] + r[0] * img1.shape[1]) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), (0, 255, 0), 1)

        x, y = pt1
        # Draw the corresponding points
        img1 = cv2.circle(img1, (int(x), int(y)), 5, (0, 0, 255), -1)

    return img1, img2

left_P, right_P = eight_point(non_dups_left, non_dups_right)
F = fundamentalMatrix(left_P, right_P)
epiline_l, epiline_r = draw_epipolar_lines(cv2.imread("left.png"), cv2.imread("right.png"), left_P, right_P, F)

# Display the images with epipolar lines
cv2.imwrite(os.path.join("data", "epip_l.png"), epiline_l)
cv2.imwrite(os.path.join("data", "epip_r.png"), epiline_r)

U, S, V = np.linalg.svd(F)
e = V[-1]

# print(right_[1].T@F@left_[1])
# print(right_[2].T@F@left_[2])
# print(right_[3].T@F@left_[3])
# print(right_[4].T@F@left_[4])
# print(right_[5].T@F@left_[5])
# print(right_[6].T@F@left_[6])
# print(right_[7].T@F@left_[7])

size = cv2.imread("left.png").shape[:2]
_, h1, h2 = cv2.stereoRectifyUncalibrated(left_P, right_P, F, size)

left_rectified = cv2.warpPerspective(cv2.imread("left.png"), h1, size)
right_rectified = cv2.warpPerspective(cv2.imread("right.png"), h2, size)

# Display the rectified images
cv2.imwrite(os.path.join("data", "leftrectified.png"), left_rectified)
cv2.imwrite(os.path.join("data", "rightrectified.png"), right_rectified)

# Compare F with fundamental matrix from OpenCV's 8-point algorithm
F, mask = cv2.findFundamentalMat(left_P, right_P, method=cv2.RANSAC)
img1_with_lines, img2_with_lines = draw_epipolar_lines(cv2.imread("left.png"), cv2.imread("right.png"), left_P, right_P, F)

cv2.imwrite(os.path.join("data", "epip_l1.png"), img1_with_lines)
cv2.imwrite(os.path.join("data", "epip_r1.png"), img2_with_lines)
size = cv2.imread("left.png").shape[:2]
_, h1, h2 = cv2.stereoRectifyUncalibrated(left_P, right_P, F, size)
left_rectified = cv2.warpPerspective(cv2.imread("left.png"), h1, size)
right_rectified = cv2.warpPerspective(cv2.imread("right.png"), h2, size)
cv2.imwrite(os.path.join("data", "leftrectified1.png"), left_rectified)
cv2.imwrite(os.path.join("data", "rightrectified1.png"), right_rectified)
