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
# lefti = "wagner_left.png"
# righti = "wagner_right.png"
lefti = "left.png"
righti = "right.png"
np.set_printoptions(suppress=True)
# YOLO.get_pair([lefti, righti])

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

left_P, right_P = eight_point()

def drawLines(img, lines):
   _, c, _ = img.shape
   for r in lines:
      x0, y0 = map(int, [0, -r[2]/r[1]])
      x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
      cv2.line(img, (x0, y0), (x1, y1), (0, 0, 255), 1)
   return img

F = fundamentalMatrix(left_P, right_P)
left_ = np.hstack((left_P, np.ones((left_P.shape[0], 1))))
right_ = np.hstack((right_P, np.ones((right_P.shape[0], 1))))

# F, mask = cv2.findFundamentalMat(left_P, right_P, method=cv2.RANSAC)
epiline_l, epiline_r = draw_epipolar_lines(cv2.imread(lefti), cv2.imread(righti), left_P, right_P, F)

# Display the images with epipolar lines
cv2.imwrite(os.path.join("data", "epip_l.png"), epiline_l)
cv2.imwrite(os.path.join("data", "epip_r.png"), epiline_r)

U, S, V = svd(F)
el, d1, d2 = np.array([[0,0,1], [0,1,0], [1,0,0]]) @ V
er, s1, s2 = np.array([[0,0,1], [0,1,0], [1,0,0]]) @ U.T
el1, el2, el3 = el
er1, er2, er3 = er
v = S[1]/S[0]

HR = np.array([er, s1, np.sqrt(v)*s2]).T
HL = np.array([el, np.sqrt(v)*d2, -d1])

elx = np.array([[0,-el3,el2], [el3,0,-el1], [-el2,el1,0]])
erx = np.array([[0,-er3,er2], [er3,0,-er1], [-er2,er1,0]])

HL = cv2.findHomography(left_P, right_P)[0]
HR = cv2.findHomography(right_P, left_P)[0]

size = cv2.imread(lefti).shape[:2]

left_rectified = cv2.warpPerspective(cv2.imread(lefti), HL, (size[1],size[0]))
right_rectified = cv2.warpPerspective(cv2.imread(righti), HR, (size[1],size[0]))

# # Display the rectified images
cv2.imwrite(os.path.join("data", "leftrectified.png"), left_rectified)
cv2.imwrite(os.path.join("data", "rightrectified.png"), right_rectified)

lefti = "data/leftrectified.png"
righti = "data/rightrectified.png"
F = np.dot(np.dot(HR.T, F), HL)

pts1 = np.hstack((left_P, np.ones((left_P.shape[0], 1))))
pts2 = np.hstack((right_P, np.ones((right_P.shape[0], 1))))
left_P = cv2.perspectiveTransform(pts1.reshape(-1, 1, 2), HL).reshape(-1, 2)
right_P = cv2.perspectiveTransform(pts2.reshape(-1, 1, 2), HR).reshape(-1, 2)

# Verify F is calculated correctly -> recitfy the image -> generate a depth map / point cloud via stereo matching -> get 3D points
epilinesR = cv2.computeCorrespondEpilines(right_P.reshape(-1, 1, 2), 2, F)
epilinesR = epilinesR.reshape(-1, 3)
lef = drawLines(cv2.imread(lefti), epilinesR)

# find epilines corresponding to points in left image and draw them on the right image
epilinesL = cv2.computeCorrespondEpilines(left_P.reshape(-1, 1, 2), 1, F)
epilinesL = epilinesL.reshape(-1, 3)
igh = drawLines(cv2.imread(righti), epilinesL)

cv2.imwrite(os.path.join("data", "epip_lrec.png"), lef)
cv2.imwrite(os.path.join("data", "epip_rrec.png"), igh)