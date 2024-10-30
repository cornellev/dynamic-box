"""
Implement YOLO without the object classification task: any obstacle is an obstacle:
try cv2's selective search: proposes all regions in an image -> associate with obstacle. """

""" 
After everything works, change YOLO.py into a publisher node with name : "yolo" that takes in
ZED stereo images and publishes left, right annotated bbox images (more specifically triple (boxes, confs, class_ids)).
"""

import cv2
import numpy as np
import math
import threading
import matplotlib.pyplot as plt

classes = []
non_dups_left = []
non_dups_right = []
pool = []
pair = []
left = []
right = []
left_P = np.empty((1,2))
right_P = np.empty((1,2))

# Non-maximum supression.
def NMSBoxes(boxes, confs, score_thres, nms_thres):
   nondup = []
   indices = sorted(range(len(confs)), key=lambda i: confs[i], reverse = True)
   while len(indices) != 0 and confs[indices[0]] >= score_thres:
      nondup.append(indices.pop(0))
      x1, y1, w1, h1 = boxes[nondup[-1]]
      i = 0
      while i != len(indices):      
         x2, y2, w2, h2 = boxes[indices[i]]
         intersect = ((min(x1+w1, x2+w2) - max(x1, x2)) * (min(y1+h1, y2+h2) - max(y1, y2)) 
                     if (x2+w2 > x1 and x1+w1 > x2 and y2+h2 > y1 and y1+h1 > y2) else 0)
         jaccard = intersect / (w1*h1 + w2*h2 - intersect)
         if jaccard >= nms_thres:
            indices.remove(indices[i]) 
         else: i = i + 1
   return nondup

def reorder_boxes (): 
   global non_dups_right
   non_dups_r = []
   for left in non_dups_left:
      x1, y1, w1, h1 = left
      index_l = non_dups_left.index(left)
      non_dups_r.append(left)
      max_IOU = 0
      for right in non_dups_right:
         x2, y2, w2, h2 = right
         intersect = ((min(x1+w1, x2+w2) - max(x1, x2)) * (min(y1+h1, y2+h2) - max(y1, y2)) 
                        if (x2+w2 > x1 and x1+w1 > x2 and y2+h2 > y1 and y1+h1 > y2) else 0)
         jaccard = intersect / (w1*h1 + w2*h2 - intersect)
         if max_IOU < jaccard:
            max_IOU = jaccard
            non_dups_r[-1] = right
      non_dups_right.remove(non_dups_r[-1])
   non_dups_right = non_dups_r

def yolo(img):
   # Read weights file (contains pretrained weights which 
   # has been trained on coco dataset) and configuration file 
   # (has YOLOv3 network architecture)
   # [cv2.dnn.readNet] loads the pre-trained YOLO deep learning model.
   net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
   
   global classes
   # [classes] stores all names of different objects in [coco.names] 
   # that the coco model has been trained to identify 
   with open("coco.names", "r") as f:
      classes = f.read().splitlines()
   
   # Get the name of all layers of the network, then pass to forward pass.
   blob = cv2.dnn.blobFromImage(img, 1/255, (320, 320), (0,0,0), swapRB=True, crop=False)
   net.setInput(blob)

   # Perform forward propogation through OpenCV's DNN: the input image blob is
   # passed through layers of the neural network [networkarch.png] to get output
   # predictions.
   output_layers = net.getUnconnectedOutLayersNames()
   layer_outputs = net.forward(output_layers)
   return layer_outputs

def bounding_box_dim (of, img):
   boxes = []
   confs = []
   class_ids = []
   global non_dups_left, non_dups_right

   for output in yolo(img):
      for detect in output:
         scores = detect[5:]
         class_id = np.argmax(scores)
         height, width = img.shape[:2]
         conf = scores[class_id]
         # If image confidence is > 0.3, add bounding box and classification.
         if conf > 0.3:
            center_x = int(detect[0] * width)
            center_y = int(detect[1] * height)
            w = int(detect[2] * width)
            h = int(detect[3] * height)
            # (x,y) are top left coordinates of the bounding box.
            x = int(center_x - w/2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confs.append(float(conf))
            class_ids.append(class_id)

   # Performs non-maximum suppression on duplicate bounding boxes over the same
   # object, keeping only the bounding boxes that have the highest confidence.
   non_dups = NMSBoxes(boxes, confs, 0.5, 0.2)
   colors = np.random.uniform(0, 255, size=(len(boxes), 3))
   for i in non_dups: 
      x, y, w, h = boxes[i]
      label = str(classes[class_ids[i]])
      cv2.rectangle(img, (x,y), (x+w,y+h), 2)
      cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1.5, colors[i], 2)
      if (of == "boundedright.png"):
         non_dups_right.append(boxes[i])
      if (of == "boundedleft.png"):
         non_dups_left.append(boxes[i])
   
   pair.append([of, non_dups])
   return pair


def get_pair (images):
   global pair
   for img in images:
      pool.append(threading.Thread(target = bounding_box_dim, args = ("bounded" + img, cv2.imread(img), )))
      pool[-1].start()

   for thread in pool:
      thread.join()
   
   if pair[0][0] == "boundedright.png":
      pair.reverse()
   # Send pair to subscriber.

   reorder_boxes()

   for i in range(len(non_dups_left)):
      left.append(cv2.imread("left.png")[non_dups_left[i][1]:non_dups_left[i][1]+non_dups_left[i][3], non_dups_left[i][0]:non_dups_left[i][0]+non_dups_left[i][2]])
      right.append(cv2.imread("left.png")[non_dups_right[i][1]:non_dups_right[i][1]+non_dups_right[i][3], non_dups_right[i][0]:non_dups_right[i][0]+non_dups_right[i][2]])
   return pair
   # After pair is sent -> pair = []

get_pair(["left.png", "right.png"])

# left = cv2.imread("left.png")
# right = cv2.imread("right.png")
# cv2.imshow("left", left)
# cv2.waitKey(0)

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
   
   # Step 3: Detect keypoints and compute descriptors
   l_img = cv2.imread("left.png")
   r_img = cv2.imread("right.png")
   kp1, des1 = sift.detectAndCompute(l_img, None)
   kp2, des2 = sift.detectAndCompute(r_img, None)

   # Step 4: Match keypoints using BFMatcher
   bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
   matches = bf.match(des1, des2)

   # Step 5: Sort matches by distance
   matches = sorted(matches, key=lambda x: x.distance)

   # Step 6: Extract the best 8 matches
   good_matches = matches[:8]

   # Draw matches
   matched_img = cv2.drawMatches(l_img, kp1, r_img, kp2, good_matches, None, flags=2)

   # Step 7: Extract corresponding keypoints
   # for corr in (np.float32([kp1[m.queryIdx].pt for m in good_matches])):
   #    x, y = corr
   left_P = np.float32([kp1[m.queryIdx].pt for m in good_matches])
   right_P = np.float32([kp2[m.trainIdx].pt for m in good_matches])
   # left_P = np.concatenate((left_P, np.float32([kp1[m.queryIdx].pt for m in good_matches])))
   # right_P = np.concatenate((right_P, np.float32([kp2[m.trainIdx].pt for m in good_matches])))
   # return np.hstack((left_P, np.ones((points.shape[0], 1)))), np.hstack((right_P, np.ones((points.shape[0], 1))))

   # Step 8: Display or plot the results
   # plt.figure(figsize=(15, 10))
   # plt.imshow(matched_img)
   # plt.show()

   return left_P, right_P


# Normalize left, right corresponding points.
def normalize_points(points):
   """Normalize image points by translating and scaling."""
   x_, y_ = np.mean(points, axis=0)
   d = np.mean(np.sqrt(np.sum((points-np.array([x_, y_])) ** 2, axis = 1)))
   s = np.sqrt(2)/d
   T = np.array([[s, 0, -s * x_],
                 [0, s, -s * y_],
                 [0, 0, 1]])
   P = np.dot(T, np.hstack((points, np.ones((points.shape[0], 1)))).T).T
   return P, T

def fundamentalMatrix(left_P, right_P):
   A = np.zeros((8,9))
   left_P, T1 = normalize_points(left_P)
   right_P, T2 = normalize_points(right_P)
   for i in range(8):
      x1, y1 = left_P[i, 0], left_P[i, 1]
      x2, y2 = right_P[i, 0], right_P[i, 1]
      A[i] = [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1]
   U, S, V = np.linalg.svd(A)
   F = V[-1].reshape(3, 3)
   # Enforce the rank-2 constraint on F using SVD
   U, S, V = np.linalg.svd(F)
   S[-1] = 0  # Set the smallest singular value to zero
   F = U @ np.diag(S) @ V
   return T2.T @ F @ T1

def depth(left, right):
   l = cv2.imread("left.png")
   r = cv2.imread("right.png")
   for i in range(len(left)):
      Z = (2.8 * 120) / (left[i][0] - right[i][0])
      X_L, Y_L = left[i][0]-(665.465*Z/700.819), left[i][1]-(371.953*Z/700.819) 
      X_H, Y_H = (left[i][0]+left[i][2])-(665.465*Z/700.819), (left[i][1]+left[i][3])-(371.953*Z/700.819) 
      # X_L, X_H, Y_L, Y_H = int(X_L), int(X_H), int(Y_L), int(Y_H)
      # cv2.circle(r, (X_L, Y_L), 1, color=(255, 255, 255), thickness=3)
      # cv2.circle(r, (X_L, Y_H), 1, color=(255, 255, 255), thickness=3)
      # cv2.circle(r, (X_H, Y_L), 1, color=(255, 255, 255), thickness=3)
      # cv2.circle(r, (X_H, Y_H), 1, color=(255, 255, 255), thickness=3)
      cv2.putText(l, str(Z), (left[i][0], left[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=0.5, color=(255, 25, 205), thickness=1)
      cv2.putText(r, str(Z), (right[i][0], right[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=0.5, color=(255, 25, 205), thickness=1)

   cv2.imshow("left", l)
   cv2.waitKey(0)
   cv2.imshow("right", r)
   cv2.waitKey(0)
# print(non_dups_left)
left_P, right_P = eight_point(non_dups_left, non_dups_right)
# depth(non_dups_left, non_dups_right)
# Test: [6.31.51587, 64.375595, 1].T(F)[633.0455, 64.169464, 1] = 0
F = fundamentalMatrix(left_P, right_P)

U, S, V = np.linalg.svd(F)

e = V[-1]

# left_P, right_P = np.hstack((left_P, np.ones((left_P.shape[0], 1)))), np.hstack((right_P, np.ones((right_P.shape[0], 1))))

print(F)
depth(non_dups_left, non_dups_right)
# print(right_P[0].T @ F @ left_P[0])
# print(left_P[0].T @ F.T @ right_P[0])

# print(right_P[1].T@F@left_P[1])
# print(right_P[0].T@F@left_P[2])
# print(right_P[1].T@F@left_P[3])
# print(right_P[0].T@F@left_P[4])
# print(right_P[1].T@F@left_P[5])
# print(right_P[0].T@F@left_P[6])
# print(right_P[1].T@F@left_P[7])

# print(right_P[0])
# print(right_P[0].T)
# print(fundamentalMatrix(left_P, right_P))
# print(fundamentalMatrix(left_P, right_P).T)

# Find the fundamental matrix using the 8-point algorithm
F, mask = cv2.findFundamentalMat(left_P, right_P, method=cv2.RANSAC)
left_P = left_P[mask.ravel() == 1]
right_P = right_P[mask.ravel() == 1]
boo, h1, h2 = cv2.stereoRectifyUncalibrated(left_P, right_P, F, cv2.imread("left.png").shape[:2])

# fundamental_matrix, mask = cv2.findFundamentalMat(left_P, right_P, method=cv2.FM_8POINT, ransacReprojThreshold=3., confidence=0.99)

# # Output the fundamental matrix
# print("Fundamental Matrix:")
# print(fundamental_matrix)
