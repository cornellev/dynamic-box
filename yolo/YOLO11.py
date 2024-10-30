"""
1) Generate individual 2D boxes for left and right stereo images.
   Use left.png and right.png as test stereo image.
"""

"""
Proposed Regions with YOLOv3: use YOLOv3 to generate 2D bounding boxes 
for all objects in an image.

 1. Input image is divided into NxN grid cells. For each object present on image, one grid cell is responsible for predicting object.

2. Each grid predicts [B] bounding box and [C] class probabilities. And bounding box consist of 5 components (x,y,w,h,confidence)

(x,y) = coordinates representing center of box

(w,h) = width and height of box

Confidence = represents presence/absence of any object
"""


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
import ultralytics
import threading

classes = []
pool = []
pair = []

def yolo11(img):
   model = ultralytics.YOLO("yolo11n.pt")
   global classes
   # [classes] stores all names of different objects in [coco.names] 
   # that the coco model has been trained to identify 
   with open("coco.names", "r") as f:
      classes = f.read().splitlines()
   return model(img, stream = True)

def bounding_box_dim (of, img):
   boxes = []
   confs = []
   class_ids = []

   for output in yolo11(img):
      for box in output.boxes:
         class_id = int(box.cls[0])
         conf = math.ceil(box.conf[0]*100)/100
         if conf > 0.3:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            boxes.append([x1,y1,x2-x1,y2-y1])
            confs.append(float(conf))
            class_ids.append(class_id)

   colors = np.random.uniform(0, 255, size=(len(boxes), 3))
   non_dups = NMSBoxes(boxes, confs, 0.4, 0.2)
   for i in non_dups: 
      x, y, w, h = boxes[i]
      label = str(classes[class_ids[i]])
      cv2.rectangle(img, (x,y), (x+w,y+h), colors[i], 2)
      cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1.5, colors[i], 2)
   
   pair.append([of, non_dups])
   return pair

def NMSBoxes(boxes, confs, score_thres, nms_thres):
   nondup = []
   indices = sorted(range(len(confs)), key=lambda i: confs[i], reverse = True)
   while len(indices) != 0 and confs[indices[0]] >= score_thres:
      nondup.append(indices.pop(0))
      x1, y1, w1, h1 = boxes[nondup[-1]]
      i = 0
      while i != len(indices):   
         # print ("at " + str(indices[i]))     
         x2, y2, w2, h2 = boxes[indices[i]]
         intersect = ((min(x1+w1, x2+w2) - max(x1, x2)) * (min(y1+h1, y2+h2) - max(y1, y2)) 
                     if (x2+w2 > x1 and x1+w1 > x2 and y2+h2 > y1 and y1+h1 > y2) else 0)
         jaccard = intersect / (w1*h1 + w2*h2 - intersect)
         if jaccard >= nms_thres:
            indices.remove(indices[i]) 
         else: i = i + 1
   return nondup

def get_pair (images):
   global pair
   for img in images:
      pool.append(threading.Thread(target = bounding_box_dim, args = ("bounded" + img, cv2.imread(img), )))
      pool[-1].start()

   for thread in pool:
      thread.join()
   
   if pair[0][0] == "boundedright.png":
      pair.reverse()
   print(pair)
   pair = []

get_pair(["left.png", "right.png"])
get_pair(["left.png", "right.png"])
get_pair(["left.png", "right.png"])
get_pair(["left.png", "right.png"])
get_pair(["left.png", "right.png"])
get_pair(["left.png", "right.png"])

