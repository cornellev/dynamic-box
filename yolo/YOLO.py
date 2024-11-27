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
      cv2.rectangle(img, (x,y), (x+w,y+h), colors[i], 2)
      cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1.5, colors[i], 2)
      if (of == "right0.png"):
         non_dups_right.append(boxes[i])
      if (of == "left0.png"):
         non_dups_left.append(boxes[i])
   
   cv2.imwrite("bounded" + of, img)
   
   pair.append([of, non_dups])
   return pair


def get_pair (images):
   global pair
   for img in images:
      pool.append(threading.Thread(target = bounding_box_dim, args = (img, cv2.imread(img), )))
      pool[-1].start()

   for thread in pool:
      thread.join()
   
   if pair[0][0] == "boundedright.png":
      pair.reverse()
   # Send pair to subscriber.

   reorder_boxes()

   for i in range(len(non_dups_left)):
      left.append(cv2.imread("left0.png")[non_dups_left[i][1]:non_dups_left[i][1]+non_dups_left[i][3], non_dups_left[i][0]:non_dups_left[i][0]+non_dups_left[i][2]])
      right.append(cv2.imread("right0.png")[non_dups_right[i][1]:non_dups_right[i][1]+non_dups_right[i][3], non_dups_right[i][0]:non_dups_right[i][0]+non_dups_right[i][2]])
   return non_dups_left, non_dups_right
   # After pair is sent -> pair = []
