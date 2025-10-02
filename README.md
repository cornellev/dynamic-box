# dynamic-box
3D point cloud segmentation algorithm for consecutive LiDAR scans using kd-tree-based euclidean cluster extraction.

## Euclidean Cluster Extraction ROS2 Package:
It is recommended that the ROS2-integrated Euclidean Cluster Extraction algorithm is run using ... Docker container, as the container contain all dependencies:

### ROS2 Package Architecture (Not Updated):
```
ROS2 distro=Humble
package_name=my_rosbag_reader
```

UniTree LiDAR scans have message type: ```PointCloud2``` and publish to the ```unilidar/cloud``` topic.

```my_rosbag_reader/my_rosbag_reader/live.py``` is a Subscriber node subscribed to the ```unilidar/cloud``` topic.

When LiDAR scan data is published, ```my_rosbag_reader/my_rosbag_reader/live.py```'s callback function, ```listener_callback```, will run the Euclidean Cluster Extraction algorithm and return an array containing cluster partitions of the point cloud. 

Centroids and maximum radius of the clusters are published as an array of ```Obstacle(s)``` to the ```/rslidar_obstacles``` message.

## Euclidean Cluster Extraction Implementation (Not Updated, Look At Recent Updates):
### Recent Updates:
|  Date     | Changelog / Update Notes |
|:----------|:-----------|
| 9/18/25   | - Added custom single obstacle ```Obstacle.msg``` and obstacle arrays ```Obstacles.msg``` messages to [cev_msgs](https://github.com/cornellev/cev_msgs). <br> - Now publishes ```Obstacle(s)``` to ```/rslidar_obstacles``` message. |
| 9/14/25   | - Clusters with seed starting points are grown in parallel, where each ```cluster_cpp.euclidean_cluster``` call is limited to only one cluster grown. <br> - New obstacles and outlier points that are not captured in past centroid seeds are still grown sequentially from one random seed, no constraints on ```MAX_CLUSTER_NUM```. <br> - Clustering on the very first LiDAR scan is used to initialize the first set of seeds and is not parallelized. |
| 9/13/25   | - Seeds act as starting points for cluster growing. For the centroid of past cluster, the closest point in the current scan to such centroid is initialized as a new seed. <br> - ```cluster_cpp.euclidean_cluster``` now takes in arguments ```seeds : Eigen::Vector4d``` and ```MAX_CLUSTER_NUM``` that represent fixed starting points and maximum number of clusters allowed to be grown, respectively. |
| 9/12/25   | - ```cluster.cpp``` contains C++ implementations of Euclidean Cluster Extraction function ```euclidean_cluster``` and the ```Node``` class, which contains ```make_kdtree, search_point``` and ```search_tree``` functions. <br> - To speedup clustering, the C++ implementation of Euclidean Cluster Extraction ```cluster_cpp.euclidean_cluster``` is now used in place of the Python version. |

### Euclidean Cluster Extraction Algorithm 

For each scan published, we first initialize the set of unexplored points ```Node.unexplored``` as being all 3D points in such input LiDAR scan ```cloud```. 

To optimize nearest neighbor search, a KD-Tree representation of ```cloud``` is used as input into the ```euclidean_cluster``` function, which implements Euclidean Cluster Extraction.

Euclidean Cluster Extraction follows these general steps:
1) We initially have an empty array of clusters ```C```. 
2) For cluster growing with seeds, seed $(x, y, z)$ is used as the starting point for growing a new cluster.

   Otherwise, we pop some arbitrary point $(x, y, z)$ from ```Node.unexplored``` that will be used as the first element in a stack of seeds ```stack``` for growing a new cluster ```c``` $\in$ ```C```.
3) Starting from $(x, y, z)$ and leveraging the KD-Tree representation, we recursively search for neighbors in ```Node.unexplored``` around $(x, y, z)$ and add them ```stack```. 
Nearest points of $(x, y, z)$ are found by traversing the KD-Tree until the leaf containing $(x, y, z)$ and its neighbors is reached. Then for arbitrary neighbor $(x', y', z')$, let us define the "neighbor" condition: $$ \bold{if} \text{ } \ell^2 \text{-distance between } (x, y, z) \text{ and } (x', y', z') < radius \text{, } \bold{then} \text{ add } (x', y', z') \text{ to } stack.$$
If the previous condition is satisfied, also add $(x', y', z')$ to cluster ```c```.
4) The recursion terminates when there are no more unexplored points that satisfy the "neighbor" condition, signalling that ```c``` has finished growing.
5) If there exists another point from ```Node.unexplored```, repeat steps 2 to 5 to recursively grow new clusters until all points from ```cloud``` belong to some cluster in ```C```. 
6) Before outliers are filtered out, Euclidean Cluster Extraction should return a partition of clusters {```c_1, ..., c_i```} s.t. $i \in$ [1, len(```cloud```)] and ```c_1``` $\cup$ ... $\cup$ ... ```c_i``` = ```C```.
7) To filter out clusters formed by sparse, outlier points with very few to no neighbors, keep only clusters ```c'``` in ```C``` where the number of points in ```c'``` are greater than hyperparameter ```MIN_CLUSTER_SIZE```. Where default ```MIN_CLUSTER_SIZE``` = 1.

NOTE: One critical assumption made for Euclidean Cluster Extraction is that input point clouds are relatively dense such that the maximum distance ```radius``` between two "adjacent" points in the same cluster is less than the minimum distance between two points belonging to different clusters. When this assumption does not hold, we can observe the algorithm merging clusters representing different objects together. 

For the task of object tracking over different frames, we want to be able to observe some consistency in the shapes and numbers of clusters across consecutve LiDAR scans. Given consecutive LiDAR scans, we cannot assume that the number of clusters returned by Euclidean Cluster Extraction remains constant, as it is common for objects to enter and exist the LiDAR's range. 

However, suppose we made the assumption that the time passed between two consecutive scans (```A, B```) is negligibly small enough that given:
- some object ```O```,
- cluster ```c_A``` corresponding to object ```O``` in scan ```A```,
- cluster ```c_B``` corresponding to object ```O``` in scan ```B```, 

there exists some spatial overlap between ```c_A``` and ```c_B``` s.t. for any other arbitary cluster ```c_B'``` in scan ```B```, the number of points in ```c_B``` that lie in ```c_A```'s boundary should be greater than the number of points in ```c_B'``` that lie in ```c_A```'s boundary. 

We could also claim that the $\ell^2$-distance between ```c_A``` and ```c_B```'s centroids are less than the $\ell^2$-distance between ```c_A``` and ```c_B'```.

Then, if we made the assumption that negligibly small amounts of time passed between two consecutive LiDAR scans, we could use the mentioned conditions to match previous and current clusters that correspond to the same object.

### On Parallelizing Euclidean Cluster Extraction

In order to parallelize the segmentation algorithm, Euclidean Cluster Extraction is split into sequential and parallel versions:

The initial LiDAR scan is ran on sequential Euclidean Cluster Extraction, where no assumptions of the structure of the point cloud is made, and $n$ clusters are allowed to be grown from some random starting point, which we call a seed.

For the $i^{th}$ LiDAR scan, parallel Euclidean Cluster Extraction takes the $n$ centroids of the clusters grown from the $(i-1)^{st}$ LiDAR scan. $n'$ seeds, where $n' \leq n$, are intialized as being the closest points in the $i^{th}$ scan to the $n$ past centroids. 
This is made under the assumption that translation between two consecutive LiDAR scans is negligibly small.

Then, $n'$ clusters are grown in parallel from these $n'$ seeds. Specifically, Euclidean Cluster Extraction with ```MAX_CLUSTER_NUMS = 1``` is run on each seed. We do not assume that the number of clusters in consecutive scans is constant, as objects moving in and out of the LiDAR's range will vary. As such, for remaining points in the $i^{th}$ LiDAR scan that have not been clustered, sequential Euclidean Cluster Extraction is ran on this subset of points.

In terms of speedup and optimization, the runtime of segmentation on fully parallelized Euclidean Cluster Extraction is constrained by the largest cluster, or the cluster with the most points. If there are no overlaps between two consecutive LiDAR scans, then only  sequential Euclidean Cluster Extraction will grow clusters.

## Installation:
### [Recommended] Building and Running the Provided Docker Image:
All Python and ROS2 dependencies in ```my_rosbag_reader``` will be automatically installed and built with the provided Docker image:
```
git clone --branch improvements git@github.com:cornellev/dynamic-box.git --single-branch
docker build -t dbimage .
docker run -it --name dbimage dbimage bash
```
After building the docker image ```dbimage```, directories should appear in a layout similar to:
```
/home/dev/ws/src/
├── ./
├── ../
├── dynamic-box/my_rosbag_reader/
│   ├── my_rosbag_reader/ 
│   │   ├── __init__.py
│   │   ├── requirements.txt
│   │   ├── cluster.cpp
│   │   ├── live.py
│   │   ├── setup.py        
│   │   └── test_cloud.pcap
│   ├── package.xml  
│   ├── resource/ 
│   │   └── my_rosbag_reader 
│   ├── setup.cfg
│   └── setup.py  
├── rslidar_msg/
└── rslidar_sdk/
```
Running the docker image will automatically build, but not source ```rslidar_msg, rslidar_sdk```, and ```my_rosbag_reader``` ROS2 packages.
Source these packages by going to ```/home/dev/ws/src``` and running:
```
colcon build --packages-select cev_msgs --cmake-clean-cache
source install/setup.bash
cd /dynamic-box/my_rosbag_reader
source install/setup.bash
cd /my_rosbag_reader
python3 setup.py build_ext --inplace
```

## Euclidean Cluster Extraction Important links:
[Fast Euclidean Cluster Extraction Using GPUs](https://www.jstage.jst.go.jp/article/jrobomech/32/3/32_548/_pdf)

## How Dynamic Box Works (NEED TO UPDATE):
ZED camera captures stereo pairs of left (Il) and right (Ir) images -> run through a 2D detector that generates 2D boxes on regions of interest (ROIs) in Il and Ir:
1) 2D detector that generates 2D boxes in Il and Ir:
   Given stereo image pair input: an Il, Ir pair from ZED -> identify left (l) and right (r) ROIs -> threshold -> aquire m, n ROIs in Il, Ir ->
   perform association with SSIM(l,r) for each ROI pair combination 
   (assume l, r are similar).
2) Box association algorithm matches object detections across both images,
    forms a box association pair if SSIM threshold satisfied.
3) Stereo regression: with left-right box association pairs, apply ROI
   Align -> concatenate l, r into two fully-connected layers + ReLU
   layer. 
4) Given left-right 2D boxes, perspective keypoint, regressed dim, generate
   3D box: look at ```3D box Estimation``` for reference on how we will use the 
   left and right 2D boxes to generate a 3D box. 

### Steps completed:
1) Used YOLOv3 to generate 2D bounding boxes for objects in stereo images taken by the ZED camera. Returns 2D bounding box information in the form of $(x_l,y_l,w,h)$.
2) Given derived $F$, and assuming that the left camera is placed at the origin, then let us define the left camera in a canonical form where the camera projection or intrinsic matrix is $K_L = [I|0]$, where $I$ is the identity matrix and $0$ is the zero vector. Then we can compute the epipole $e_R$ of the right image with $F^Te_R=0$, where $e_R$ is in the null space of $F$. Finally, we derive the right camera's projection matrix as $K_R=[[e_R]_xF+e_2v^T]$.

4) Use $F$, $K_R$ and $K_L$ to derive essential matrix $E={K_R^T}\cdot{F}\cdot{K_L}={[T_x]}\cdot{R}$, which is used to compute epipolar lines $l={E}\cdot{P}$ and $l^{'}={E^T}\cdot{P^{'}}$ with $P$ being a point on the left image and $P^{'}$ on the right image. The location of right $P^{'}$ in the left image is derived by rotating $P^{'}$ with rotation matrix $R$ and translating with translation matrix $T$ to get $P={R}\cdot{P^{'}}+T$.
   
3) Found the correspondence of each point on the left image to the right image, where given a point in the left image, the corresponding epipolar line equation in the right image was derived by [formula] and exhaustive block matching with sum of absolute differences (SAD) was conducted to derive the actual corresponding right point.

4) Given corresponding left and right points, we can convert 2D points in the images to 3D points by calculating the depth of that point from the camera. We use the formula $Z=\dfrac{{f}\cdot{B}}{d}$, where $Z$ is distance of the object from the camera, $f$ is focal length, $B=120mm$ (for ZED) is baseline, and $d$ is disparity (horizontal shift between left and right image). Given a point $P$ that corresponds in the left and right images, disparity $d=x_l-x_r$, where $x_l$ is the horizontal pixel distance of $P$ in the left image, and $x_r$ is the horizontal pixel distance of $P$ in the right image.
   
5) Estimate 3D coordinates with depth and ZED camera calibration information. The 3D position $(X,Y,Z)$ (a point on the epipolar plane $\pi$ which contains the baseline $B$) of the front face of the object is calculated by: $X={x-c_x}\cdot{\dfrac{Z}{f_x}}$, $Y={y-c_y}\cdot{\dfrac{Z}{f_y}}$, and $Z=depth(x,y)$ where $(x,y)$ is the pixel coordinates of the corners of the 2D bounding box, focal lengths of the left ZED camera $f_x=700.819$ and $f_y=700.819$.

6) For each object detected by YOLOv3, convert 2D points constrained in its corresponding bounding box into a 3D point cloud. Then find the eigenvalues and eigenvectors of the covariance matrix of the 3D dataset to construct a rotation matrix, which is used to derive yaw, pitch, and roll of the object. The dimensions of the 3D bounding box is constrained within the 2D bounding box, and orientation and pose are estimated using the calculated yaw, pitch, and roll.

### Currently working on:
1) Getting more accurate 3D bounding boxes that consider perspective, rotation of the object in 3D space, and its overall pose.




## Important links:

[An Introduction to 3D Computer Vision Techniques and Algorithms](https://ia801208.us.archive.org/12/items/an-introduction-to-3-d-computer-vision-techniques-and-algorithms-cyganek-siebert-2009-02-09/An%20Introduction%20to%203D%20Computer%20Vision%20Techniques%20and%20Algorithms%20%5BCyganek%20%26%20Siebert%202009-02-09%5D.pdf)

[ZED Calibration File](https://support.stereolabs.com/hc/en-us/articles/360007497173-What-is-the-calibration-file)

[CS231A Course Notes 3: Epipolar Geometry](https://web.stanford.edu/class/cs231a/course_notes/03-epipolar-geometry.pdf)

[Epipolar Geometry and the Fundamental Matrix](https://www.robots.ox.ac.uk/~vgg/hzbook/hzbook2/HZepipolar.pdf)

[Complex YOLO: YOLOv4 for 3D Object Detection](https://medium.com/@mohit_gaikwad/complex-yolo-yolov4-for-3d-object-detection-3c9746281cd2)

[Stereo R-CNN based 3D Object Detection](https://github.com/HKUST-Aerial-Robotics/Stereo-RCNN?tab=readme-ov-file)

[3D Reconstruction and Epipolar Geometry](https://github.com/laavanyebahl/3D-Reconstruction-and-Epipolar-Geometry)

[The 8-point algorithm](https://www.cs.cmu.edu/~16385/s17/Slides/12.4_8Point_Algorithm.pdf)

