# dynamic-box
3rd times the charm

## Update Notes:
The better dynamic-box.

## Installation:
### OpenCV:
Run:
```
pip3 install opencv-python
```

### Conda: 
https://docs.anaconda.com/anaconda/install/linux/:
Download in Ubuntu home directory:
``` 
Anaconda3-2024.06-1-Linux-x86_64.sh from https://repo.anaconda.com/archive/
```

Run:
```
curl -O https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
bash ~/<Wherever you downloaded it>/Anaconda3-2024.06-1-Linux-x86_64.sh
```

Anaconda is now downloaded in ``` /home/<USER>/anaconda3 ``` in Ubuntu

Refresh terminal: ``` source ~/.bashrc ```

### Activate conda environment:
``` python
conda create -n env_stereo python=3.10
conda activate env_stereo
conda install pytorch
conda install cuda80 -c pytorch
conda install torchvision -c pytorch
```


## How This Works (NEED TO UPDATE):
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
   
## Eventual ROS2 Package Implementation:
...

## Important links:
[An Introduction to 3D Computer Vision Techniques and Algorithms](https://ia801208.us.archive.org/12/items/an-introduction-to-3-d-computer-vision-techniques-and-algorithms-cyganek-siebert-2009-02-09/An%20Introduction%20to%203D%20Computer%20Vision%20Techniques%20and%20Algorithms%20%5BCyganek%20%26%20Siebert%202009-02-09%5D.pdf)

[ZED Calibration File](https://support.stereolabs.com/hc/en-us/articles/360007497173-What-is-the-calibration-file)

[CS231A Course Notes 3: Epipolar Geometry](https://web.stanford.edu/class/cs231a/course_notes/03-epipolar-geometry.pdf)

[Epipolar Geometry and the Fundamental Matrix](https://www.robots.ox.ac.uk/~vgg/hzbook/hzbook2/HZepipolar.pdf)

[Complex YOLO: YOLOv4 for 3D Object Detection](https://medium.com/@mohit_gaikwad/complex-yolo-yolov4-for-3d-object-detection-3c9746281cd2)

[Stereo R-CNN based 3D Object Detection](https://github.com/HKUST-Aerial-Robotics/Stereo-RCNN?tab=readme-ov-file)

[3D Reconstruction and Epipolar Geometry](https://github.com/laavanyebahl/3D-Reconstruction-and-Epipolar-Geometry)

[The 8-point algorithm](https://www.cs.cmu.edu/~16385/s17/Slides/12.4_8Point_Algorithm.pdf)

