# dynamic-box
3rd times the charm

## Update Notes:
The better dynamic-box.

## Installation:
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


## How This Works:
ZED camera captures stereo pairs of left (Il) and right (Ir) images -> run through a 2D detector that generates 2D boxes on regions of interest (ROIs) in Il and Ir:
1) 2D detector that generates 2D boxes in Il and Ir:
   Given stereo image pair input: an Il, Ir pair from ZED -> identify left (l) and right (r) ROIs -> threshold -> aquire m, n ROIs in Il, Ir ->
   perform association with SSIM(l,r) for each ROI pair combination 
   (assume l, r are similar).
2) Box association algorithm matches obejct detections across both images,
    forms a box association pair if SSIM threshold satisfied.
3) Stereo regression: with left-right box association pairs, apply ROI
   Align -> concatenate l, r into two fully-connected layers + ReLU
   layer. 
4) Given left-right 2D boxes, perspective keypoint, regressed dim, generate
   3D box: look at ```3D box Estimation``` for reference on how we will use the 
   left and right 2D boxes to generate a 3D box. 

### Steps completed:
1) Implemented YOLOv3 and YOLOv11 2D bounding box generators for objects in stereo images. Returns bounding box information in the form of $(x_l,y_l,w,h)$.
2) Used SIFT to 8 corresponding keypoints on the left and right stereo images, then used these 8 keypoints (left $P_i=(x_i,y_i,1)$, right $P_i^{'}=(x_i^{'},y_i^{'},1)$ to derive the fundamental matrix $F$, by finding coefficients $f_{11}$, $f_{12}$, $f_{13}$, $f_{21}$, $f_{22}$, $f_{23}$, $f_{31}$, $f_{32}$, $f_{33}$ that satisfy the epipolar constraint ${P_i^T}Fp_i^{'}=0$.
   
   _Eventually when I start working with the ZED, I'll also derive the Fundamental matrix_ $F=K^{'}[T_x]RK^{-1}$ _using the intrisinc and extrinsic parameters given by the calibration file to test accuracy of_ $F$. 

### Currently working on:
1) Given derived $F$, and assuming that the left camera is placed at the origin, then let us define the left camera in a canonical form where the camera projection or intrinsic matrix is $K_L = [I|0]$, where $I$ is the identity matrix and $0$ is the zero vector. Then we can compute the epipole $e_R$ of the right image with $F^Te_R=0$, where $e_R$ is in the null space of $F$. Finally, we derive the right camera's projection matrix as $K_R=[[e_R]_xF+e_2v^T]$.
2) Use $F$, $K_R$ and $K_L$ to derive essential matrix $E={K_R^T}\cdot{F}\cdot{K_L}={[T_x]}\cdot{R}$, which is used to compute epipolar lines $l={E}\cdot{P}$ and $l^{'}={E^T}\cdot{P^{'}}$ with $P$ being a point on the left image and $P^{'}$ on the right image. The location of right $P^{'}$ in the left image is derived by rotating $P^{'}$ with rotation matrix $R$ and translating with translation matrix $T$ to get $P={R}\cdot{P^{'}}+T$.
   
   _With ZED calibration file, also find instrinsic matrices_ $K_R$ _and_ $K_L$.
3) __FIGURE OUT THE MATH FOR THIS:__ Rectify the stereo images with epipolar geometry and epipolar lines $l$ so that corresponding points between the two images lie on the same horizontal line (epipolar line).

   _The rectification problem setup: we compute two homographies_ $H_1$ _and_ $H_2$ _that we can apply to the image planes to make the resulting planes parallel. This would make the epipole_ $e$ _at infinity._
4) Derive depth of objects (most likely the corners of each 2D bounding box) in the rectified stereo images. Use the formula $Z=\dfrac{{f}\cdot{B}}{d}$, where $Z$ is distance of the object from the camera, $f$ is focal length, $B=120mm$ (for ZED) is baseline, and $d$ is disparity (horizontal shift between left and right image). Given a point $P$ that corresponds in the left and right images, disparity $d=x_l-x_r$, where $x_l$ is the horizontal pixel distance of $P$ in the left image, and $x_r$ is the horizontal pixel distance of $P$ in the right image.

   _Only need to calculate horizontal disparity between stereo images have been rectified, so we ignore vertical disparity._
5) Estimate 3D coordinates with depth and ZED camera calibration information. The 3D position $(X,Y,Z)$ (a point on the epipolar plane $\pi$ which contains the baseline $B$) of the front face of the object is calculated by: $X={x-c_x}\cdot{\dfrac{Z}{f_x}}$, $Y={y-c_y}\cdot{\dfrac{Z}{f_y}}$, and $Z=depth(x,y)$ where $(x,y)$ is the pixel coordinates of the corners of the 2D bounding box, focal lengths of the left ZED camera $f_x=700.819$ and $f_y=700.819$, and principle point or optical center coordinates of the left ZED camera $c_x=665.465$ and $c_y=371.953$. The back face of the object is estimated with the object's depth and width.

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

